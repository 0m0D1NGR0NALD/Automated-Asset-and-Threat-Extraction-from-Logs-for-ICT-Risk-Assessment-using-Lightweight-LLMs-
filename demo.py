"""
Demo: runs TinyLlama (optimized for hardware constraints) on a given log file.
Usage: python demo.py <input_file> [output_file]
If output_file is omitted, defaults to "demo_results.csv"

Optimized for memory usage with batch processing and memory cleanup
"""
import sys
import subprocess
import tempfile
import time
import re
import gc
import os
import psutil
from pathlib import Path
from functools import wraps

# Memory optimization environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['OMP_NUM_THREADS'] = '2'  # Limit CPU threads
os.environ['MALLOC_TRIM_THRESHOLD_'] = '100000'

def get_memory_usage():
    """Get current memory usage in MB."""
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return None

def log_memory(step=""):
    """Log memory usage at different steps."""
    memory_mb = get_memory_usage()
    if memory_mb:
        print(f"[Memory] {step}: {memory_mb:.1f} MB")
    return memory_mb

def timeit(func):
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"[Timer] {func.__name__} took {elapsed:.2f} seconds")
        return result
    return wrapper

def chunked_processing(lines, chunk_size=50):
    """Generator that yields chunks of lines for batch processing."""
    for i in range(0, len(lines), chunk_size):
        yield lines[i:i + chunk_size]

def filter_comment_lines(input_file):
    """Filter out comment lines and process in chunks to save memory."""
    clean_lines = []
    original_size = 0
    
    print(f"Reading and filtering comments from {input_file}...")
    log_memory("Before reading file")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        # Get file size for progress reporting
        f.seek(0, 2)
        file_size = f.tell()
        f.seek(0)
        
        for line in f:
            original_size += len(line)
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                clean_lines.append(stripped)
            
            # Periodic memory check
            if len(clean_lines) % 1000 == 0:
                log_memory(f"After {len(clean_lines)} lines")
    
    log_memory("After filtering")
    return clean_lines

@timeit
def process_in_memory(lines, output_file, model_name="tinyllama", few_shot=True, chunk_size=25):
    """
    Process logs with memory optimization using subprocess with chunking.
    This avoids loading everything into the model at once.
    """
    total_lines = len(lines)
    print(f"Total valid log lines to process: {total_lines}")
    
    if total_lines == 0:
        print("Error: No valid log lines found after removing comments.")
        return False
    
    # Create temporary directory for chunk files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        chunk_files = []
        
        # Split into chunks and process each
        for chunk_idx, chunk in enumerate(chunked_processing(lines, chunk_size)):
            chunk_file = temp_dir_path / f"chunk_{chunk_idx:04d}.txt"
            with open(chunk_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(chunk))
            chunk_files.append(chunk_file)
            print(f"Created chunk {chunk_idx + 1}/{len(list(chunked_processing(lines, chunk_size)))} "
                  f"({len(chunk)} lines)")
        
        # Process each chunk
        all_success = True
        chunk_outputs = []
        
        for chunk_idx, chunk_file in enumerate(chunk_files):
            print(f"Processing chunk {chunk_idx + 1}/{len(chunk_files)}")
            
            # Create chunk-specific output file
            chunk_output = temp_dir_path / f"output_{chunk_idx:04d}.csv"
            
            # Build command with memory optimizations
            python_exe = sys.executable
            
            cmd = [
                python_exe, "-m", "src.main",
                "--model", model_name,
                "-i", str(chunk_file),
                "-o", str(chunk_output)
            ]
            
            if few_shot:
                cmd.append("--few-shot")
            
            # Add memory optimization flags if supported by your model
            # (These would need to be implemented in your model code)
            
            print(f"Running: {' '.join(cmd)}")
            log_memory(f"Before processing chunk {chunk_idx + 1}")
            
            # Run subprocess with memory limits
            success = run_with_memory_limit(cmd, chunk_idx + 1, len(chunk_files))
            
            if success and chunk_output.exists():
                chunk_outputs.append(chunk_output)
            else:
                all_success = False
                print(f"Error processing chunk {chunk_idx + 1}")
            
            # Force garbage collection between chunks
            gc.collect()
            log_memory(f"After processing chunk {chunk_idx + 1}")
            
            # Small delay to allow system to recover
            time.sleep(1)
        
        # Merge all chunk outputs if processing was successful
        if all_success and chunk_outputs:
            print(f"\n{'='*60}")
            print("Merging results from all chunks...")
            merge_chunk_outputs(chunk_outputs, output_file)
            return True
        else:
            print("Error: Some chunks failed to process")
            return False

def run_with_memory_limit(cmd, chunk_num, total_chunks):
    """Run subprocess with memory monitoring and cleanup."""
    try:
        # Start subprocess with reduced buffer size
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True,
            env=get_optimized_env()  # Use optimized environment
        )
        
        # Pattern to track progress
        pattern = re.compile(r'Processed\s+(\d+)/(\d+)')
        processed = 0
        total = 0
        start_time = time.time()
        
        for line in process.stdout:
            # Print the line (but maybe filter some noise)
            if not line.startswith('[Memory]'):  # Filter our own memory logs
                print(line, end='')
            
            match = pattern.search(line)
            if match:
                processed = int(match.group(1))
                total = int(match.group(2))
                
                # Show progress every 10%
                if processed % max(1, total // 10) == 0:
                    elapsed = time.time() - start_time
                    if processed > 0:
                        avg_time = elapsed / processed
                        eta_seconds = avg_time * (total - processed)
                        eta_minutes = eta_seconds / 60
                        progress_pct = (processed / total) * 100
                        print(f"  Progress: {progress_pct:.1f}% | "
                              f"ETA: {eta_minutes:.1f} min | "
                              f"Memory: {get_memory_usage():.1f} MB")
        
        return_code = process.wait()
        
        if return_code != 0:
            print(f"Error: Process exited with code {return_code}")
            return False
        
        print(f"  Completed chunk {chunk_num}/{total_chunks}")
        return True
        
    except Exception as e:
        print(f"Error running subprocess: {e}")
        return False
    finally:
        # Ensure cleanup
        gc.collect()

def get_optimized_env():
    """Get optimized environment variables for memory efficiency."""
    env = os.environ.copy()
    
    # PyTorch optimizations
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    env['OMP_NUM_THREADS'] = '2'
    env['MKL_NUM_THREADS'] = '2'
    
    # Memory optimization
    env['MALLOC_TRIM_THRESHOLD_'] = '100000'
    env['MALLOC_ARENA_MAX'] = '2'
    
    # Disable some features to save memory
    env['TRANSFORMERS_OFFLINE'] = '1'  # If models are cached locally
    
    # Reduce logging verbosity
    env['TRANSFORMERS_VERBOSITY'] = 'error'
    
    return env

def merge_chunk_outputs(chunk_files, output_file):
    """Merge CSV outputs from multiple chunks."""
    import pandas as pd
    
    all_dfs = []
    for chunk_file in chunk_files:
        try:
            df = pd.read_csv(chunk_file)
            all_dfs.append(df)
            print(f"  Loaded {len(df)} results from {chunk_file.name}")
        except Exception as e:
            print(f"  Error reading {chunk_file.name}: {e}")
    
    if not all_dfs:
        print("Error: No data to merge")
        return
    
    # Concatenate all dataframes
    merged_df = pd.concat(all_dfs, ignore_index=True)
    
    # Save merged results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    
    print(f"\nSuccessfully merged {len(all_dfs)} chunks")
    print(f"Total results: {len(merged_df)} rows")
    print(f"Results saved to: {output_file}")
    
    # Clean up
    del all_dfs
    del merged_df
    gc.collect()

@timeit
def process_streaming(input_file, output_file, model_name="tinyllama", few_shot=True):
    """
    Alternative: Process with streaming to minimize memory usage.
    This processes line by line without chunking.
    """
    print("Using streaming processing mode (lowest memory usage)...")
    
    # Create temporary file for filtered lines
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', 
                                     suffix='.txt', delete=False) as tmp:
        tmp_path = tmp.name
        
        # Stream filtering - write filtered lines directly to temp file
        filtered_count = 0
        with open(input_file, 'r', encoding='utf-8') as infile:
            for line in infile:
                stripped = line.strip()
                if stripped and not stripped.startswith('#'):
                    tmp.write(stripped + '\n')
                    filtered_count += 1
                    
                    # Periodic flush to disk
                    if filtered_count % 1000 == 0:
                        tmp.flush()
        
        print(f"Filtered {filtered_count} valid log lines")
    
    if filtered_count == 0:
        print("Error: No valid log lines found")
        Path(tmp_path).unlink(missing_ok=True)
        return False
    
    # Run the model on the filtered file
    python_exe = sys.executable
    cmd = [
        python_exe, "-m", "src.main",
        "--model", model_name,
        "-i", tmp_path,
        "-o", output_file
    ]
    
    if few_shot:
        cmd.append("--few-shot")
    
    print(f"Running: {' '.join(cmd)}")
    
    # Run with memory monitoring
    success = run_with_memory_limit(cmd, 1, 1)
    
    # Clean up temp file
    Path(tmp_path).unlink(missing_ok=True)
    
    return success

def main():
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python demo.py <input_file> [output_file] [--streaming] [--chunk-size N]")
        print("Options:")
        print("  --streaming    Use streaming mode (lowest memory, but may be slower)")
        print("  --chunk-size N Set chunk size for batch processing (default: 25)")
        print("  --no-few-shot  Disable few-shot prompting")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = "results/demo_results.csv"
    use_streaming = False
    chunk_size = 25
    few_shot = True
    
    # Parse additional arguments
    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--streaming":
            use_streaming = True
        elif arg == "--chunk-size" and i + 1 < len(sys.argv):
            chunk_size = int(sys.argv[i + 1])
            i += 1
        elif arg == "--no-few-shot":
            few_shot = False
        else:
            output_file = arg
        i += 1
    
    # Check if input file exists
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    
    print("ICT Risk Assessment Demo - Memory Optimized")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Model: tinyllama")
    print(f"Few-shot: {few_shot}")
    print(f"Mode: {'Streaming' if use_streaming else f'Chunked (size={chunk_size})'}")
    
    # Log initial memory
    log_memory("Initial")
    
    # Filter comment lines
    clean_lines = filter_comment_lines(input_file)
    
    if not clean_lines:
        print("Error: No valid log lines found after removing comments.")
        sys.exit(1)
    
    print(f"Found {len(clean_lines)} valid log lines")
    
    # Choose processing method
    start_time = time.time()
    
    if use_streaming:
        success = process_streaming(input_file, output_file, "tinyllama", few_shot)
    else:
        success = process_in_memory(clean_lines, output_file, "tinyllama", few_shot, chunk_size)
    
    elapsed_time = time.time() - start_time
    
    # Final memory check
    log_memory("Final")
    
    if success:
        print(f"Processing complete!")
        print(f"Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        print(f"Results saved to: {output_file}")
        
        # Show result summary
        try:
            import pandas as pd
            df = pd.read_csv(output_file)
            print(f"\nResult Summary:")
            print(f"  Total entries: {len(df)}")
            if 'Risk' in df.columns:
                high_risk = len(df[df['Risk'] >= 13])
                print(f"  High/Critical risk: {high_risk}")
            if 'Confidence' in df.columns:
                print(f"  Avg confidence: {df['Confidence'].mean():.3f}")
        except:
            pass
    else:
        print("Processing failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()