"""
Demo: runs the best model (Qwen2.5-3B + few‑shot) on a given log file.
Usage: python demo.py <input_file> [output_file]
If output_file is omitted, defaults to "demo_results.csv"
"""
import sys
import subprocess
import tempfile
import time
import re
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: python demo.py <input_file> [output_file]")
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "results/demo_results.csv"

    # Filter out comment lines in log (starting with '#')
    clean_lines = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                clean_lines.append(stripped)

    if not clean_lines:
        print("Error: No valid log lines found after removing comments.")
        sys.exit(1)

    # Write to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.txt', delete=False) as tmp:
        tmp.write('\n'.join(clean_lines))
        tmp_path = tmp.name

    print(f"Filtered input: {len(clean_lines)} log lines (comments removed)")

    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    
    # Use the same Python interpreter that is running this script
    python_exe = sys.executable

    cmd = [
        python_exe, "-m", "src.main",
        "--model", "qwen",
        "--few-shot",
        "-i", tmp_path,
        "-o", output_file
    ]
    print(f"Running: {' '.join(cmd)}")
    print("\nProcessing logs... (ETA shown below)\n")

    start_time = time.time()
    processed = 0
    total = len(clean_lines)  # initial total from filtered lines
    pattern = re.compile(r'Processed\s+(\d+)/(\d+)')

    # Start the subprocess with real‑time output capture
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )

    for line in process.stdout:
        # Print the original line (so you still see the tool's logs)
        print(line, end='')
        match = pattern.search(line)
        if match:
            processed = int(match.group(1))
            total_from_log = int(match.group(2))
            if total_from_log != total:
                total = total_from_log  # use the tool's total if different
            elapsed = time.time() - start_time
            if processed > 0:
                avg_time = elapsed / processed
                eta_seconds = avg_time * (total - processed)
                eta_minutes = eta_seconds / 60
                print(f"  >>> Progress: {processed}/{total} | ETA: {eta_minutes:.1f} min", flush=True)

    # Wait for the process to finish
    return_code = process.wait()
    if return_code != 0:
        print(f"Error: Tool exited with code {return_code}")
        sys.exit(1)

    print(f"\nRisk assessment results saved to {output_file}")

    # Remove the temporary file
    Path(tmp_path).unlink(missing_ok=True)

if __name__ == "__main__":
    main()