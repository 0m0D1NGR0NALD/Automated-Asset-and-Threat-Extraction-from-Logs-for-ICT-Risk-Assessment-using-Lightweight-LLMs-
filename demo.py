"""
Demo: runs the best model (Qwen2.5-3B + few‑shot) on a given log file.
Usage: python demo.py <input_file> [output_file]
If output_file is omitted, defaults to "demo_results.csv"
"""
import sys
import subprocess
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: python demo.py <input_file> [output_file]")
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "demo_results.csv"

    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)

    cmd = [
        "python", "-m", "src.main",
        "--model", "qwen",
        "--few-shot",
        "-i", input_file,
        "-o", output_file
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"\nRisk assessment results saved to {output_file}")

if __name__ == "__main__":
    main()