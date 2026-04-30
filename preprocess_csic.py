"""
Preprocess CSIC 2010 HTTP dataset CSV into a plain text file.
Each output line is: "METHOD URL [POST body]"
Suitable for feeding into the ICT Risk Assessment Tool.
"""

import csv
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Convert CSIC 2010 CSV to raw HTTP request lines."
    )
    parser.add_argument(
        "-i", "--input",
        default="data/datasets/csic_database.csv",
        help="Path to input CSV file (default: data/datasets/csic_database.csv)"
    )
    parser.add_argument(
        "-o", "--output",
        default="data/datasets/csic_requests.txt",
        help="Path to output text file (default: data/datasets/csic_requests.txt)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    line_count = 0
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:

        reader = csv.DictReader(infile)
        for row in reader:
            method = row.get('Method', '').strip()
            url = row.get('URL', '').strip()
            content = row.get('content', '').strip()

            # Build a raw log line similar to an Apache log entry
            if method and url:
                if content:
                    raw_line = f"{method} {url} {content}"
                else:
                    raw_line = f"{method} {url}"
            else:
                # Fallback: use the whole row as a string (should not happen)
                raw_line = ' '.join(str(v) for v in row.values())

            outfile.write(raw_line + '\n')
            line_count += 1

    print(f"Successfully processed {line_count} lines.")
    print(f"Output written to: {output_path}")
    return 0

if __name__ == "__main__":
    exit(main())