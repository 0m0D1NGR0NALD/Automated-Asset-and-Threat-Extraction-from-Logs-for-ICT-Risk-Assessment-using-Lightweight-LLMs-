import re
import csv
from pathlib import Path

ATTACK_PATTERNS = [
    r'waitfor\s+delay',           # SQL time-based
    r'--',                         # SQL comment
    r"' OR '1'='1",                # SQL tautology
    r'<script',                    # XSS
    r'javascript:',                # XSS
    r'on\w+\s*=',                  # event handler XSS
    r'\.\./',                      # path traversal
    r'%2e%2e%2f',                 # URL encoded ../
    r'%27',                        # URL encoded '
    r'%22',                        # URL encoded "
    r'%3cscript%3e',              # URL encoded <script>
    r'alert\(',                    # JS alert
    r'style=.*url\(javascript',    # CSS injection
    r'\.\.%2F',                    # another traversal
]

def is_anomalous(line: str) -> bool:
    line_lower = line.lower()
    for pat in ATTACK_PATTERNS:
        if re.search(pat, line_lower):
            return True
    return False

def main():
    input_file = "data/datasets/csic_database_sample.txt"
    output_file = "data/datasets/ground_truth_sample.csv"

    with open(input_file, 'r', encoding='utf-8') as fin:
        lines = [l.strip() for l in fin if l.strip()]

    with open(output_file, 'w', encoding='utf-8', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerow(["raw_log", "true_classification", "true_threat"])
        for line in lines:
            if is_anomalous(line):
                # Heuristic: pick a threat type – only for illustration
                # In real evaluation, you would use the original dataset's labels.
                threat = "attack"
            else:
                threat = "benign"
            writer.writerow([line, "anomalous" if threat == "attack" else "normal", threat])

    print(f"Ground truth saved to {output_file}")

if __name__ == "__main__":
    main()