import csv
import pandas as pd
from datetime import datetime

class CSVGenerator:
    @staticmethod
    def generate(entries, output_path):
        """
        entries: list of dicts with keys:
          timestamp, raw_preview, asset, threat, likelihood, impact, risk, confidence, requires_review
        """
        fieldnames = [
            'timestamp', 'raw_log_preview', 'extracted_asset', 'extracted_threat',
            'likelihood_score', 'impact_score', 'risk_score',
            'confidence', 'requires_review', 'human_override'
        ]
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for entry in entries:
                writer.writerow(entry)
        print(f"Risk register saved to {output_path}")