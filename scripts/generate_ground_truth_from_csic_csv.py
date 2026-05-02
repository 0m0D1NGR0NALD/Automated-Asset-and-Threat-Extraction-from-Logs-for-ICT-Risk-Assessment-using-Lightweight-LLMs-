import csv
import pandas as pd
from pathlib import Path

def main():
    input_csv = "data/datasets/csic_database.csv"
    output_ground_truth = "data/datasets/ground_truth_sample.csv"
    output_raw_text = "data/datasets/csic_database_sample.txt"

    # Read original CSV
    df = pd.read_csv(input_csv, encoding='utf-8', on_bad_lines='skip')
    
    # Reconstruct raw log line from Method, URL, and content
    def make_raw(row):
        method = str(row['Method']).strip() if pd.notna(row['Method']) else ''
        url = str(row['URL']).strip() if pd.notna(row['URL']) else ''
        content = str(row['content']).strip() if pd.notna(row['content']) else ''
        if method and url:
            if content:
                return f"{method} {url} {content}"
            else:
                return f"{method} {url}"
        return ''
    
    df['raw_log'] = df.apply(make_raw, axis=1)
    df = df[df['raw_log'] != '']  # remove empty rows
    
    # Split into Normal and Anomalous
    normal = df[df['classification'] == 'Normal']
    anomalous = df[df['classification'] == 'Anomalous']
    
    # Take 50 from each (or fewer if not enough)
    n_normal = min(50, len(normal))
    n_anomalous = min(50, len(anomalous))
    sample_normal = normal.sample(n_normal, random_state=42)
    sample_anomalous = anomalous.sample(n_anomalous, random_state=42)
    
    sample = pd.concat([sample_normal, sample_anomalous]).sample(frac=1, random_state=42)
    
    # Save ground truth CSV
    sample[['raw_log', 'classification']].to_csv(output_ground_truth, index=False)
    
    # Save raw text file for the tool to process
    sample['raw_log'].to_csv(output_raw_text, index=False, header=False)
    
    print(f"Created ground truth: {output_ground_truth} ({len(sample)} entries)")
    print(f"Created raw text input: {output_raw_text}")

if __name__ == "__main__":
    main()