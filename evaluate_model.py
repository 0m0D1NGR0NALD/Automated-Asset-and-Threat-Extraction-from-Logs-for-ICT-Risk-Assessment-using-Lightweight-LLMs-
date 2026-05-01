import argparse
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, mean_absolute_error
import re

def normalize_text(text: str) -> str:
    """Normalize raw log for matching."""
    # Remove timestamp-like prefixes if any
    # Keep only the request line
    return text.strip()

def evaluate(risk_csv, ground_truth_csv):
    risk_df = pd.read_csv(risk_csv)
    gt_df = pd.read_csv(ground_truth_csv)

    # Create a mapping from normalized raw log to ground truth
    gt_map = {}
    for _, row in gt_df.iterrows():
        norm = normalize_text(row['raw_log'])
        gt_map[norm] = (row['true_classification'], row['true_threat'])

    y_true_class = []
    y_pred_class = []
    y_true_risk = []   # optional
    y_pred_risk = []

    for _, risk_row in risk_df.iterrows():
        raw_preview = risk_row['raw_log_preview']
        # Find best match in ground truth (partial match because preview is truncated)
        matched = None
        for gt_log in gt_map.keys():
            if raw_preview in gt_log or gt_log in raw_preview:
                matched = gt_map[gt_log]
                break
        if not matched:
            continue   # skip if no match found (should not happen for the sample)

        true_class, true_threat = matched
        pred_threat = risk_row['extracted_threat']

        # Convert to binary: attack vs benign
        true_binary = 1 if true_class == 'anomalous' else 0
        pred_binary = 1 if pred_threat != 'benign' and pred_threat != 'info_leak' else 0
        # Note: info_leak is considered benign for CSIC normal requests

        y_true_class.append(true_binary)
        y_pred_class.append(pred_binary)

        # Risk scores if we have true risk values (not available in simple ground truth)
        # We'll skip MAE unless you provide a column 'true_risk_score'

    if not y_true_class:
        print("No matching entries found. Check the raw_log_preview matching.")
        return

    prec, rec, f1, _ = precision_recall_fscore_support(y_true_class, y_pred_class, average='binary')
    acc = accuracy_score(y_true_class, y_pred_class)

    print(f"\n=== Evaluation for {risk_csv} ===")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1-score:  {f1:.3f}")

    # Optional: write confusion matrix
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_true_class, y_pred_class).ravel()
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--risk_csv', required=True, help='Risk register CSV from model')
    parser.add_argument('--ground_truth', required=True, help='Ground truth CSV (raw_log, true_classification, true_threat)')
    args = parser.parse_args()
    evaluate(args.risk_csv, args.ground_truth)