import argparse
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--risk_csv', required=True, help='Risk register CSV from any model')
    parser.add_argument('--ground_truth', required=True, help='Ground truth CSV with raw_log and classification')
    parser.add_argument('--model_name', default='Model', help='Name for display')
    args = parser.parse_args()

    # Load ground truth
    gt_df = pd.read_csv(args.ground_truth)
    # Create a mapping from raw log preview to label
    gt_df['preview'] = gt_df['raw_log'].apply(lambda x: x[:100])
    gt_map = dict(zip(gt_df['preview'], gt_df['classification']))

    # Load risk register
    risk_df = pd.read_csv(args.risk_csv)

    # Align predictions with ground truth
    y_true = []
    y_pred = []
    for _, row in risk_df.iterrows():
        preview = row['raw_log_preview']
        if preview not in gt_map:
            continue
        true_label = gt_map[preview]
        # Map model's threat output to binary: anomalous if threat not in benign/info_leak
        threat = row['extracted_threat'].lower()
        if threat in ['benign', 'info_leak', 'normal']:
            pred_binary = 0
        else:
            pred_binary = 1
        y_true.append(1 if true_label == 'anomalous' else 0)
        y_pred.append(pred_binary)

    if not y_true:
        print("No matching entries found. Check raw_log_preview alignment.")
        return

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

    print(f"\n=== {args.model_name} Evaluation ===")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1-score:  {f1:.3f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['normal', 'anomalous']))

if __name__ == "__main__":
    main()