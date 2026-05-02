# compare_models_uniform.py
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics_for_model(risk_csv, ground_truth_csv):
    gt_df = pd.read_csv(ground_truth_csv)
    gt_df['preview'] = gt_df['raw_log'].apply(lambda x: x[:100])
    gt_map = dict(zip(gt_df['preview'], gt_df['classification']))

    risk_df = pd.read_csv(risk_csv)

    y_true = []
    y_pred = []
    for _, row in risk_df.iterrows():
        preview = row['raw_log_preview']
        if preview not in gt_map:
            continue
        true_label = gt_map[preview]
        threat = row['extracted_threat'].lower()
        if threat in ['benign', 'info_leak', 'normal']:
            pred_binary = 0
        else:
            pred_binary = 1
        y_true.append(1 if true_label == 'anomalous' else 0)
        y_pred.append(pred_binary)

    if not y_true:
        return None

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', required=True,
                        help='List of model names and CSV paths in format name:path (e.g., SmolLM2:smol_results.csv)')
    parser.add_argument('--ground_truth', required=True)
    parser.add_argument('--output', default='results/comparison_report.txt')
    args = parser.parse_args()

    # Parse model spec
    models = {}
    for spec in args.models:
        name, path = spec.split(':')
        models[name] = path

    print("Loading ground truth...")
    results = {}
    for name, path in models.items():
        print(f"Evaluating {name}...")
        metrics = compute_metrics_for_model(path, args.ground_truth)
        if metrics:
            results[name] = metrics
        else:
            print(f"  Warning: No matching entries for {name}")

    # Generate report
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("MODEL COMPARISON REPORT")
    report_lines.append("=" * 70)
    report_lines.append(f"\n{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    report_lines.append("-" * 70)

    for name in sorted(results.keys()):
        m = results[name]
        report_lines.append(f"{name:<20} {m['accuracy']:.3f}      {m['precision']:.3f}      {m['recall']:.3f}      {m['f1']:.3f}")

    # Print and save
    for line in report_lines:
        print(line)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"\nReport saved to {args.output}")

if __name__ == "__main__":
    main()