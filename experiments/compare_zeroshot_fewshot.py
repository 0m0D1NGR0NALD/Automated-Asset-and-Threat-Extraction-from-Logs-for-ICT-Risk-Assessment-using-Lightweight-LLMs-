import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def evaluate(risk_csv, ground_truth_csv, model_name):
    gt = pd.read_csv(ground_truth_csv)
    gt['preview'] = gt['raw_log'].apply(lambda x: x[:100])
    gt_map = dict(zip(gt['preview'], gt['classification']))
    df = pd.read_csv(risk_csv)
    y_true, y_pred = [], []
    for _, row in df.iterrows():
        preview = row['raw_log_preview']
        if preview not in gt_map:
            continue
        true = 1 if gt_map[preview] == 'Anomalous' else 0
        threat = row['extracted_threat'].lower()
        pred = 1 if threat not in ['benign', 'info_leak', 'normal'] else 0
        y_true.append(true)
        y_pred.append(pred)
    if not y_true:
        return None
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1}

def main():
    ground_truth = "data/datasets/ground_truth_sample.csv"
    zero_shot_files = {
        "SmolLM2": "results/smolLM2_results.csv",
        "Qwen": "results/qwen_results.csv",
        "TinyLlama": "results/tinyllama_results.csv"
    }
    few_shot_files = {
        "SmolLM2": "results/smolLM2_fewshot.csv",
        "Qwen": "results/qwen_fewshot.csv",
        "TinyLlama": "results/tinyllama_fewshot.csv"
    }

    print("\n" + "="*70)
    print("Zero‑Shot vs Few‑Shot Performance Comparison")
    print("="*70)
    print(f"{'Model':<12} {'Technique':<12} {'Acc':<6} {'Prec':<6} {'Rec':<6} {'F1':<6}")
    print("-"*70)

    for model in zero_shot_files.keys():
        zs = evaluate(zero_shot_files[model], ground_truth, model)
        fs = evaluate(few_shot_files[model], ground_truth, model)
        if zs:
            print(f"{model:<12} Zero-shot    {zs['acc']:.3f}  {zs['prec']:.3f}  {zs['rec']:.3f}  {zs['f1']:.3f}")
        if fs:
            print(f"{model:<12} Few-shot    {fs['acc']:.3f}  {fs['prec']:.3f}  {fs['rec']:.3f}  {fs['f1']:.3f}")
        print("-"*70)

if __name__ == "__main__":
    main()