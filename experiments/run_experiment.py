import argparse
import pandas as pd
from src.extractor.distilroberta_extractor import DistilRoBERTaExtractor
from src.risk.risk_scorer import RiskScorer
from src.utils.config import Config
from experiments.metrics import MetricsCalculator
from src.parser.log_preprocessor import LogPreprocessor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labeled', required=True, help='CSV with columns: raw_log, true_asset, true_threat, true_risk_score')
    args = parser.parse_args()
    
    df = pd.read_csv(args.labeled)
    config = Config()
    scorer = RiskScorer(config)
    extractor = DistilRoBERTaExtractor()
    
    pred_assets = []
    pred_threats = []
    pred_risks = []
    
    for _, row in df.iterrows():
        cleaned, _, _ = LogPreprocessor.clean_line(row['raw_log'])
        if not cleaned:
            continue
        extr = extractor.extract(cleaned)
        asset = extr['asset']
        threat = extr['threat']
        _, _, risk = scorer.compute_risk(asset, threat)
        pred_assets.append(asset)
        pred_threats.append(threat)
        pred_risks.append(risk)
    
    # Trim to same length
    true_assets = df['true_asset'][:len(pred_assets)]
    true_threats = df['true_threat'][:len(pred_assets)]
    true_risks = df['true_risk_score'][:len(pred_assets)]
    
    print("=== DistilRoBERTa Performance ===")
    print("Asset extraction:", MetricsCalculator.classification_metrics(true_assets, pred_assets))
    print("Threat extraction:", MetricsCalculator.classification_metrics(true_threats, pred_threats))
    print("Risk MAE:", MetricsCalculator.regression_mae(true_risks, pred_risks))

if __name__ == '__main__':
    main()