import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from src.utils.config import Config
from src.utils.logger import setup_logger
from src.parser.input_reader import InputReader
from src.parser.log_preprocessor import LogPreprocessor
from src.extractor.distilroberta_extractor import DistilRoBERTaExtractor
from src.extractor.smolLM2_extractor import SmolLM2Extractor
from src.extractor.qwen_extractor import QwenExtractor
from src.extractor.tinyllama_extractor import TinyLlamaExtractor
from src.risk.risk_scorer import RiskScorer
from src.risk.confidence_filter import ConfidenceFilter
from src.output.csv_generator import CSVGenerator
from datetime import datetime

logger = setup_logger()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True, help='Input file (log, CSV, JSON)')
    parser.add_argument('--output', '-o', default='risk_register.csv', help='Output CSV path')
    parser.add_argument('--model', choices=['distilroberta', 'smolLM2', 'qwen', 'tinyllama'], default='smolLM2')
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    if args.model != config.get('model.backend'):
        logger.warning(f"Overriding model backend to {args.model}")
    
    # Choose extractor
    if args.model == 'distilroberta':
        extractor = DistilRoBERTaExtractor()
    elif args.model == 'smolLM2':
        extractor = SmolLM2Extractor()
    elif args.model == 'qwen':
        extractor = QwenExtractor()
    elif args.model == 'tinyllama':
        extractor = TinyLlamaExtractor()
    
    # Risk assessment components
    risk_scorer = RiskScorer(config)
    confidence_filter = ConfidenceFilter(threshold=config.get('model.confidence_threshold', 0.75))
    
    # Read input
    raw_entries = InputReader.read(args.input)
    logger.info(f"Read {len(raw_entries)} entries")
    
    output_entries = []
    for idx, entry in enumerate(raw_entries):
        raw_text = entry['raw']
        cleaned, timestamp, ip = LogPreprocessor.clean_line(raw_text)
        if not cleaned:
            continue
        
        # Extract
        extraction = extractor.extract(cleaned)
        asset = extraction['asset']
        threat = extraction['threat']
        confidence = extraction['confidence']
        
        # Risk scoring
        likelihood, impact, risk = risk_scorer.compute_risk(asset, threat)
        requires_review = confidence_filter.requires_review(confidence)
        
        # Prepare output row
        out_row = {
            'timestamp': timestamp or datetime.now().isoformat(),
            'raw_log_preview': raw_text[:100],
            'extracted_asset': asset,
            'extracted_threat': threat,
            'likelihood_score': likelihood,
            'impact_score': impact,
            'risk_score': risk,
            'confidence': round(confidence, 3),
            'requires_review': requires_review,
            'human_override': ''
        }
        output_entries.append(out_row)
        
        if idx % 10 == 0:
            logger.info(f"Processed {idx+1}/{len(raw_entries)}")
    
    CSVGenerator.generate(output_entries, args.output)
    logger.info("Done.")

if __name__ == '__main__':
    main()