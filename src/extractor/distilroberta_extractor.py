from transformers import pipeline
from .base_extractor import BaseExtractor
import logging

logger = logging.getLogger(__name__)

class DistilRoBERTaExtractor(BaseExtractor):
    def __init__(self, asset_labels=None, threat_labels=None):
        # Use zero-shot classification as a lightweight proxy for extraction
        self.classifier = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")
        self.asset_labels = asset_labels or ["public_web_server", "internal_database", "developer_workstation", "test_environment", "unknown"]
        self.threat_labels = threat_labels or ["sql_injection", "xss", "dos", "brute_force", "privilege_escalation", "malicious_file", "info_leak"]
    
    def extract(self, text: str) -> dict:
        """Perform zero-shot classification to guess asset and threat."""
        # Truncate long text to 512 tokens approx
        truncated = text[:512]
        
        # Asset classification
        asset_result = self.classifier(truncated, self.asset_labels)
        asset = asset_result['labels'][0]
        asset_conf = asset_result['scores'][0]
        
        # Threat classification
        threat_result = self.classifier(truncated, self.threat_labels)
        threat = threat_result['labels'][0]
        threat_conf = threat_result['scores'][0]
        
        # Combined confidence (average)
        confidence = (asset_conf + threat_conf) / 2.0
        
        return {
            "asset": asset,
            "threat": threat,
            "confidence": confidence
        }