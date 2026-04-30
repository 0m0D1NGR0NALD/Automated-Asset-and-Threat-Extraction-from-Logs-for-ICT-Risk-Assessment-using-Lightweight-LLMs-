class ConfidenceFilter:
    def __init__(self, threshold=0.75):
        self.threshold = threshold
    
    def requires_review(self, confidence: float) -> bool:
        return confidence < self.threshold