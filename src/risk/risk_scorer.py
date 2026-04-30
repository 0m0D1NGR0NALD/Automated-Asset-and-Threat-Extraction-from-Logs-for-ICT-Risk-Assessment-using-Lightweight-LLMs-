class RiskScorer:
    def __init__(self, config):
        self.asset_criticality = config.get('asset_criticality', {})
        self.likelihood_matrix = config.get('likelihood_matrix', {})
        self.default_likelihood = self.likelihood_matrix.get('default', 3)
    
    def get_impact(self, asset: str) -> int:
        return self.asset_criticality.get(asset, self.asset_criticality.get('unknown', 3))
    
    def get_likelihood(self, threat: str) -> int:
        return self.likelihood_matrix.get(threat, self.default_likelihood)
    
    def compute_risk(self, asset: str, threat: str) -> tuple:
        impact = self.get_impact(asset)
        likelihood = self.get_likelihood(threat)
        risk = impact * likelihood
        return likelihood, impact, risk