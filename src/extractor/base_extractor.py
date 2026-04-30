from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, text: str) -> Dict[str, Any]:
        """
        Returns dictionary with:
        - asset: str (one of the asset keys from config)
        - threat: str (matching likelihood_matrix keys)
        - confidence: float (0-1)
        """
        pass