import pandas as pd
from typing import Tuple

class BaseStrategy:
    """Base class for all trading strategies"""
    def __init__(self, name):
        self.name = name
        
    def get_signal(self, data: pd.DataFrame, symbol: str = None) -> Tuple[str, float]:
        """
        Return a tuple of (signal, strength) where:
        - signal: 'buy', 'sell', or 'hold'
        - strength: float between 0.0 and 1.0 indicating signal confidence
        """
        raise NotImplementedError("All strategies must implement get_signal method")