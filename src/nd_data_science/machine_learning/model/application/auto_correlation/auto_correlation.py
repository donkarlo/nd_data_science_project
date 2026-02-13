from abc import ABC, abstractmethod
import numpy as np

class AutoCorrelation(ABC):
    """
    definition:
        1:
            cite:
                url: https://en.wikipedia.org/wiki/Autocorrelation
                say: measures the correlation of a signal with a delayed copy of itself.
    """

    @abstractmethod
    def compute_at_lag(self, signal: np.ndarray, lag: int) -> float:
        """
        Compute the autocorrelation value at a specific lag.
        """
        pass

    @abstractmethod
    def get_supported_lags(self, signal_length: int) -> np.ndarray:
        """
        Return the set of valid lags for a signal of given length.
        """
        pass

    @abstractmethod
    def is_normalized(self) -> bool:
        """
        Indicate whether the autocorrelation is normalized.
        """
        pass