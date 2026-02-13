from abc import ABC, abstractmethod
import numpy as np

class PeriodDetector(ABC):
    @abstractmethod
    def detect(self, autocorrelation: np.ndarray) -> list[int]:
        pass
