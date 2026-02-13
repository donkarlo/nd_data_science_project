from abc import ABC, abstractmethod
import numpy as np

class Supervised(ABC):
    @abstractmethod
    def train(self, training_set:np.ndarray)->None:
        pass

    @abstractmethod
    def test(self, test_set:np.ndarray)->None:
        pass

    @abstractmethod
    def validate(self, validation_set:np.ndarray)->None:
        pass