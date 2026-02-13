import numpy as np


class PairWiseEuclideanDistances:
    """
    Measures the distance between each pair of elements from predicted and target array
    """
    def __init__(self, target: np.ndarray, predicted: np.ndarray):
        self._target = target
        self._predicted = predicted
