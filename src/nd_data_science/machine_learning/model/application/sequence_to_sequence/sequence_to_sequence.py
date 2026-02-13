from abc import ABC, abstractmethod

import numpy as np
from nd_data_science.machine_learning.model.model import Model


class SequenceToSequence(Model, ABC):
    """
    \cite{Me}
        Not all sequence to ssequenc models are spervion reliable. For example Kallman Filter is not trained but it can predict . Or even with velocity and acceleration we can do the same. these are sequence to sequence model but they are rul based
    """

    def __init__(self):
        pass

    @abstractmethod
    def get_predictions(self, input:np.ndarray)->np.ndarray:
        pass