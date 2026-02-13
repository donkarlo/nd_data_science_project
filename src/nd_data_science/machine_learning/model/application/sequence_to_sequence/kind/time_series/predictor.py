from abc import abstractmethod, ABC

from nd_data_science.machine_learning.model.application.sequence_to_sequence.predictor.predicting import Predicting as SequenceToSequencePredictor
from nd_data_science.machine_learning.model.architecture.architecture import Architecture as ModelArchitecture
import numpy as np

class Predictor(SequenceToSequencePredictor, ABC):
    def __init__(self, achitecture: ModelArchitecture, learned_parameters):
        SequenceToSequencePredictor.__init__(self, achitecture, learned_parameters)

    @abstractmethod
    def get_predictions(self, input_array)->np.ndarray:
        pass