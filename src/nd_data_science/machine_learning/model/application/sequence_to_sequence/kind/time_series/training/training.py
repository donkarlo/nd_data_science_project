from abc import abstractmethod
from nd_data_science.machine_learning.model.architecture.architecture import Architecture
from nd_data_science.machine_learning.model.architecture.config import Config
from nd_data_science.machine_learning.model.supervision.kind.supervion_dependent.training.training import Training as SupervionDependentTraining
from nd_data_science.machine_learning.model.supervision.kind.supervion_dependent.training.learned_parameters import LearnedParameters as SupervisionDependentLearnedParameters
import numpy as np

class Training(SupervionDependentTraining):
    def __init__(self, architecture: Architecture, config:Config, input_traget_pairs:np.ndarray):
        SupervionDependentTraining.__init__(self, architecture, config, input_traget_pairs)
        self._learned_parameters = None

    @abstractmethod
    def get_learned_parameters(self)-> SupervisionDependentLearnedParameters:
        ...