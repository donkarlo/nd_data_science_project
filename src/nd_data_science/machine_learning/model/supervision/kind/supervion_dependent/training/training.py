import numpy as np

from nd_data_science.machine_learning.model.architecture.architecture import Architecture
from nd_data_science.machine_learning.model.supervision.kind.supervion_dependent.training.config import Config


class Training():
    def __init__(self, architecture: Architecture, config: Config, input_target_pairs: np.ndarray):
        self._architecture = architecture
        self._config = config
        self._input_target_pairs = input_target_pairs

    def get_architecture(self) -> Architecture:
        return self._architecture

    def get_config(self) -> Config:
        return self._config

    def get_input_target_pairs(self) -> np.ndarray:
        return self._input_target_pairs
