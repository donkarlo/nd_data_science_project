from nd_data_science.machine_learning.model.application.sequence_to_sequence.training.learned_parameters import LearnedParameters
from nd_data_science.machine_learning.model.architecture.architecture import Architecture


class Training:
    def __init__(self, architecture: Architecture, learned_parameters):
        self._architecture = architecture
        self._learned_parameters = learned_parameters

    def get_architecture(self) -> Architecture:
        return self._architecture

    def get_learned_parameters(self) -> LearnedParameters:
        return self._learned_parameters
