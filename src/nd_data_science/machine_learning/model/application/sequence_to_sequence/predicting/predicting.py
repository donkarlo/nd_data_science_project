from nd_data_science.machine_learning.model.application.sequence_to_sequence.training.learned_parameters import LearnedParameters
from nd_data_science.machine_learning.model.architecture.architecture import Architecture as ModelArchitecture


class Predicting:
    def __init__(self, achitecture: ModelArchitecture, learned_parameters:LearnedParameters):
        self._architecture = achitecture
        self._learned_parameters = learned_parameters

    def get_learned_parameters(self)-> LearnedParameters:
        return self._learned_parameters

    def get_architecture(self) -> ModelArchitecture:
        return self._architecture