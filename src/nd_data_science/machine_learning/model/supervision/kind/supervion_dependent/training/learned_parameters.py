from typing import Any


class LearnedParameters:
    def __init__(self, learned_parameter_values: Any):
        self._learned_parameter_values = learned_parameter_values

    def get_learned_parameter_values(self) -> Any:
        return self._learned_parameter_values