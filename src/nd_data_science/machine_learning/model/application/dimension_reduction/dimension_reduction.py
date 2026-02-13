from typing import List

from nd_data_science.machine_learning.model.application.dimension_reduction.interface import Interface
from nd_math.linear_algebra.tensor.vector.vector import Vector


class DimentionReduction(Interface):
    def get_reduced_dimension_vectors(self) -> List[Vector]:
        return super().get_reduced_dimension_vectors()

    def get_high_dimension_vectors(self) -> List[Vector]:
        return super().get_high_dimension_vectors()