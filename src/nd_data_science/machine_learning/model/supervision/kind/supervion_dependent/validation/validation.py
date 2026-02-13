import numpy as np

from nd_data_science.machine_learning.model.application.sequence_to_sequence.kind.time_series_forcating.kind.transformer.transformer_draft import TransformerDraft
from nd_math.number.kind.real.interval.unit.open_unit_interval import OpenUnitInterval
from nd_math.probability.statistic.population.kind.countable.finite.member_mentioned.numbered.numbered import Numbered as NumpiedPopulation


class Validation:
    def __init__(self, model:TransformerDraft, inputs:np.ndarray, targets:np.ndarray, predictions: np.ndarray):
        pass

    def init_from_partitionaning_ratio(self, population: NumpiedPopulation, ratio:OpenUnitInterval):
        pass




