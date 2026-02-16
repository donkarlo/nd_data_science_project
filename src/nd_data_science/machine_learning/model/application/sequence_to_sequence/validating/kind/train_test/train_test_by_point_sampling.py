import numpy as np

from nd_data_science.machine_learning.model.application.sequence_to_sequence.predictor.predicting import Predicting
from nd_data_science.machine_learning.model.application.sequence_to_sequence.trainer.trainer import Trainer
from nd_data_science.machine_learning.model.architecture.architecture import Architecture
from nd_data_science.machine_learning.model.supervision.kind.supervion_dependent.training.config import Config
from nd_math.number.kind.real.interval.unit.close_unit_interval_number import CloseUnitIntervalNumber
from nd_math.probability.statistic.population.sampling.kind.countable.finite.members_mentioned.numbered.random.random import \
    Random
from nd_math.probability.statistic.population.sampling.kind.countable.finite.members_mentioned.numbered.sequence.sliding_window.generator import \
    Generator as SlidingWindowGenerator
from nd_math.probability.statistic.population.sampling.kind.countable.finite.members_mentioned.numbered.sequence.sliding_window.sliding_window import SlidingWindow
from nd_math.probability.statistic.population.sampling.size.kind.ratio import Ratio
from nd_math.view.kind.point_cloud.decorator.lined.ordered_intra_line_connected import OrderedIntraLineConnected
from nd_data_science.machine_learning.model.validation.validation import Validation
from nd_math.probability.statistic.population.kind.countable.finite.member_mentioned.numbered.numbered import Numbered as NumberedPopulation
from nd_math.view.kind.point_cloud.point_cloud import PointCloud
from nd_math.view.kind.point_cloud.point.group.group import Group


class TrainTestByPointSampling(Validation):
    def __init__(self, predictor:Predicting, train_data, test_data):
        self._test_predictions: np.ndarray | None = None
        self._test_target_values: np.ndarray | None = None
        self._predictor = predictor

        self._train_data_pairs = train_data
        self._train_data_inputs = self._train_data_pairs[:, 0]
        self._train_data_targets = self._train_data_pairs[:, 1]

        self._test_set_pairs = test_data
        self._test_set_inputs = self._test_set_pairs[:, 0]
        self._test_set_targets = self._test_set_pairs[:, 1]

        self._test()

    def _test(self)->None:
        self._test_predictions = self._predictor.get_predictions(self._test_set_inputs)



    def render_euclidean_distance(self):
        residuals = self._test_predictions - self._test_set_targets
        residuals = np.linalg.norm(residuals, axis=-1)
        distance_curve = residuals.mean(axis=1)

        residuals = distance_curve  # shape: (N,)
        indices = np.arange(1, len(residuals) + 1)
        two_dimensional = np.column_stack((indices, residuals))

        pair_set = Group(two_dimensional)
        point_cloud = OrderedIntraLineConnected(PointCloud(pair_set))
        point_cloud.render()


    @classmethod
    def init_from_partitionaning_ratio(cls, trainer_class:Trainer, architecture:Architecture, trainer_configs:Config, predictor_class: Predicting, population: NumberedPopulation, ratio_value: float)->None:

        ratio_value = Ratio(CloseUnitIntervalNumber(ratio_value), population.get_size())
        random_point_sampler = Random(population, ratio_value)

        train_data = random_point_sampler.get_samples()
        test_data = random_point_sampler.get_complements()

        sliding_window_step = 3
        train_set_sliding_window = SlidingWindow(100, 100, sliding_window_step)
        training_set_sliding_windows_generator = SlidingWindowGenerator(train_data, train_set_sliding_window)
        train_set_input_target_pairs = training_set_sliding_windows_generator.get_input_output_pairs()

        test_set_sliding_window = SlidingWindow(100, 100, sliding_window_step)
        test_set_sliding_windows_generator = SlidingWindowGenerator(test_data, test_set_sliding_window)
        test_set_input_target_pairs = test_set_sliding_windows_generator.get_input_output_pairs()





        trainer = trainer_class(architecture, trainer_configs, train_set_input_target_pairs)
        learned_parameters = trainer.get_learned_parameters()

        predictor  = predictor_class(architecture, learned_parameters)


        return cls(predictor, train_set_input_target_pairs, test_set_input_target_pairs)