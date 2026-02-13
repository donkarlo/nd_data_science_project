import numpy as np

from nd_data_science.machine_learning.model.application.sequence_to_sequence.predictor.predicting import Predicting
from nd_data_science.machine_learning.model.application.sequence_to_sequence.trainer.trainer import Trainer
from nd_data_science.machine_learning.model.architecture.architecture import Architecture
from nd_data_science.machine_learning.model.supervision.kind.supervion_dependent.training.config import Config
from nd_math.view.kind.point_cloud.decorator.lined.ordered_intra_line_connected import OrderedIntraLineConnected
from nd_data_science.machine_learning.model.validation.validation import Validation
from nd_math.set_nd.decorator.factory.numpied import Numpied as NumpiedSet
from nd_math.number.kind.real.interval.unit.open_unit_interval import OpenUnitInterval
from nd_math.probability.statistic.population.kind.countable.finite.member_mentioned.numbered.numbered import Numbered as NumpiedPopulation
from nd_math.view.kind.point_cloud.point_cloud import PointCloud
from nd_math.view.kind.point_cloud.point.group.group.Group import Group as GroupPairSet
from nd_math.view.kind.point_cloud.point.group.group import Group


class TrainTestBySlidingWindowSampling(Validation):
    def __init__(self, predictor: Predicting, train_set: NumpiedSet, test_set: NumpiedSet):
        self._test_set_predictions: np.ndarray | None = None
        self._test_set_target_values: np.ndarray | None = None
        self._predictor = predictor

        self._train_set_pairs = np.asarray(train_set.get_members())
        self._train_set_inputs = self._train_set_pairs[:, 0]
        self._train_set_targets = self._train_set_pairs[:, 1]

        self._test_set_pairs = np.asarray(test_set.get_members())
        self._test_set_inputs = self._test_set_pairs[:, 0]
        self._test_set_targets = self._test_set_pairs[:, 1]

        self._test()

    def _test(self) -> None:
        self._test_set_predictions = self._predictor.get_predictions(self._test_set_inputs)

    def render_euclidean_distance(self):
        residuals = self._test_set_predictions - self._test_set_targets
        residuals = np.linalg.norm(residuals, axis=-1)
        distance_curve = residuals.mean(axis=1)

        residuals = distance_curve  # shape: (N,)
        indices = np.arange(1, len(residuals) + 1)
        two_dimensional = np.column_stack((indices, residuals))

        pair_set = Group(two_dimensional)
        point_cloud = OrderedIntraLineConnected(PointCloud(pair_set))
        point_cloud._build()
        point_cloud.render()


    @classmethod
    def init_from_partitionaning_ratio(cls, trainer_class: Trainer, architecture: Architecture, trainer_configs: Config,
                                       predictor_class: Predicting, sample_population: NumpiedPopulation,
                                       ratio: OpenUnitInterval) -> None:
        subset_complement_partition = sample_population.get_random_sample_and_complement_by_ratio(ratio)

        train_set = subset_complement_partition.get_subset()
        test_set = subset_complement_partition.get_complement()

        trainer = trainer_class(architecture, trainer_configs, train_set.get_members())
        learned_parameters = trainer.get_learned_parameters()

        predictor = predictor_class(architecture, learned_parameters)

        return cls(predictor, train_set, test_set)