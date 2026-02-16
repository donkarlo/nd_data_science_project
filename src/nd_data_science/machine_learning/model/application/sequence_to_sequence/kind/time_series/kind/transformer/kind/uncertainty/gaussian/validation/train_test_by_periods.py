import numpy as np

from nd_data_science.machine_learning.model.application.sequence_to_sequence.kind.time_series.kind.transformer.kind.uncertainty.gaussian.predicting.predicting import \
    Predicting
from nd_math.view.kind.point_cloud.decorator.lined.ordered_intra_line_connected import OrderedIntraLineConnected
from nd_data_science.machine_learning.model.validation.validation import Validation
from nd_math.view.kind.point_cloud.point_cloud import PointCloud
from nd_math.view.kind.point_cloud.point.group.group import Group as PointGroupCloudPlotView


class TrainTestByPeriods(Validation):
    def __init__(self, predictor:Predicting, train_data, test_data):
        self._test_predicted_distribution_mean: np.ndarray | None = None
        self._test_set_target_values: np.ndarray | None = None
        self._predictor = predictor

        self._train_set_pairs = train_data
        self._train_set_inputs = self._train_set_pairs[:, 0]
        self._train_set_targets = self._train_set_pairs[:, 1]

        self._test_set_pairs = test_data
        self._test_set_inputs = self._test_set_pairs[:, 0]
        self._test_set_targets = self._test_set_pairs[:, 1]

        self._test()

    def _test(self)->None:
        self._test_predicted_distribution_mean, self._test_predicted_distribution_variance = self._predictor.get_predicted_distributions(self._test_set_inputs)



    def render_euclidean_distance(self)->None:

        residuals = self._test_predicted_distribution_mean - self._test_set_targets
        residuals = np.linalg.norm(residuals, axis=-1)
        distance_curve = residuals.mean(axis=1)

        residuals = distance_curve  # shape: (N,)
        indices = np.arange(1, len(residuals) + 1)
        two_dimensional = np.column_stack((indices, residuals))

        point_group = PointGroupCloudPlotView(two_dimensional)
        point_cloud = OrderedIntraLineConnected(PointCloud(point_group))
        point_cloud.render()