import numpy as np

from nd_data_science.machine_learning.model.application.sequence_to_sequence.predictor.predicting import Predicting
from nd_data_science.machine_learning.model.application.sequence_to_sequence.trainer.trainer import Trainer
from nd_data_science.machine_learning.model.architecture.architecture import Architecture
from nd_data_science.machine_learning.model.supervision.kind.supervion_dependent.training.config import Config
from nd_math.probability.statistic.population.sampling.sampler.kind.countable.finite.members_mentioned.numbered.sequence.sliding_window.generator import \
    Generator as SlidingWindowGenerator
from nd_math.probability.statistic.population.sampling.sampler.kind.countable.finite.members_mentioned.numbered.sequence.sliding_window.sliding_window import SlidingWindow
from nd_math.view.kind.point_cloud.decorator.lined.ordered_intra_line_connected import OrderedIntraLineConnected
from nd_data_science.machine_learning.model.validation.validation import Validation
from nd_math.view.kind.point_cloud.point_cloud import PointCloud
from nd_math.view.kind.point_cloud.point.group.group import Group


class TrainTestByPeriods(Validation):
    def __init__(self, predictor:Predicting, train_data, test_data):
        self._test_set_predictions: np.ndarray | None = None
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
        print("test_inputs:", self._test_set_inputs.shape, self._test_set_inputs.dtype)
        self._test_set_predictions = self._predictor.get_predictions(self._test_set_inputs)
        print("predictions:", self._test_set_predictions.shape, self._test_set_predictions.dtype,
              self._test_set_predictions.nbytes / 1024 / 1024, "MB")
        self._test_set_predictions = self._predictor.get_predictions(self._test_set_inputs)



    def render_euclidean_distance(self):
        print("pred:", self._test_set_predictions.shape, self._test_set_predictions.dtype)
        print("tgt:", self._test_set_targets.shape, self._test_set_targets.dtype)

        residuals = self._test_set_predictions - self._test_set_targets
        residuals = np.linalg.norm(residuals, axis=-1)
        distance_curve = residuals.mean(axis=1)

        residuals = distance_curve  # shape: (N,)
        indices = np.arange(1, len(residuals) + 1)
        two_dimensional = np.column_stack((indices, residuals))

        pair_set = Group(two_dimensional)
        point_cloud = OrderedIntraLineConnected(PointCloud(pair_set))
        point_cloud.render()

    @classmethod
    def init_from_one_split(cls, train_data: np.ndarray, test_data: np.ndarray, trainer_class:Trainer, architecture:Architecture, trainer_configs:Config, predictor_class: Predicting, sliding_window_step)->None:
        #preparing data
        sliding_window_step = sliding_window_step
        sliding_window_length = architecture.get_output_time_steps()

        train_sliding_window = SlidingWindow(sliding_window_length, sliding_window_length, sliding_window_step)
        training_sliding_windows_generator = SlidingWindowGenerator(train_data, train_sliding_window)
        train_input_target_pairs = training_sliding_windows_generator.get_input_output_pairs()

        test_sliding_window = SlidingWindow(sliding_window_length, sliding_window_length, sliding_window_step)
        test_sliding_windows_generator = SlidingWindowGenerator(test_data, test_sliding_window)
        test_input_target_pairs = test_sliding_windows_generator.get_input_output_pairs()

        trainer = trainer_class(architecture, trainer_configs, train_input_target_pairs)
        learned_parameters = trainer.get_learned_parameters()

        predictor  = predictor_class(architecture, learned_parameters)


        return cls(predictor, train_input_target_pairs, test_input_target_pairs)