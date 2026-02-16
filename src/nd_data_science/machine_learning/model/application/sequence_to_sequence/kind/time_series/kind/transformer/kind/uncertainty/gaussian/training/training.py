import numpy as np
import tensorflow as tf

from nd_data_science.machine_learning.model.application.sequence_to_sequence.kind.time_series.kind.transformer.architecture.architecture import \
    Architecture as GaussianArchitecture
from nd_data_science.machine_learning.model.application.sequence_to_sequence.kind.time_series.kind.transformer.kind.uncertainty.gaussian.training.config import Config as TrainingConfig
from nd_data_science.machine_learning.model.application.sequence_to_sequence.kind.time_series.kind.transformer.kind.uncertainty.gaussian.training.learned_parameter.learned_parameter import \
    LearnedParameter
from nd_math.probability.statistic.population.sampling.kind.countable.finite.members_mentioned.numbered.sequence.sliding_window.sliding_window import \
    SlidingWindow

from nd_math.probability.statistic.population.sampling.kind.countable.finite.members_mentioned.numbered.sequence.sliding_window.generator import \
    Generator as SlidingWindowGenerator


class Training:
    """Train a Gaussian forecaster: predicts (mean, log_var) and trains with Gaussian NLL (diagonal)."""

    def __init__(self, architecture: GaussianArchitecture, config: TrainingConfig, training_sequence: np.ndarray):
        self._architecture = architecture
        self._config = config
        self._training_sequence = training_sequence
        self._generate_input_traget_sequence_pairs()

        input_array = self._training_input_target_sequence_pairs[:, 0]
        target_array = self._training_input_target_sequence_pairs[:, 1]

        if not isinstance(input_array, np.ndarray) or not isinstance(target_array, np.ndarray):
            raise TypeError("input_array and target_array must be np.ndarray.")
        if input_array.ndim != 3 or target_array.ndim != 3:
            raise ValueError("input_array and target_array must have shape (B, T, F).")

        output_time_steps = int(self._architecture.get_output_time_steps())
        input_feature_count = int(self._architecture.get_input_feature_dimension())
        output_feature_count = int(self._architecture.get_output_feature_dimension())

        if int(input_array.shape[1]) != output_time_steps:
            raise ValueError("This vanilla version assumes T_in == output_sequence_size.")
        if int(input_array.shape[2]) != input_feature_count:
            raise ValueError("Input feature count mismatch.")
        if int(target_array.shape[1]) != output_time_steps:
            raise ValueError("Target time steps mismatch.")
        if int(target_array.shape[2]) != output_feature_count:
            raise ValueError("Target feature count mismatch.")

        self._input_array = input_array.astype(np.float32, copy=False)
        self._target_array = target_array.astype(np.float32, copy=False)

        self._tf_model = self._architecture.build_tf_model()
        self._learned_parameters: LearnedParameter | None = None

        self._train_once()

    def _generate_input_traget_sequence_pairs(self):
        sliding_window = SlidingWindow(self._config.get_config_dic()["input_sequence_size"],
                                       self._config.get_config_dic()["output_sequence_size"],
                                       self._config.get_config_dic()["sequence_overlap_size"])

        # generating sliding window
        self._training_input_target_sequence_pairs = SlidingWindowGenerator(self._training_sequence,
                                                                      sliding_window).get_input_output_pairs()


    def _gaussian_nll(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        output_feature_count = int(self._architecture.get_output_feature_dimension())

        mu = y_pred[..., :output_feature_count]
        log_var = y_pred[..., output_feature_count:]

        eps = tf.constant(1e-6, dtype=log_var.dtype)
        log_var = tf.maximum(log_var, tf.math.log(eps))

        inv_var = tf.exp(-log_var)
        squared = tf.square(y_true - mu)

        per_dim = 0.5 * (log_var + squared * inv_var)
        per_step = tf.reduce_sum(per_dim, axis=-1)
        return tf.reduce_mean(per_step)

    def _train_once(self) -> None:
        x = tf.convert_to_tensor(self._input_array, dtype=tf.float32)
        y = tf.convert_to_tensor(self._target_array, dtype=tf.float32)

        def loss_fn(y_true: tf.Tensor, y_pred: list[tf.Tensor]) -> tf.Tensor:
            return self._gaussian_nll(y_true, y_pred)

        self._tf_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self._config.get_learning_rate()),
            loss=self._gaussian_nll,
        )

        self._tf_model.fit(
            x,
            y,
            epochs=self._config.get_epochs(),
            batch_size=self._config.get_batch_size(),
            shuffle=self._config.get_shuffle(),
            verbose=2,
        )

        self._learned_parameters = LearnedParameter(weights=self._tf_model.get_weights())

    def get_architecture(self) -> GaussianArchitecture:
        return self._architecture

    def get_learned_parameters(self) -> LearnedParameter:
        return self._learned_parameters
