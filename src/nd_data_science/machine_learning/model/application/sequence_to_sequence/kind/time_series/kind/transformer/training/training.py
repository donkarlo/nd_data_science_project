import numpy as np
import tensorflow as tf

from nd_data_science.machine_learning.model.application.sequence_to_sequence.kind.time_series.kind.transformer.architecture.architecture import \
    Architecture
from nd_data_science.machine_learning.model.application.sequence_to_sequence.kind.time_series.kind.transformer.kind.uncertainty.gaussian.training.config import Config
from nd_data_science.machine_learning.model.application.sequence_to_sequence.kind.time_series.kind.transformer.kind.uncertainty.gaussian.training.learned_parameter.learned_parameter import \
    LearnedParameter


class Training:
    def __init__(self, architecture: Architecture, config: Config, input_target_pairs: np.ndarray):
        self._architecture = architecture
        self._config = config

        input_array = input_target_pairs[:, 0]
        target_array = input_target_pairs[:, 1]

        if input_array.ndim != 3 or target_array.ndim != 3:
            raise ValueError("input_array and target_array must have shape (B, T, F).")

        if int(input_array.shape[1]) != int(self._architecture.get_output_time_steps()):
            raise ValueError("This vanilla version assumes T_in == output_time_steps.")

        if int(input_array.shape[2]) != int(self._architecture.get_input_feature_count()):
            raise ValueError("Input feature count mismatch.")

        if int(target_array.shape[1]) != int(self._architecture.get_output_time_steps()):
            raise ValueError("Target time steps mismatch.")

        if int(target_array.shape[2]) != int(self._architecture.get_output_feature_count()):
            raise ValueError("Target feature count mismatch.")

        self._input_array = input_array.astype(np.float32, copy=False)
        self._target_array = target_array.astype(np.float32, copy=False)

        self._tf_model = self._architecture.build_tf_model()
        self._learned_parameters: LearnedParameter | None = None

        self._train_once()

    def _train_once(self) -> None:
        self._tf_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self._config.get_learning_rate()),
            loss=tf.keras.losses.MeanSquaredError(),
        )

        self._tf_model.fit(
            self._input_array,
            self._target_array,
            epochs=self._config.get_epochs(),
            batch_size=self._config.get_batch_size(),
            shuffle=self._config.get_shuffle(),
            verbose=2,
        )

        self._learned_parameters = LearnedParameter(weights=self._tf_model.get_weights())

    def get_architecture(self) -> Architecture:
        return self._architecture

    def get_learned_parameters(self) -> LearnedParameter:
        return self._learned_parameters
