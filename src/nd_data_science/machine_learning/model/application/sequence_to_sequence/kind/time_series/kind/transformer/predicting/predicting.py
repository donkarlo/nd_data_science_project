import numpy as np
import tensorflow as tf

from nd_data_science.machine_learning.model.application.sequence_to_sequence.kind.time_series.kind.transformer.architecture.architecture import \
    Architecture
from nd_data_science.machine_learning.model.application.sequence_to_sequence.trainer.learned_parameters import LearnedParameters


class Predictor:
    def __init__(self, architecture: Architecture, learned_parameters: LearnedParameters | None = None):
        self._architecture = architecture
        self._learned_parameters = learned_parameters

        self._tf_model = self._architecture.build_tf_model()

        dummy = tf.zeros((1, self._architecture.get_output_time_steps(), self._architecture.get_input_feature_count()),
                         dtype=tf.float32)
        _ = self._tf_model(dummy, training=False)

        self._maybe_load_weights()

    def _maybe_load_weights(self) -> None:
        if self._learned_parameters is None:
            return
        weights = self._learned_parameters.get_weights()
        if weights is None:
            return
        self._tf_model.set_weights(weights)

    def get_predictions(self, input_array: np.ndarray) -> np.ndarray:
        if not isinstance(input_array, np.ndarray):
            raise TypeError("input_array must be np.ndarray.")
        if input_array.ndim != 3:
            raise ValueError("input_array must have shape (B, T, F).")

        if int(input_array.shape[1]) != int(self._architecture.get_output_time_steps()):
            raise ValueError("This vanilla version assumes T_in == output_time_steps.")

        out = self._tf_model(input_array.astype(np.float32, copy=False), training=False)
        return out.numpy()
