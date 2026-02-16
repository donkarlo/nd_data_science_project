# file: nd_data_science/ml/period_estimation/predicting/predicting.py
from __future__ import annotations

import numpy as np
import tensorflow as tf
from scipy.signal import find_peaks

from nd_data_science.machine_learning.model.application.auto_correlation.sequence.period.neural.training.learned_parameters import \
    LearnedParameters
from nd_data_science.machine_learning.model.application.auto_correlation.sequence.period.neural.architecture.architecture import \
    Architecture


class Predicting:
    def __init__(self, architecture: Architecture, learned_parameters: LearnedParameters, batch_size: int = 256):
        self._architecture = architecture
        self._learned_parameters = learned_parameters
        self._batch_size = int(batch_size)

        if self._batch_size <= 0:
            raise ValueError("batch_size must be > 0.")

        self._encoder = None
        self._encoder_is_ready = False

    def estimate_period(self, sequence: np.ndarray, min_period: int = 10) -> int:
        if not isinstance(sequence, np.ndarray):
            raise TypeError("sequence must be np.ndarray.")
        if sequence.ndim != 2:
            raise ValueError("sequence must have shape (time_steps, features).")

        min_period = int(min_period)
        if min_period < 1:
            raise ValueError("min_period must be >= 1.")

        sequence = sequence.astype(np.float32, copy=False)

        self._ensure_encoder_ready(num_features=int(sequence.shape[1]))

        centered = sequence - sequence.mean(axis=0, keepdims=True)

        embeddings = self._encode_per_timestep(centered)
        autocorrelation = self._multivariate_autocorrelation_fft(embeddings)

        peaks, _ = find_peaks(autocorrelation[min_period:])
        if len(peaks) == 0:
            raise RuntimeError("No period detected.")

        return int(peaks[0] + min_period)

    def _ensure_encoder_ready(self, num_features: int) -> None:
        if self._encoder_is_ready:
            return

        encoder = self._build_encoder(num_features=num_features)

        weights = self._learned_parameters.get_weights()
        encoder.set_weights(weights)

        self._encoder = encoder
        self._encoder_is_ready = True

    def _build_encoder(self, num_features: int) -> tf.keras.Model:
        window_length = int(self._architecture.get_window_length())
        latent_size = int(self._architecture.get_latent_size())

        input_layer = tf.keras.Input(shape=(window_length, num_features), dtype=tf.float32, name="x_in")
        hidden_one = tf.keras.layers.GRU(latent_size, return_sequences=True, name="enc_gru_1")(input_layer)
        latent_sequence = tf.keras.layers.GRU(latent_size, return_sequences=True, name="enc_gru_2")(hidden_one)

        encoder = tf.keras.Model(input_layer, latent_sequence, name="gru_encoder")

        dummy = tf.zeros((1, window_length, num_features), dtype=tf.float32)
        _ = encoder(dummy, training=False)

        return encoder

    def _encode_per_timestep(self, sequence: np.ndarray) -> np.ndarray:
        time_steps, feature_count = sequence.shape
        window_length = int(self._architecture.get_window_length())
        latent_size = int(self._architecture.get_latent_size())

        number_of_windows = time_steps - window_length + 1
        if number_of_windows <= 0:
            raise ValueError("sequence is shorter than window_length.")

        windows = np.zeros((number_of_windows, window_length, feature_count), dtype=np.float32)
        for index in range(number_of_windows):
            windows[index] = sequence[index:index + window_length]

        encoded = self._encoder.predict(windows, batch_size=self._batch_size, verbose=0)

        embeddings = np.zeros((time_steps, latent_size), dtype=np.float64)
        counts = np.zeros(time_steps, dtype=np.float64)

        for window_index in range(number_of_windows):
            for offset in range(window_length):
                embeddings[window_index + offset] += encoded[window_index, offset]
                counts[window_index + offset] += 1.0

        embeddings /= counts[:, None]
        return embeddings

    def _multivariate_autocorrelation_fft(self, sequence: np.ndarray) -> np.ndarray:
        time_steps, feature_count = sequence.shape

        fft_length = 1
        while fft_length < 2 * time_steps:
            fft_length *= 2

        autocorrelation = np.zeros(time_steps, dtype=np.float64)

        for feature_index in range(feature_count):
            signal = sequence[:, feature_index]
            spectrum = np.fft.rfft(signal, n=fft_length)
            correlation = np.fft.irfft(spectrum * np.conj(spectrum), n=fft_length)
            autocorrelation += correlation[:time_steps]

        return autocorrelation / np.arange(time_steps, 0, -1)

