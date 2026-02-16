# file: nd_data_science/ml/period_estimation/training/training.py
from __future__ import annotations

import numpy as np
import tensorflow as tf

from nd_data_science.machine_learning.model.application.auto_correlation.sequence.period.neural.architecture.architecture import \
    Architecture
from nd_data_science.machine_learning.model.application.auto_correlation.sequence.period.neural.training.config import \
    Config as TrainingConfig
from nd_data_science.machine_learning.model.application.auto_correlation.sequence.period.neural.training.learned_parameters import \
    LearnedParameters


class Training:
    def __init__(self, architecture: Architecture, training_config: TrainingConfig):
        self._architecture = architecture
        self._training_config = training_config

    def fit(self, sequence: np.ndarray) -> LearnedParameters:
        if not isinstance(sequence, np.ndarray):
            raise TypeError("sequence must be np.ndarray.")
        if sequence.ndim != 2:
            raise ValueError("sequence must have shape (time_steps, features).")

        sequence = sequence.astype(np.float32, copy=False)
        num_features = int(sequence.shape[1])

        autoencoder, encoder = self._build_models_with_shared_layers(num_features=num_features)
        dataset = self._build_window_dataset(sequence)

        autoencoder.fit(dataset, epochs=self._training_config.get_epochs(), verbose=0)

        learned_parameters = LearnedParameters(weights=encoder.get_weights())
        return learned_parameters

    def _build_window_dataset(self, sequence: np.ndarray) -> tf.data.Dataset:
        window_length = int(self._architecture.get_window_length())
        batch_size = int(self._training_config.get_batch_size())

        dataset = tf.data.Dataset.from_tensor_slices(sequence)
        dataset = dataset.window(window_length, shift=1, drop_remainder=True)
        dataset = dataset.flat_map(lambda window: window.batch(window_length))
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.map(lambda batch: (batch, batch))
        return dataset.prefetch(tf.data.AUTOTUNE)

    def _build_models_with_shared_layers(self, num_features: int) -> tuple[tf.keras.Model, tf.keras.Model]:
        window_length = int(self._architecture.get_window_length())
        latent_size = int(self._architecture.get_latent_size())

        input_layer = tf.keras.Input(shape=(window_length, num_features), dtype=tf.float32, name="x_in")

        encoder_gru_one = tf.keras.layers.GRU(latent_size, return_sequences=True, name="enc_gru_1")
        encoder_gru_two = tf.keras.layers.GRU(latent_size, return_sequences=True, name="enc_gru_2")
        decoder_dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_features), name="decoder")

        hidden_one = encoder_gru_one(input_layer)
        latent_sequence = encoder_gru_two(hidden_one)
        reconstruction = decoder_dense(latent_sequence)

        autoencoder = tf.keras.Model(input_layer, reconstruction, name="gru_autoencoder")
        autoencoder.compile(optimizer="adam", loss="mse")

        encoder = tf.keras.Model(input_layer, latent_sequence, name="gru_encoder")

        return autoencoder, encoder
