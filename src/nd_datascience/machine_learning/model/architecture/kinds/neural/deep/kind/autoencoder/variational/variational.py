from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import tensorflow as tensorflow


class Variational(tensorflow.keras.Model):
    def __init__(self, input_feature_count: int, latent_dimension_count: int = 8) -> None:
        super().__init__()

        self.input_feature_count = input_feature_count
        self.latent_dimension_count = latent_dimension_count

        self.encoder_network = tensorflow.keras.Sequential([
            tensorflow.keras.layers.InputLayer(shape=(self.input_feature_count,)),
            tensorflow.keras.layers.Dense(64, activation="relu"),
            tensorflow.keras.layers.Dense(32, activation="relu")
        ])

        self.latent_mean_layer = tensorflow.keras.layers.Dense(self.latent_dimension_count)
        self.latent_log_variance_layer = tensorflow.keras.layers.Dense(self.latent_dimension_count)

        self.fmm_parameter_layer = tensorflow.keras.Sequential([
            tensorflow.keras.layers.Dense(32, activation="relu"),
            tensorflow.keras.layers.Dense(5)
        ])

    def encode(self, input_values: tensorflow.Tensor) -> Tuple[tensorflow.Tensor, tensorflow.Tensor]:
        hidden_values = self.encoder_network(input_values)
        latent_mean = self.latent_mean_layer(hidden_values)
        latent_log_variance = self.latent_log_variance_layer(hidden_values)
        return latent_mean, latent_log_variance

    def sample_latent_vector(self, latent_mean: tensorflow.Tensor,
                             latent_log_variance: tensorflow.Tensor) -> tensorflow.Tensor:
        noise_values = tensorflow.random.normal(shape=tensorflow.shape(latent_mean))
        standard_deviation = tensorflow.exp(0.5 * latent_log_variance)
        latent_vector = latent_mean + standard_deviation * noise_values
        return latent_vector

    def constrain_action_potential_parameters(self, raw_parameters: tensorflow.Tensor) -> tensorflow.Tensor:
        raw_alpha = raw_parameters[:, 0:1]
        raw_beta = raw_parameters[:, 1:2]
        raw_omega = raw_parameters[:, 2:3]
        raw_membrane_baseline = raw_parameters[:, 3:4]
        raw_amplitude = raw_parameters[:, 4:5]

        alpha = 2.0 * math.pi * tensorflow.sigmoid(raw_alpha)
        beta = 2.0 * math.pi * tensorflow.sigmoid(raw_beta)

        omega_lower = 0.001
        omega_upper = 1.0
        omega = omega_lower + (omega_upper - omega_lower) * tensorflow.sigmoid(raw_omega)

        membrane_baseline_lower = -90.0
        membrane_baseline_upper = -45.0
        membrane_baseline = membrane_baseline_lower + (
                membrane_baseline_upper - membrane_baseline_lower
        ) * tensorflow.sigmoid(raw_membrane_baseline)

        amplitude_lower = 1.0
        amplitude_upper = 120.0
        amplitude = amplitude_lower + (amplitude_upper - amplitude_lower) * tensorflow.sigmoid(raw_amplitude)

        constrained_parameters = tensorflow.concat(
            [alpha, beta, omega, membrane_baseline, amplitude],
            axis=1
        )

        return constrained_parameters

    def call(self, input_values: tensorflow.Tensor, training: bool = False) -> tensorflow.Tensor:
        latent_mean, latent_log_variance = self.encode(input_values)

        if training:
            latent_vector = self.sample_latent_vector(latent_mean, latent_log_variance)
        else:
            latent_vector = latent_mean

        raw_parameters = self.fmm_parameter_layer(latent_vector)
        constrained_parameters = self.constrain_action_potential_parameters(raw_parameters)

        return constrained_parameters

    def map_numpy_to_action_potential_latent_space(self, input_values: np.ndarray) -> np.ndarray:
        input_tensor = tensorflow.convert_to_tensor(input_values, dtype=tensorflow.float32)
        constrained_parameters = self(input_tensor, training=False)
        return constrained_parameters.numpy()