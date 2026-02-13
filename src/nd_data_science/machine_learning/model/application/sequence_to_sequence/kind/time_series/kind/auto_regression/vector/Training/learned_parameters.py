import numpy as np
from typing import Sequence


class LearnedParameters:
    def __init__(
            self,
            coefficient_matrices: Sequence[np.ndarray],
            noise_covariance: np.ndarray,
            intercept: np.ndarray | None = None,
            forecast_horizon: int | None = None,
    ):
        self._coefficient_matrices = list(coefficient_matrices)
        self._noise_covariance = noise_covariance
        self._intercept = intercept

        if forecast_horizon is None:
            self._forecast_horizon = 1
        else:
            self._forecast_horizon = int(forecast_horizon)

        if len(self._coefficient_matrices) == 0:
            raise ValueError("coefficient_matrices must not be empty.")

        matrix_shape = self._coefficient_matrices[0].shape
        for matrix in self._coefficient_matrices:
            if matrix.shape != matrix_shape:
                raise ValueError("All coefficient matrices must have the same shape.")

        if self._noise_covariance.shape[0] != self._noise_covariance.shape[1]:
            raise ValueError("noise_covariance must be a square matrix.")

        if self._intercept is not None:
            if self._intercept.ndim != 1:
                raise ValueError("intercept must be a 1D vector.")
            if self._intercept.shape[0] != matrix_shape[0]:
                raise ValueError("intercept dimension must match coefficient matrix dimension.")

        if self._forecast_horizon <= 0:
            raise ValueError("forecast_horizon must be > 0.")

    def get_coefficient_matrices(self) -> Sequence[np.ndarray]:
        return self._coefficient_matrices

    def get_feature_count(self) -> int:
        return self._coefficient_matrices[0].shape[0]

    def get_noise_covariance(self) -> np.ndarray:
        return self._noise_covariance

    def get_intercept(self) -> np.ndarray | None:
        return self._intercept

    def get_forecast_horizon(self) -> int:
        return self._forecast_horizon