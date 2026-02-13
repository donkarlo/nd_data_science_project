import numpy as np


class VectorAutoRegressiveModel:
    """
    VAR(1) model with least-squares fitting.

    Notes:
        - Works on ordered samples (no time needed), shape: (n_samples, n_dimensions)
        - Fits: x_k ≈ A @ x_{k-1} + b
        - Uses ordinary least squares.
    """

    def __init__(self):
        self._matrix_a = None
        self._bias_b = None
        self._is_fitted = False

    def fit(self, samples: np.ndarray) -> None:
        """
        Fit VAR(1) parameters (A, b) from samples.

        Args:
            samples: array of shape (n_samples, n_dimensions)

        Raises:
            ValueError: if input shape is invalid or too short.
        """
        if samples is None:
            raise ValueError("samples must not be None.")
        if samples.ndim != 2:
            raise ValueError("samples must have shape (n_samples, n_dimensions).")
        if samples.shape[0] < 2:
            raise ValueError("samples must contain at least 2 rows.")

        previous = samples[:-1]  # x_{k-1}
        current = samples[1:]  # x_k

        # Add a column of ones to learn bias term b
        ones = np.ones((previous.shape[0], 1), dtype=previous.dtype)
        design_matrix = np.concatenate([previous, ones], axis=1)  # [x_{k-1}, 1]

        # Solve design_matrix @ W ≈ current  -> W has shape (n_dimensions + 1, n_dimensions)
        weight_matrix, residuals, rank, singular_values = np.linalg.lstsq(design_matrix, current, rcond=None)

        self._matrix_a = weight_matrix[:-1, :].T  # (n_dim, n_dim)
        self._bias_b = weight_matrix[-1, :]  # (n_dim,)
        self._is_fitted = True

    def predict_next(self, last_sample: np.ndarray) -> np.ndarray:
        """
        Predict next vector from the last observed vector.

        Args:
            last_sample: array of shape (n_dimensions,)

        Returns:
            predicted_next: array of shape (n_dimensions,)

        Raises:
            RuntimeError: if model is not fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Architecture is not fitted. Call fit() first.")
        if last_sample is None:
            raise ValueError("last_sample must not be None.")
        if last_sample.ndim != 1:
            raise ValueError("last_sample must have shape (n_dimensions,).")

        return (self._matrix_a @ last_sample) + self._bias_b

    def predict_sequence(self, initial_sample: np.ndarray, steps: int) -> np.ndarray:
        """
        Predict multiple future steps by rolling the model forward.

        Args:
            initial_sample: array of shape (n_dimensions,)
            steps: number of steps to predict (>= 1)

        Returns:
            predictions: array of shape (steps, n_dimensions)
        """
        if steps < 1:
            raise ValueError("steps must be >= 1.")

        predictions = []
        current = np.asarray(initial_sample, dtype=float)

        for _ in range(steps):
            next_value = self.predict_next(current)
            predictions.append(next_value)
            current = next_value

        return np.asarray(predictions)

    def get_matrix_a(self) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Architecture is not fitted.")
        return self._matrix_a.copy()

    def get_bias_b(self) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Architecture is not fitted.")
        return self._bias_b.copy()


if __name__ == "__main__":
    # Example: 2D "GPS-like" sequence (lat, lon) without time, only order
    samples = np.array(
        [
            [47.0700, 15.4400],
            [47.0701, 15.4402],
            [47.0702, 15.4404],
            [47.0704, 15.4407],
            [47.0707, 15.4411],
        ],
        dtype=float,
    )



    model = VectorAutoRegressiveModel()
    model.fit(samples)

    last_point = samples[-1]
    next_point = model.predict_next(last_point)
    future = model.predict_sequence(last_point, steps=5)

    print("A:\n", model.get_matrix_a())
    print("b:\n", model.get_bias_b())
    print("next:\n", next_point)
    print("future:\n", future)
