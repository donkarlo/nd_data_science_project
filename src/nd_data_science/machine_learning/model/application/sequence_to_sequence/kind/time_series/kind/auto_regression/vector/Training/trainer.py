import numpy as np
from numpy.linalg import LinAlgError
from typing import Sequence

from nd_data_science.machine_learning.model.application.sequence_to_sequence.kind.time_series.kind.auto_regression.vector.architecture import \
    Architecture
from nd_data_science.machine_learning.model.application.sequence_to_sequence.kind.time_series.kind.auto_regression.vector.Training.config import \
    Config
from nd_data_science.machine_learning.model.application.sequence_to_sequence.kind.time_series.kind.auto_regression.vector.Training.learned_parameters import \
    LearnedParameters
from nd_data_science.machine_learning.model.application.sequence_to_sequence.kind.time_series.training.training import \
    Training as TimeSeriesTraining, Training
from nd_utility.oop.inheritance.overriding.override_from import override_from


class Trainer(TimeSeriesTraining):
    def __init__(self, architecture: Architecture, config: Config, input_target_array: np.ndarray):
        TimeSeriesTraining.__init__(self, architecture, config, input_target_array)

        if input_target_array is None:
            raise ValueError("input_target_array must not be None.")
        if not isinstance(input_target_array, np.ndarray):
            raise ValueError("input_target_array must be a numpy.ndarray.")

        self._input_target_array = input_target_array
        self._learned_parameters = None

    @override_from(Training, False, False)
    def get_learned_parameters(self) -> LearnedParameters:
        if self._learned_parameters is None:
            architecture = self.get_architecture()
            config = self.get_config()

            include_intercept = architecture.get_include_intercept()

            input_series_matrix = self._extract_input_series_matrix(input_target_array=self._input_target_array)

            maximum_lag_order = int(architecture.get_lag_order())
            if maximum_lag_order <= 0:
                raise ValueError("config.lag_order must be > 0.")

            if config.get_select_lag_by() is None:
                selected_lag_order = maximum_lag_order
            else:
                selected_lag_order = self._select_lag_order(
                    input_array=input_series_matrix,
                    maximum_lag_order=maximum_lag_order,
                    include_intercept=include_intercept,
                    criterion=config.get_select_lag_by(),
                    fit_method=config.get_fit_method(),
                    regularization_strength=config.get_regularization_strength(),
                )

            forecast_horizon = self._infer_forecast_horizon(input_target_array=self._input_target_array)

            self._learned_parameters = self._fit_var(
                input_array=input_series_matrix,
                lag_order=selected_lag_order,
                include_intercept=include_intercept,
                fit_method=config.get_fit_method(),
                regularization_strength=config.get_regularization_strength(),
                forecast_horizon=forecast_horizon,
            )

        return self._learned_parameters

    def _extract_input_series_matrix(self, input_target_array: np.ndarray) -> np.ndarray:
        """
        Accepts any of these layouts and returns (time_steps, feature_count):

            - (B, 2, T, F) numeric: uses input_target_array[:, 0, :, :] and flattens to (B*T, F)
            - (2, T, F) numeric: uses input_target_array[0, :, :] -> (T, F)
            - (T, F) numeric: uses it directly
            - (pair_count, 2) object: each cell is a 1D/2D numeric array; concatenates inputs along time

        Notes:
            - VAR is not a batched model. Batch is converted into a single long series by concatenation.
        """
        if input_target_array.ndim == 4:
            if input_target_array.shape[1] != 2:
                raise ValueError("For 4D input_target_array, expected shape (B, 2, T, F).")
            input_tensor = np.asarray(input_target_array[:, 0, :, :], dtype=float)
            return input_tensor.reshape(input_tensor.shape[0] * input_tensor.shape[1], input_tensor.shape[2])

        if input_target_array.ndim == 3:
            if input_target_array.shape[0] != 2:
                raise ValueError("For 3D input_target_array, expected shape (2, T, F).")
            return np.asarray(input_target_array[0, :, :], dtype=float)

        if input_target_array.ndim == 2:
            if input_target_array.shape[1] == 2 and input_target_array.dtype == object:
                return self._concat_object_input_pairs(input_target_pairs=input_target_array)

            return np.asarray(input_target_array, dtype=float)

        raise ValueError("Unsupported input_target_array shape for VAR training.")

    def _concat_object_input_pairs(self, input_target_pairs: np.ndarray) -> np.ndarray:
        """
        training_input_target_pairs: shape (pair_count, 2), dtype object
        training_input_target_pairs[:, 0] elements must be 1D or 2D numeric arrays.
        """
        input_segments: list[np.ndarray] = []

        for index in range(input_target_pairs.shape[0]):
            segment = np.asarray(input_target_pairs[index, 0], dtype=float)

            if segment.ndim == 1:
                segment = segment.reshape(-1, 1)
            elif segment.ndim != 2:
                raise ValueError("Each input segment must be a 1D or 2D numeric array.")

            input_segments.append(segment)

        if len(input_segments) == 0:
            raise ValueError("training_input_target_pairs must not be empty.")

        feature_count = input_segments[0].shape[1]
        for segment in input_segments:
            if segment.shape[1] != feature_count:
                raise ValueError("All input segments must have the same feature_count.")

        return np.concatenate(input_segments, axis=0)

    def _fit_var(
            self,
            input_array: np.ndarray,
            lag_order: int,
            include_intercept: bool,
            fit_method: str,
            regularization_strength: float | None,
            forecast_horizon: int,
    ) -> LearnedParameters:
        design_matrix, target_matrix = self._build_design_matrices(
            input_array=input_array,
            lag_order=lag_order,
            include_intercept=include_intercept,
        )

        coefficient_block = self._estimate_coefficients(
            design_matrix=design_matrix,
            target_matrix=target_matrix,
            fit_method=fit_method,
            regularization_strength=regularization_strength,
            include_intercept=include_intercept,
        )

        coefficient_matrices, intercept = self._unpack_coefficients(
            coefficient_block=coefficient_block,
            feature_count=input_array.shape[1],
            lag_order=lag_order,
            include_intercept=include_intercept,
        )

        residuals = target_matrix - (design_matrix @ coefficient_block)
        noise_covariance = self._estimate_noise_covariance(
            residuals=residuals,
            parameter_count=coefficient_block.shape[0],
        )

        return LearnedParameters(
            coefficient_matrices=coefficient_matrices,
            noise_covariance=noise_covariance,
            intercept=intercept,
            forecast_horizon=forecast_horizon,
        )

    def _build_design_matrices(self, input_array: np.ndarray, lag_order: int, include_intercept: bool) -> tuple[
        np.ndarray, np.ndarray]:
        time_steps = int(input_array.shape[0])
        feature_count = int(input_array.shape[1])

        if time_steps <= lag_order:
            raise ValueError("time_steps must be > lag_order.")

        observation_count = time_steps - lag_order
        regressor_count = lag_order * feature_count

        if include_intercept:
            design_matrix = np.zeros((observation_count, regressor_count + 1), dtype=float)
            design_matrix[:, 0] = 1.0
            column_offset = 1
        else:
            design_matrix = np.zeros((observation_count, regressor_count), dtype=float)
            column_offset = 0

        target_matrix = np.zeros((observation_count, feature_count), dtype=float)

        for row_index in range(observation_count):
            target_time_index = row_index + lag_order
            target_matrix[row_index, :] = input_array[target_time_index, :]

            for lag_index in range(1, lag_order + 1):
                source_time_index = target_time_index - lag_index
                start_column = column_offset + (lag_index - 1) * feature_count
                end_column = start_column + feature_count
                design_matrix[row_index, start_column:end_column] = input_array[source_time_index, :]

        return design_matrix, target_matrix

    def _estimate_coefficients(
            self,
            design_matrix: np.ndarray,
            target_matrix: np.ndarray,
            fit_method: str,
            regularization_strength: float | None,
            include_intercept: bool,
    ) -> np.ndarray:
        if fit_method == "ols":
            return self._estimate_coefficients_ols(design_matrix=design_matrix, target_matrix=target_matrix)

        if fit_method == "ridge":
            if regularization_strength is None:
                raise ValueError("regularization_strength must be provided for ridge.")
            return self._estimate_coefficients_ridge(
                design_matrix=design_matrix,
                target_matrix=target_matrix,
                regularization_strength=regularization_strength,
                include_intercept=include_intercept,
            )

        if fit_method == "lasso":
            if regularization_strength is None:
                raise ValueError("regularization_strength must be provided for lasso.")
            return self._estimate_coefficients_lasso(
                design_matrix=design_matrix,
                target_matrix=target_matrix,
                regularization_strength=regularization_strength,
                include_intercept=include_intercept,
            )

        raise ValueError("fit_method must be one of {'ols', 'ridge', 'lasso'}.")

    def _estimate_coefficients_ols(self, design_matrix: np.ndarray, target_matrix: np.ndarray) -> np.ndarray:
        try:
            return np.linalg.lstsq(design_matrix, target_matrix, rcond=None)[0]
        except LinAlgError as exception:
            raise RuntimeError("OLS estimation failed due to a linear algebra error.") from exception

    def _estimate_coefficients_ridge(
            self,
            design_matrix: np.ndarray,
            target_matrix: np.ndarray,
            regularization_strength: float,
            include_intercept: bool,
    ) -> np.ndarray:
        parameter_count = int(design_matrix.shape[1])
        identity_matrix = np.eye(parameter_count, dtype=float)

        if include_intercept:
            identity_matrix[0, 0] = 0.0

        left_matrix = (design_matrix.T @ design_matrix) + (regularization_strength * identity_matrix)
        right_matrix = design_matrix.T @ target_matrix

        try:
            return np.linalg.solve(left_matrix, right_matrix)
        except LinAlgError as exception:
            raise RuntimeError("Ridge estimation failed due to a linear algebra error.") from exception

    def _estimate_coefficients_lasso(
            self,
            design_matrix: np.ndarray,
            target_matrix: np.ndarray,
            regularization_strength: float,
            include_intercept: bool,
    ) -> np.ndarray:
        try:
            from sklearn.linear_model import MultiTaskLasso
        except Exception as exception:
            raise RuntimeError("Lasso requires scikit-learn to be installed.") from exception

        if include_intercept:
            feature_matrix = design_matrix[:, 1:]
        else:
            feature_matrix = design_matrix

        model = MultiTaskLasso(alpha=float(regularization_strength), fit_intercept=False, max_iter=10000)
        model.fit(feature_matrix, target_matrix)

        coefficient_without_intercept = model.coef_.T

        if include_intercept:
            intercept_row = np.zeros((1, target_matrix.shape[1]), dtype=float)
            return np.vstack([intercept_row, coefficient_without_intercept])

        return coefficient_without_intercept

    def _unpack_coefficients(
            self,
            coefficient_block: np.ndarray,
            feature_count: int,
            lag_order: int,
            include_intercept: bool,
    ) -> tuple[Sequence[np.ndarray], np.ndarray | None]:
        if include_intercept:
            intercept = coefficient_block[0, :].copy()
            coefficient_rows = coefficient_block[1:, :]
        else:
            intercept = None
            coefficient_rows = coefficient_block

        coefficient_matrices: list[np.ndarray] = []
        for lag_index in range(lag_order):
            start_row = lag_index * feature_count
            end_row = start_row + feature_count
            matrix = coefficient_rows[start_row:end_row, :].T
            coefficient_matrices.append(matrix)

        return coefficient_matrices, intercept

    def _estimate_noise_covariance(self, residuals: np.ndarray, parameter_count: int) -> np.ndarray:
        observation_count = int(residuals.shape[0])
        degrees_of_freedom = observation_count - int(parameter_count)

        if degrees_of_freedom <= 0:
            degrees_of_freedom = observation_count

        return (residuals.T @ residuals) / float(degrees_of_freedom)

    def _select_lag_order(
            self,
            input_array: np.ndarray,
            maximum_lag_order: int,
            include_intercept: bool,
            criterion: str,
            fit_method: str,
            regularization_strength: float | None,
    ) -> int:
        if maximum_lag_order <= 0:
            raise ValueError("maximum_lag_order must be > 0.")

        best_lag_order = 1
        best_score = None

        for candidate_lag_order in range(1, maximum_lag_order + 1):
            learned_parameters = self._fit_var(
                input_array=input_array,
                lag_order=candidate_lag_order,
                include_intercept=include_intercept,
                fit_method=fit_method,
                regularization_strength=regularization_strength,
            )

            noise_covariance = learned_parameters.get_noise_covariance()
            observation_count = int(input_array.shape[0]) - int(candidate_lag_order)

            score = self._information_criterion(
                noise_covariance=noise_covariance,
                observation_count=observation_count,
                feature_count=int(input_array.shape[1]),
                lag_order=int(candidate_lag_order),
                include_intercept=include_intercept,
                criterion=criterion,
            )

            if best_score is None or score < best_score:
                best_score = score
                best_lag_order = candidate_lag_order

        return best_lag_order

    def _information_criterion(
            self,
            noise_covariance: np.ndarray,
            observation_count: int,
            feature_count: int,
            lag_order: int,
            include_intercept: bool,
            criterion: str,
    ) -> float:
        if observation_count <= 0:
            raise ValueError("observation_count must be > 0.")

        try:
            determinant = float(np.linalg.det(noise_covariance))
        except LinAlgError as exception:
            raise RuntimeError("Failed to compute determinant for information criterion.") from exception

        if determinant <= 0.0:
            determinant = 1e-12

        log_determinant = float(np.log(determinant))

        parameter_count = feature_count * feature_count * lag_order
        if include_intercept:
            parameter_count += feature_count

        if criterion == "aic":
            return log_determinant + (2.0 * float(parameter_count) / float(observation_count))

        if criterion == "bic":
            return log_determinant + (
                        np.log(float(observation_count)) * float(parameter_count) / float(observation_count))

        if criterion == "hqic":
            return log_determinant + (2.0 * np.log(np.log(float(observation_count))) * float(parameter_count) / float(
                observation_count))

        raise ValueError("criterion must be one of {'aic', 'bic', 'hqic'}.")

    def _infer_forecast_horizon(self, input_target_array: np.ndarray) -> int:
        if input_target_array is None:
            raise ValueError("input_target_array must not be None.")

        if input_target_array.ndim == 4:
            if input_target_array.shape[1] != 2:
                raise ValueError("For 4D input_target_array, expected shape (B, 2, T, F).")
            return int(input_target_array.shape[2])

        if input_target_array.ndim == 3:
            if input_target_array.shape[0] != 2:
                raise ValueError("For 3D input_target_array, expected shape (2, T, F).")
            return int(input_target_array.shape[1])

        if input_target_array.ndim == 2:
            if input_target_array.shape[1] == 2 and input_target_array.dtype == object:
                first_target = np.asarray(input_target_array[0, 1], dtype=float)
                if first_target.ndim == 1:
                    return int(first_target.shape[0])
                if first_target.ndim == 2:
                    return int(first_target.shape[0])
                raise ValueError("Each target segment must be a 1D or 2D numeric array.")
            # numeric (T,F)
            return int(input_target_array.shape[0])

        raise ValueError("Unsupported input_target_array shape for horizon inference.")
