from nd_data_science.machine_learning.model.application.sequence_to_sequence.kind.time_series.kind.auto_regression.vector.architecture import Architecture
from nd_data_science.machine_learning.model.application.sequence_to_sequence.kind.time_series.kind.auto_regression.vector.Training.learned_parameters import LearnedParameters
from nd_data_science.machine_learning.model.application.sequence_to_sequence.kind.time_series.predictor import Predictor as TimeSeriesPredictor
from nd_utility.oop.inheritance.overriding.override_from import override_from
import numpy as np
from nd_data_science.machine_learning.model.application.sequence_to_sequence.kind.time_series.predictor import \
    Predictor as TimeSeriesPredictor
from nd_utility.oop.inheritance.overriding.override_from import override_from


class Predictor(TimeSeriesPredictor):
    def __init__(self, architecture, learned_parameters):
        TimeSeriesPredictor.__init__(self, architecture, learned_parameters)

    @override_from(TimeSeriesPredictor, False, False)
    def get_predictions(self, input_sequence: np.ndarray) -> np.ndarray:
        if input_sequence is None:
            raise ValueError("input_sequence must not be None.")

        feature_count = self.get_architecture().get_feature_count()
        lag_order = self.get_architecture().get_lag_order()

        if input_sequence.shape[-1] != feature_count:
            raise ValueError("feature dimension mismatch.")

        time_axis = -2
        time_steps = int(input_sequence.shape[time_axis])
        if time_steps < lag_order:
            raise ValueError("input_sequence length must be >= lag_order.")

        coefficient_matrices = self.get_learned_parameters().get_coefficient_matrices()
        intercept = self.get_learned_parameters().get_intercept()
        horizon = int(self.get_learned_parameters().get_forecast_horizon())

        if input_sequence.ndim == 2:
            return self._forecast_one(
                input_sequence=input_sequence,
                lag_order=lag_order,
                coefficient_matrices=coefficient_matrices,
                intercept=intercept,
                horizon=horizon,
            )

        if input_sequence.ndim == 3:
            batch_size = int(input_sequence.shape[0])
            forecasts = []
            for index in range(batch_size):
                forecasts.append(self._forecast_one(
                    input_sequence=input_sequence[index],
                    lag_order=lag_order,
                    coefficient_matrices=coefficient_matrices,
                    intercept=intercept,
                    horizon=horizon,
                ))
            return np.stack(forecasts, axis=0)

        raise ValueError("Unsupported input_sequence rank. Expected 2D or 3D.")

    def _forecast_one(
            self,
            input_sequence: np.ndarray,
            lag_order: int,
            coefficient_matrices,
            intercept: np.ndarray | None,
            horizon: int,
    ) -> np.ndarray:
        history = input_sequence[-lag_order:, :].astype(float, copy=False)
        predictions = np.zeros((horizon, input_sequence.shape[1]), dtype=float)

        for step_index in range(horizon):
            predicted = np.zeros((input_sequence.shape[1],), dtype=float)

            for lag_index in range(1, lag_order + 1):
                past = history[-lag_index, :]
                predicted += past @ coefficient_matrices[lag_index - 1].T

            if intercept is not None:
                predicted += intercept

            predictions[step_index, :] = predicted
            history = np.vstack([history[1:, :], predicted.reshape(1, -1)])

        return predictions

