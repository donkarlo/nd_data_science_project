# file: nd_data_science/ml/period_estimation/training/training_config.py
from typing import Any, Dict


class TrainingConfig:
    def __init__(self, batch_size: int = 256, epochs: int = 5):
        self._batch_size = int(batch_size)
        self._epochs = int(epochs)

        if self._batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        if self._epochs <= 0:
            raise ValueError("epochs must be > 0.")

    def get_batch_size(self) -> int:
        return int(self._batch_size)

    def get_epochs(self) -> int:
        return int(self._epochs)

    def to_dict(self) -> Dict[str, Any]:
        return {"batch_size": int(self._batch_size), "epochs": int(self._epochs)}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TrainingConfig":
        if not isinstance(payload, dict):
            raise TypeError("payload must be dict.")
        return cls(batch_size=int(payload["batch_size"]), epochs=int(payload["epochs"]))


# file: nd_data_science/ml/period_estimation/architecture/architecture.py
from typing import Any, Dict, Tuple

import tensorflow as tf
from tensorflow.keras import Model as TfModel
from tensorflow.keras import layers as TfLayers


class Architecture:
    def __init__(self, window_length: int = 128, latent_size: int = 16):
        self._window_length = int(window_length)
        self._latent_size = int(latent_size)

        if self._window_length <= 1:
            raise ValueError("window_length must be > 1.")
        if self._latent_size <= 0:
            raise ValueError("latent_size must be > 0.")

    def get_window_length(self) -> int:
        return int(self._window_length)

    def get_latent_size(self) -> int:
        return int(self._latent_size)

    def build_models(self, num_features: int) -> Tuple[TfModel, TfModel]:
        num_features = int(num_features)
        if num_features <= 0:
            raise ValueError("num_features must be > 0.")

        inp = tf.keras.Input(shape=(self._window_length, num_features), dtype=tf.float32, name="x_in")

        x = TfLayers.GRU(self._latent_size, return_sequences=True, name="enc_gru_1")(inp)
        z = TfLayers.GRU(self._latent_size, return_sequences=True, name="enc_gru_2")(x)

        dec = TfLayers.TimeDistributed(TfLayers.Dense(num_features), name="decoder")(z)

        autoencoder = tf.keras.Model(inp, dec, name="gru_autoencoder")
        autoencoder.compile(optimizer="adam", loss="mse")

        encoder = tf.keras.Model(inp, z, name="gru_encoder")

        return autoencoder, encoder

    def to_dict(self) -> Dict[str, Any]:
        return {"window_length": int(self._window_length), "latent_size": int(self._latent_size)}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Architecture":
        if not isinstance(payload, dict):
            raise TypeError("payload must be dict.")
        return cls(window_length=int(payload["window_length"]), latent_size=int(payload["latent_size"]))


# file: nd_data_science/ml/period_estimation/data/window_dataset_builder.py
import numpy as np
import tensorflow as tf


class WindowDatasetBuilder:
    def __init__(self, window_length: int, batch_size: int):
        self._window_length = int(window_length)
        self._batch_size = int(batch_size)

        if self._window_length <= 1:
            raise ValueError("window_length must be > 1.")
        if self._batch_size <= 0:
            raise ValueError("batch_size must be > 0.")

    def build(self, sequence: np.ndarray) -> tf.data.Dataset:
        if not isinstance(sequence, np.ndarray):
            raise TypeError("sequence must be np.ndarray.")
        if sequence.ndim != 2:
            raise ValueError("sequence must have shape (time_steps, features).")

        sequence = sequence.astype(np.float32, copy=False)

        dataset = tf.data.Dataset.from_tensor_slices(sequence)
        dataset = dataset.window(self._window_length, shift=1, drop_remainder=True)
        dataset = dataset.flat_map(lambda w: w.batch(self._window_length))
        dataset = dataset.batch(self._batch_size, drop_remainder=True)
        dataset = dataset.map(lambda x: (x, x))
        return dataset.prefetch(tf.data.AUTOTUNE)


# file: nd_data_science/ml/period_estimation/data/window_dataset_builder.py
import numpy as np
import tensorflow as tf


class WindowDatasetBuilder:
    def __init__(self, window_length: int, batch_size: int):
        self._window_length = int(window_length)
        self._batch_size = int(batch_size)

        if self._window_length <= 1:
            raise ValueError("window_length must be > 1.")
        if self._batch_size <= 0:
            raise ValueError("batch_size must be > 0.")

    def build(self, sequence: np.ndarray) -> tf.data.Dataset:
        if not isinstance(sequence, np.ndarray):
            raise TypeError("sequence must be np.ndarray.")
        if sequence.ndim != 2:
            raise ValueError("sequence must have shape (time_steps, features).")

        sequence = sequence.astype(np.float32, copy=False)

        dataset = tf.data.Dataset.from_tensor_slices(sequence)
        dataset = dataset.window(self._window_length, shift=1, drop_remainder=True)
        dataset = dataset.flat_map(lambda w: w.batch(self._window_length))
        dataset = dataset.batch(self._batch_size, drop_remainder=True)
        dataset = dataset.map(lambda x: (x, x))
        return dataset.prefetch(tf.data.AUTOTUNE)


# file: nd_data_science/ml/period_estimation/embedding/per_timestep_embedder.py
import numpy as np
import tensorflow as tf


class PerTimestepEmbedder:
    def __init__(self, encoder: tf.keras.Model, window_length: int, latent_size: int, batch_size: int):
        self._encoder = encoder
        self._window_length = int(window_length)
        self._latent_size = int(latent_size)
        self._batch_size = int(batch_size)

        if self._window_length <= 1:
            raise ValueError("window_length must be > 1.")
        if self._latent_size <= 0:
            raise ValueError("latent_size must be > 0.")
        if self._batch_size <= 0:
            raise ValueError("batch_size must be > 0.")

    def embed(self, sequence: np.ndarray) -> np.ndarray:
        if not isinstance(sequence, np.ndarray):
            raise TypeError("sequence must be np.ndarray.")
        if sequence.ndim != 2:
            raise ValueError("sequence must have shape (time_steps, features).")

        sequence = sequence.astype(np.float32, copy=False)

        time_steps = int(sequence.shape[0])
        feature_count = int(sequence.shape[1])
        window_length = int(self._window_length)

        number_of_windows = time_steps - window_length + 1
        if number_of_windows <= 0:
            raise ValueError("sequence is shorter than window_length.")

        windows = np.zeros((number_of_windows, window_length, feature_count), dtype=np.float32)
        for i in range(number_of_windows):
            windows[i] = sequence[i:i + window_length]

        encoded = self._encoder.predict(windows, batch_size=self._batch_size, verbose=0)

        embeddings = np.zeros((time_steps, self._latent_size), dtype=np.float64)
        counts = np.zeros(time_steps, dtype=np.float64)

        for i in range(number_of_windows):
            for j in range(window_length):
                embeddings[i + j] += encoded[i, j]
                counts[i + j] += 1.0

        embeddings /= counts[:, None]
        return embeddings


# file: nd_data_science/ml/period_estimation/signal/autocorrelation_fft.py
import numpy as np


class AutocorrelationFft:
    def get_multivariate_autocorrelation(self, sequence: np.ndarray) -> np.ndarray:
        if not isinstance(sequence, np.ndarray):
            raise TypeError("sequence must be np.ndarray.")
        if sequence.ndim != 2:
            raise ValueError("sequence must have shape (time_steps, features).")

        time_steps = int(sequence.shape[0])
        feature_count = int(sequence.shape[1])

        fft_length = 1
        while fft_length < 2 * time_steps:
            fft_length *= 2

        autocorr = np.zeros(time_steps, dtype=np.float64)

        for k in range(feature_count):
            s = sequence[:, k]
            f = np.fft.rfft(s, n=fft_length)
            c = np.fft.irfft(f * np.conj(f), n=fft_length)
            autocorr += c[:time_steps]

        return autocorr / np.arange(time_steps, 0, -1)


# file: nd_data_science/ml/period_estimation/predicting/period_length_predictor.py
import numpy as np
from scipy.signal import find_peaks

from nd_data_science.ml.period_estimation.embedding.per_timestep_embedder import PerTimestepEmbedder
from nd_data_science.ml.period_estimation.signal.autocorrelation_fft import AutocorrelationFft


class PeriodLengthPredictor:
    def __init__(self, embedder: PerTimestepEmbedder):
        self._embedder = embedder
        self._autocorrelation = AutocorrelationFft()

    def estimate_period(self, sequence: np.ndarray, min_period: int = 10) -> int:
        if not isinstance(sequence, np.ndarray):
            raise TypeError("sequence must be np.ndarray.")
        if sequence.ndim != 2:
            raise ValueError("sequence must have shape (time_steps, features).")

        min_period = int(min_period)
        if min_period < 1:
            raise ValueError("min_period must be >= 1.")

        sequence = sequence.astype(np.float32, copy=False)
        centered = sequence - sequence.mean(axis=0, keepdims=True)

        embeddings = self._embedder.embed(centered)
        autocorr = self._autocorrelation.get_multivariate_autocorrelation(embeddings)

        peaks, _ = find_peaks(autocorr[min_period:])
        if len(peaks) == 0:
            raise RuntimeError("No period detected.")

        return int(peaks[0] + min_period)


# file: nd_data_science/ml/period_estimation/neural_period_estimator.py
import numpy as np

from nd_data_science.ml.period_estimation.architecture.architecture import Architecture
from nd_data_science.ml.period_estimation.embedding.per_timestep_embedder import PerTimestepEmbedder
from nd_data_science.ml.period_estimation.predicting.period_length_predictor import PeriodLengthPredictor
from nd_data_science.ml.period_estimation.training.trainer import Trainer
from nd_data_science.ml.period_estimation.training.training_config import TrainingConfig


class NeuralPeriodEstimator:
    def __init__(self, architecture: Architecture, training_config: TrainingConfig):
        self._architecture = architecture
        self._training_config = training_config
        self._trainer = Trainer(self._architecture, self._training_config)
        self._predictor = None

    def fit(self, sequence: np.ndarray) -> None:
        self._trainer.fit(sequence)

        encoder = self._trainer.get_encoder()
        embedder = PerTimestepEmbedder(
            encoder,
            self._architecture.get_window_length(),
            self._architecture.get_latent_size(),
            self._training_config.get_batch_size(),
        )
        self._predictor = PeriodLengthPredictor(embedder)

    def estimate_period(self, sequence: np.ndarray, min_period: int = 10) -> int:
        if self._predictor is None:
            raise RuntimeError("Call fit(sequence) before estimate_period(sequence).")
        return self._predictor.estimate_period(sequence, min_period=min_period)
