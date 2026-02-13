import numpy as np
import tensorflow as tf
from scipy.signal import find_peaks


class NeuralPeriodEstimator:
    def __init__(self, window_length: int = 128, latent_size: int = 16, batch_size: int = 256, epochs: int = 5):
        self._window_length = int(window_length)
        self._latent_size = int(latent_size)
        self._batch_size = int(batch_size)
        self._epochs = int(epochs)
        self._encoder = None
        self._autoencoder = None

    def fit(self, sequence: np.ndarray) -> None:
        if sequence.ndim != 2:
            raise ValueError("sequence must have shape (time_steps, features)")

        sequence = sequence.astype(np.float32, copy=False)

        dataset = self._build_window_dataset(sequence)
        self._build_models(num_features=int(sequence.shape[1]))
        self._autoencoder.fit(dataset, epochs=self._epochs, verbose=0)

    def estimate_period(self, sequence: np.ndarray, min_period: int = 10) -> int:
        if self._encoder is None:
            raise RuntimeError("Call fit(sequence) before estimate_period(sequence).")

        sequence = sequence.astype(np.float32, copy=False)
        centered = sequence - sequence.mean(axis=0, keepdims=True)

        embeddings = self._encode_per_timestep(centered)
        autocorr = self._multivariate_autocorrelation_fft(embeddings)

        peaks, _ = find_peaks(autocorr[min_period:])
        if len(peaks) == 0:
            raise RuntimeError("No period detected")

        return int(peaks[0] + min_period)

    def _build_window_dataset(self, sequence: np.ndarray) -> tf.data.Dataset:
        dataset = tf.data.Dataset.from_tensor_slices(sequence)
        dataset = dataset.window(self._window_length, shift=1, drop_remainder=True)
        dataset = dataset.flat_map(lambda w: w.batch(self._window_length))
        dataset = dataset.batch(self._batch_size, drop_remainder=True)
        dataset = dataset.map(lambda x: (x, x))
        return dataset.prefetch(tf.data.AUTOTUNE)

    def _build_models(self, num_features: int) -> None:
        inp = tf.keras.Input(shape=(self._window_length, num_features))
        x = tf.keras.layers.GRU(self._latent_size, return_sequences=True)(inp)
        z = tf.keras.layers.GRU(self._latent_size, return_sequences=True)(x)

        dec = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(num_features)
        )(z)

        autoencoder = tf.keras.Model(inp, dec)
        autoencoder.compile(optimizer="adam", loss="mse")

        encoder = tf.keras.Model(inp, z)

        self._autoencoder = autoencoder
        self._encoder = encoder

    def _encode_per_timestep(self, sequence: np.ndarray) -> np.ndarray:
        T, d = sequence.shape
        w = self._window_length

        num_windows = T - w + 1
        windows = np.zeros((num_windows, w, d), dtype=np.float32)

        for i in range(num_windows):
            windows[i] = sequence[i:i + w]

        encoded = self._encoder.predict(windows, batch_size=self._batch_size, verbose=0)

        embeddings = np.zeros((T, self._latent_size), dtype=np.float64)
        counts = np.zeros(T, dtype=np.float64)

        for i in range(num_windows):
            for j in range(w):
                embeddings[i + j] += encoded[i, j]
                counts[i + j] += 1.0

        embeddings /= counts[:, None]
        return embeddings

    def _multivariate_autocorrelation_fft(self, sequence: np.ndarray) -> np.ndarray:
        T, d = sequence.shape
        fft_length = 1
        while fft_length < 2 * T:
            fft_length *= 2

        autocorr = np.zeros(T, dtype=np.float64)

        for k in range(d):
            s = sequence[:, k]
            f = np.fft.rfft(s, n=fft_length)
            c = np.fft.irfft(f * np.conj(f), n=fft_length)
            autocorr += c[:T]

        return autocorr / np.arange(T, 0, -1)
