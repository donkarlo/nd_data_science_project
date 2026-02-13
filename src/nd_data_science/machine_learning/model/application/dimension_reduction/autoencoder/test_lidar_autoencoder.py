from nd_data_science.machine_learning.model.application.dimension_reduction.autoencoder.autoencoder import Autoencoder
from nd_utility.data.storage.kind.file.pkl.pkl import Pkl
from nd_utility.os.file_system.file.file import File
from nd_utility.os.file_system.path.path import Path

import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt


class TemporalWindowDataset(Dataset):
    """
    Dataset for sliding-window vectors built from norm of LiDAR deltas.
    Each sample is a window: [||delta_{t-w+1}||, ..., ||delta_t||].
    """

    def __init__(self, window_vectors: np.ndarray) -> None:
        self._window_vectors = window_vectors.astype(np.float32)

    def __len__(self) -> int:
        return self._window_vectors.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        sample = self._window_vectors[index]
        return torch.from_numpy(sample)


class LidarTemporalAutoencoder:
    """
    Builds sliding-window norm(delta) vectors from LiDAR scans,
    trains an autoencoder on them, and provides a low-dimensional embedding.
    """

    def __init__(self, window_size: int, bottleneck_dim: int) -> None:
        self._window_size = window_size
        self._bottleneck_dim = bottleneck_dim

        # ----- Load LiDAR scans -----
        path = Path(
            "/home/donkarlo/Dropbox/phd/data/experiements/oldest/robots/uav1/mind/memory/long_term/explicit/episodic/normal/lidar/lidar.pkl"
        )

        os_file = File.init_from_path(path)
        pickle = Pkl(os_file, False)
        scan_sliced_values = pickle.load()
        scan_values = scan_sliced_values.get_values()

        lidar_vectors = []
        for scan_value in scan_values:
            lidar_vectors.append(
                scan_value
                .get_formatted_data()
                .get_vector_representation()
                .get_components()
            )

        lidar_vectors = np.array(lidar_vectors, dtype=np.float64)

        max_range = 15.0
        lidar_vectors[~np.isfinite(lidar_vectors)] = max_range
        lidar_vectors[lidar_vectors > max_range] = max_range
        lidar_vectors[lidar_vectors < 0.0] = 0.0

        self._lidar_vectors = lidar_vectors

        # ----- Build norm(delta) time series -----
        self._delta_norms, self._delta_norms_mean, self._delta_norms_std = \
            self._build_delta_norm_series()

        # ----- Build sliding-window vectors -----
        self._window_vectors = self._build_sliding_windows()

        self._input_dim = self._window_vectors.shape[1]

        # ----- Set device and transformer_model -----
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device_name)

        self._model = Autoencoder(self._input_dim, self._bottleneck_dim)
        self._model.to(self._device)

    def _build_delta_norm_series(self) -> tuple[np.ndarray, float, float]:
        """
        Computes ||delta_t||_2 over LiDAR scans and normalizes it.
        Returns normalized series and its mean, std.
        """
        # lidar_vectors shape: (T, D)
        deltas = np.diff(self._lidar_vectors, axis=0)  # shape: (T-1, D)
        norms = np.linalg.norm(deltas, axis=1)  # shape: (T-1,)

        norms = norms.astype(np.float32)

        mean_value = float(np.mean(norms))
        std_value = float(np.std(norms))

        if std_value == 0.0:
            std_value = 1.0

        normalized_norms = (norms - mean_value) / std_value

        return normalized_norms, mean_value, std_value

    def _build_sliding_windows(self) -> np.ndarray:
        """
        Builds sliding windows over the normalized delta norms.
        Result shape: (num_windows, window_size).
        """
        series = self._delta_norms
        total_length = series.shape[0]

        if total_length < self._window_size:
            raise ValueError("Time series is shorter than the window size.")

        num_windows = total_length - self._window_size + 1
        windows = np.empty((num_windows, self._window_size), dtype=np.float32)

        for index in range(num_windows):
            start_index = index
            end_index = index + self._window_size
            windows[index] = series[start_index:end_index]

        return windows

    def _create_dataloader(self, batch_size: int, shuffle: bool) -> DataLoader:
        dataset = TemporalWindowDataset(self._window_vectors)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False
        )
        return dataloader

    def train(self, epochs: int, batch_size: int, learning_rate: float) -> None:
        dataloader = self._create_dataloader(batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        self._model.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            number_of_batches = 0

            for batch in dataloader:
                batch = batch.to(self._device)

                optimizer.zero_grad()
                reconstruction = self._model(batch)
                loss = criterion(reconstruction, batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                number_of_batches += 1

            if number_of_batches > 0:
                epoch_loss = epoch_loss / float(number_of_batches)
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.6f}")

    def encode_all(self, batch_size: int) -> np.ndarray:
        """
        Returns a NumPy array of shape (num_windows, bottleneck_dim)
        with the encoded representation of all sliding windows.
        """
        dataloader = self._create_dataloader(batch_size, shuffle=False)

        self._model.eval()
        latent_list = []

        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self._device)
                z = self._model.encode(batch)
                z_cpu = z.cpu().numpy()
                latent_list.append(z_cpu)

        if len(latent_list) == 0:
            return np.zeros((0, self._bottleneck_dim), dtype=np.float32)

        latents = np.concatenate(latent_list, axis=0)
        return latents

    def get_delta_norms(self) -> np.ndarray:
        """
        Returns the normalized delta norms time series.
        """
        return self._delta_norms


if __name__ == "__main__":
    # Configuration similar in spirit to "UMAP on sliding-window norm(delta)"
    window_size = 30
    bottleneck_dim = 2

    temporal_autoencoder = LidarTemporalAutoencoder(
        window_size=window_size,
        bottleneck_dim=bottleneck_dim
    )

    temporal_autoencoder.train(
        epochs=40,
        batch_size=256,
        learning_rate=1e-3
    )

    latents_2d = temporal_autoencoder.encode_all(batch_size=512)
    delta_norms = temporal_autoencoder.get_delta_norms()

    print("Latent shape:", latents_2d.shape)

    # ----- 2D plot of latent space with color = current ||delta|| -----
    # We have one delta_norm per time overlap_size (T-1) and one window per valid end index.
    # The last element in each window corresponds to a time index in delta_norms.
    num_windows = latents_2d.shape[0]
    aligned_norms = delta_norms[window_size - 1:window_size - 1 + num_windows]

    if aligned_norms.shape[0] != num_windows:
        raise RuntimeError("Mismatch between value of windows and color series length.")

    x_coordinates = latents_2d[:, 0]
    y_coordinates = latents_2d[:, 1]

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        x_coordinates,
        y_coordinates,
        c=aligned_norms,
        s=3,
        cmap="viridis"
    )
    plt.colorbar(scatter, label="||delta|| (normalized)")
    plt.title("Autoencoder on sliding-window norm(delta)")
    plt.xlabel("Latent 1")
    plt.ylabel("Latent 2")
    plt.tight_layout()
    plt.show()
