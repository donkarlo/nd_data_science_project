from nd_sociomind.experiment.parts.oldest.uav1_300k_normal_time_position_modality import \
    Uav1300kNormalTimePositionModality

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


class DbscanGps:
    def __init__(self, data_slice, eps=0.5, min_samples=5):
        self.data_slice = data_slice
        self.eps = eps
        self.min_samples = min_samples
        self.model = None
        self.labels_ = None
        self.gps_time_series = None
        self.time_positions_composite_memory = None

    def fit(self, positions: np.ndarray) -> np.ndarray:
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.labels_ = self.model.fit_predict(positions)
        return self.labels_

    def fit_loaded_gps(self) -> np.ndarray:
        self.gps_time_series = self._load_gps()
        self.labels_ = self.fit(self.gps_time_series)
        return self.labels_

    def plot_loaded_gps_clusters(self) -> None:
        if self.gps_time_series is None:
            raise ValueError("GPS data has not been loaded. Call fit_loaded_gps first.")

        self.plot_clusters(self.gps_time_series)

    def plot_clusters(self, positions: np.ndarray) -> None:
        if self.labels_ is None:
            raise ValueError("DBSCAN has not been fitted. Call fit or fit_loaded_gps first.")

        unique_labels = sorted(set(self.labels_))

        plt.figure(figsize=(8, 6))

        for label_index, cluster_label in enumerate(unique_labels):
            class_member_mask = self.labels_ == cluster_label
            cluster_positions = positions[class_member_mask]

            if cluster_label == -1:
                plt.scatter(cluster_positions[:, 0], cluster_positions[:, 1], c="k", marker="x", label="Noise")
            else:
                color = plt.cm.tab10(label_index % 10)
                plt.scatter(cluster_positions[:, 0], cluster_positions[:, 1], c=[color],
                            label=f"Cluster {cluster_label}")

        plt.title("DBSCAN Clustering on GPS Positions")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True)
        plt.axis("equal")
        plt.tight_layout()
        plt.show()

    def _load_gps(self) -> np.ndarray:
        self.time_positions_composite_memory = Uav1300kNormalTimePositionModality(self.data_slice)
        data = self.time_positions_composite_memory.get_np_positions()

        if data.ndim != 2:
            raise ValueError("Position data must have shape (time_steps, features).")

        if data.shape[1] < 3:
            raise ValueError("Position data must contain at least three columns: x, y, z.")

        gps_time_series = data[:, 0:3]
        gps_time_series = self._remove_nan_rows(gps_time_series)
        gps_time_series = self._center_gps_series(gps_time_series)

        return gps_time_series

    def _remove_nan_rows(self, gps_time_series: np.ndarray) -> np.ndarray:
        valid_row_mask = ~np.isnan(gps_time_series).any(axis=1)
        cleaned_gps_time_series = gps_time_series[valid_row_mask]

        if cleaned_gps_time_series.shape[0] == 0:
            raise ValueError("GPS data contains no valid rows after removing NaN values.")

        return cleaned_gps_time_series

    def _center_gps_series(self, gps_time_series: np.ndarray) -> np.ndarray:
        center = np.mean(gps_time_series, axis=0)
        centered_gps_time_series = gps_time_series - center
        return centered_gps_time_series


if __name__ == "__main__":
    data_slice = slice(0, 50000)

    clusterer = DbscanGps(data_slice=data_slice, eps=0.016, min_samples=5)
    clusterer.fit_loaded_gps()
    clusterer.plot_loaded_gps_clusters()