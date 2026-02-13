from nd_utility.data.storage.kind.file.pkl.pkl import Pkl
from nd_utility.os.file_system.file.file import File
from nd_utility.os.file_system.path.path import Path

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class Pca:
    """
    Loads LiDAR scans, computes per-scan cornerness based on PCA of (x, y),
    classifies each scan as 'wall' or 'corner', and visualizes the two classes
    in a 2D PCA space of the original scan vectors.
    """

    def __init__(self) -> None:
        # ----- Load LiDAR scans from pickle -----
        path = Path(
            "/robotic/experiements/oldest/robots/uav1/mind/memory/long_term/explicit/episodic/normal/lidar/lidar.pkl"
        )

        os_file = File.init_from_path(path)
        pickle = Pkl(os_file, False)
        scan_sliced_values = pickle.load()
        scan_values = scan_sliced_values.get_values()

        scan_vecs = []
        for scan_value in scan_values:
            scan_vecs.append(
                scan_value \
                    .get_formatted_data() \
                    .get_vector_representation() \
                    .get_components()
            )

        scan_vecs = np.array(scan_vecs, dtype=np.float64)

        # ----- Clean LiDAR robotic -----
        # RPLidar A2 maximum usable range is about 14 m, we keep 15.0 as a safe clipping bound.
        max_range = 15.0

        # Clip very large ranges to max_range, but DO NOT touch NaN/inf here.
        scan_vecs = np.minimum(scan_vecs, max_range)

        self._scan_vecs = scan_vecs
        self._max_range = max_range

        # Build angle array assuming full 360 degrees coverage
        number_of_rays = self._scan_vecs.shape[1]
        self._angles = np.linspace(-np.pi, np.pi, number_of_rays, endpoint=False)

        # ----- Compute cornerness for each scan -----
        self._kappa_values = self._compute_cornerness_for_all_scans(
            min_points=40
        )

        # Choose a threshold: top 10% as corners (you can tune this)
        corner_threshold = float(np.percentile(self._kappa_values, 90.0))
        self._corner_threshold = corner_threshold

        labels = np.where(self._kappa_values >= corner_threshold, 1, 0)
        self._labels = labels

        print("Numbered of scans:", self._scan_vecs.shape[0])
        print("Corner threshold (kappa):", corner_threshold)
        print("Wall scans:", int(np.sum(labels == 0)))
        print("Corner scans:", int(np.sum(labels == 1)))

        # ----- PCA of scan vectors to 2 dimensions for visualization -----
        pca = PCA(n_components=2)
        scan_vecs_transformed = pca.fit_transform(self._scan_vecs)

        print("Original shape:", self._scan_vecs.shape)
        print("Transformed shape:", scan_vecs_transformed.shape)
        print("Explained variance ratio_value:", pca.explained_variance_ratio_)

        # ----- 2D plot: wall vs corner in PCA space -----
        fig, ax = plt.subplots(figsize=(8, 6))

        wall_mask = labels == 0
        corner_mask = labels == 1

        ax.scatter(
            scan_vecs_transformed[wall_mask, 0],
            scan_vecs_transformed[wall_mask, 1],
            s=3,
            alpha=0.4,
            label="wall"
        )

        ax.scatter(
            scan_vecs_transformed[corner_mask, 0],
            scan_vecs_transformed[corner_mask, 1],
            s=8,
            alpha=0.9,
            label="corner"
        )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("LiDAR scans: walls vs corners in PCA space")
        ax.legend()
        plt.tight_layout()
        plt.show()

        # ----- Optional: kappa over time -----
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.plot(self._kappa_values, linewidth=0.5)
        ax2.axhline(self._corner_threshold, color="red", linestyle="--")
        ax2.set_xlabel("Scan index")
        ax2.set_ylabel("kappa (cornerness)")
        ax2.set_title("Cornerness over time")
        plt.tight_layout()
        plt.show()

    def _compute_cornerness_for_all_scans(self, min_points: int) -> np.ndarray:
        """
        Computes cornerness kappa for all scans.

        :param min_points: minimum value of valid points required to compute covariance
        :return: array of shape (num_scans,) with kappa values
        """
        num_scans = self._scan_vecs.shape[0]
        kappa_values = np.zeros(num_scans, dtype=np.float32)

        for index in range(num_scans):
            ranges = self._scan_vecs[index]
            kappa = self._compute_cornerness_for_single_scan(
                ranges=ranges,
                min_points=min_points
            )
            kappa_values[index] = kappa

        return kappa_values

    def _compute_cornerness_for_single_scan(
            self,
            ranges: np.ndarray,
            min_points: int
    ) -> float:
        """
        Computes cornerness kappa for a single LiDAR scan.

        :param ranges: LiDAR ranges, shape (N,)
        :param min_points: minimum value of valid points
        :return: kappa in [0, 0.5] approximately
        """
        # Valid points: finite, positive, and within max_range
        finite_mask = np.isfinite(ranges)
        positive_mask = ranges > 0.0
        in_range_mask = ranges <= self._max_range

        mask = np.logical_and(finite_mask, np.logical_and(positive_mask, in_range_mask))

        valid_ranges = ranges[mask]
        valid_angles = self._angles[mask]

        if valid_ranges.shape[0] < min_points:
            return 0.0

        x_coordinates = valid_ranges * np.cos(valid_angles)
        y_coordinates = valid_ranges * np.sin(valid_angles)

        points = np.stack((x_coordinates, y_coordinates), axis=1)

        # Center points
        mean = np.mean(points, axis=0, keepdims=True)
        centered = points - mean

        covariance = np.cov(centered, rowvar=False)

        try:
            eigenvalues = np.linalg.eigvalsh(covariance)
        except np.linalg.LinAlgError:
            return 0.0

        eigenvalues_sorted = np.sort(eigenvalues)
        lambda_small = float(eigenvalues_sorted[0])
        lambda_large = float(eigenvalues_sorted[1])

        denominator = lambda_small + lambda_large
        if denominator <= 0.0:
            return 0.0

        kappa = lambda_small / denominator
        return kappa

    def train(self) -> None:
        """
        Placeholder to keep the interface consistent.
        """
        pass

    def test(self) -> None:
        """
        Placeholder to keep the interface consistent.
        """
        pass


if __name__ == "__main__":
    pca = Pca()
