from nd_utility.data.storage.kind.file.pkl.pkl import Pkl
from nd_utility.os.file_system.file.file import File
from nd_utility.os.file_system.path.path import Path

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# ----- REMOVE LOCAL UMAP SHADOWING -----
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir in sys.path:
    sys.path.remove(script_dir)

import umap


class UmapNormDeltaWindow:
    """
    Apply UMAP on sliding-window features of L2 norm of LiDAR deltas.
    """

    def __init__(self) -> None:
        path = Path(
            "/robotic_group/experiements/oldest/robots/uav1/mind/memory/long_term/explicit/episodic/normal/lidar/lidar.pkl"
        )

        os_file = File.init_from_path(path)
        pickle = Pkl(os_file, False)
        scan_sliced_values = pickle.load()
        scan_values = scan_sliced_values.get_values()

        # ----- Load LiDAR vectors -----
        lidar_vectors = []
        for scan_value in scan_values:
            lidar_vectors.append(
                scan_value
                .get_formatted_data()
                .get_vector_representation()
                .get_components()
            )

        lidar_vectors = np.array(lidar_vectors, dtype=np.float64)

        # ----- Clean robotic_group -----
        max_range = 15.0
        lidar_vectors[~np.isfinite(lidar_vectors)] = max_range
        lidar_vectors[lidar_vectors > max_range] = max_range
        lidar_vectors[lidar_vectors < 0.0] = 0.0

        print("Original lidar shape:", lidar_vectors.shape)

        # ----- delta + norm -----
        delta_vectors = lidar_vectors[1:] - lidar_vectors[:-1]
        norm_delta = np.linalg.norm(delta_vectors, axis=1)
        print("norm(delta) length:", norm_delta.shape[0])

        # ----- Sliding window -----
        window_size = 10
        num_windows = len(norm_delta) - window_size + 1
        window_features = np.zeros((num_windows, window_size))

        for i in range(num_windows):
            window_features[i] = norm_delta[i:i + window_size]

        current_norm = norm_delta[window_size - 1:]
        print("Window feature shape:", window_features.shape)
        print("Current norm shape:", current_norm.shape)

        # ----- Downsample -----
        target_sample_count = 12500
        if window_features.shape[0] > target_sample_count:
            step = window_features.shape[0] // target_sample_count
            window_features = window_features[::step]
            current_norm = current_norm[::step]

        print("Shape used for UMAP:", window_features.shape)

        # ----- UMAP -----
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=30,
            min_dist=0.1,
            metric="euclidean",
            random_state=42
        )

        embedding = reducer.fit_transform(window_features)
        print("UMAP embedding shape:", embedding.shape)

        #----- View -----
        plt.figure()
        scatter = plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            s=2,
            c=current_norm,
            cmap="viridis"
        )
        plt.colorbar(scatter, label="||delta|| (current)")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.title("UMAP on sliding-window norm(delta)")
        plt.tight_layout()
        plt.show()

        # ----- 3D plot -----
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection="3d")
        #
        # scatter = ax.scatter(
        #     embedding[:, 0],
        #     embedding[:, 1],
        #     embedding[:, 2],
        #     s=2,
        #     c=current_norm,
        #     cmap="viridis",
        # )
        #
        # ax.set_xlabel("UMAP 1")
        # ax.set_ylabel("UMAP 2")
        # ax.set_zlabel("UMAP 3")
        # fig.colorbar(scatter, label="||delta|| (current)")
        # plt.title("3D UMAP on sliding-window norm(delta)")
        # plt.tight_layout()
        # plt.show()


if __name__ == "__main__":
    UmapNormDeltaWindow()
