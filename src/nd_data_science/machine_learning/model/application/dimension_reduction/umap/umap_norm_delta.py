from nd_utility.data.storage.kind.file.pkl.pkl import Pkl
from nd_utility.os.file_system.file.file import File
from nd_utility.os.file_system.path.path import Path

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# ----- FIX local 'umap' shadowing -----
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir in sys.path:
    sys.path.remove(script_dir)

import umap


class UmapNormDelta:
    """
    Computes UMAP on the L2 norm of LiDAR scan deltas.
    Intended to separate straight motion (small norm)
    from corner/turn motion (large norm).
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

        print("Original shape:", lidar_vectors.shape)

        # ----- Delta and norm(delta) -----
        delta_vectors = lidar_vectors[1:] - lidar_vectors[:-1]
        norm_delta = np.linalg.norm(delta_vectors, axis=1)
        norm_delta = norm_delta.reshape(-1, 1)

        print("Norm(delta) shape:", norm_delta.shape)

        # ----- Downsample -----
        target_sample_count = 10000
        if norm_delta.shape[0] > target_sample_count:
            step = norm_delta.shape[0] // target_sample_count
            if step < 1:
                step = 1
            norm_delta = norm_delta[::step]

        print("Shape used for UMAP:", norm_delta.shape)

        # ----- UMAP -----
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=30,
            min_dist=0.1,
            metric="euclidean",
            random_state=42
        )

        embedding = reducer.fit_transform(norm_delta)

        print("UMAP embedding shape:", embedding.shape)

        # ----- View -----
        plt.figure()
        plt.scatter(embedding[:, 0], embedding[:, 1], s=2,
                    c=norm_delta, cmap="viridis")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.title("UMAP on L2 norm of LiDAR deltas")
        plt.colorbar(label="||delta||")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    umap_norm = UmapNormDelta()
