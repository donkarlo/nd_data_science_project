from nd_utility.data.storage.kind.file.pkl.pkl import Pkl
from nd_utility.os.file_system.file.file import File
from nd_utility.os.file_system.path.path import Path

import sys
import numpy as np
import matplotlib.pyplot as plt

# Move script directory to the end of sys.path
script_dir = sys.path[0]
sys.path = sys.path[1:] + [script_dir]

import umap  # this now imports the external 'umap-learn' package


class Umap:
    """
    Computes a 2D UMAP embedding of LiDAR scans for visualization.
    """

    def __init__(self) -> None:
        path = Path(
            "/robotic_group/experiements/oldest/robots/uav1/mind/memory/long_term/explicit/episodic/normal/lidar/lidar.pkl"
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

        # Clean LiDAR robotic_group
        max_range = 15.0
        lidar_vectors[~np.isfinite(lidar_vectors)] = max_range
        lidar_vectors[lidar_vectors > max_range] = max_range
        lidar_vectors[lidar_vectors < 0.0] = 0.0

        print("Original shape:", lidar_vectors.shape)

        # Downsample for speed
        target_sample_count = 50000
        if lidar_vectors.shape[0] > target_sample_count:
            step = lidar_vectors.shape[0] // target_sample_count
            if step < 1:
                step = 1
            lidar_vectors = lidar_vectors[::step]

        print("Shape used for UMAP:", lidar_vectors.shape)

        # External UMAP (from umap-learn)
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=30,
            min_dist=0.1,
            metric="euclidean",
            random_state=42,
        )

        embedding = reducer.fit_transform(lidar_vectors)

        print("UMAP embedding shape:", embedding.shape)

        plt.figure()
        plt.scatter(embedding[:, 0], embedding[:, 1], s=2)
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.title("UMAP embedding of LiDAR scans")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    umap_embedding = Umap()
