from nd_utility.data.storage.kind.file.pkl.pkl import Pkl
from nd_utility.os.file_system.file.file import File
from nd_utility.os.file_system.path.path import Path

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class TsneEmbedding:
    """
    Computes a 2D t-SNE embedding of LiDAR scans for visualization.
    """

    def __init__(self) -> None:
        path = Path(
            "/robotic/experiements/oldest/robots/uav1/mind/memory/long_term/explicit/episodic/normal/lidar/lidar.pkl"
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

        # ----- Clean LiDAR robotic -----
        max_range = 15.0

        # Replace inf / -inf / NaN with max_range
        lidar_vectors[~np.isfinite(lidar_vectors)] = max_range

        # Clip all distances to sensor bounds
        lidar_vectors[lidar_vectors > max_range] = max_range
        lidar_vectors[lidar_vectors < 0.0] = 0.0

        print("Original shape:", lidar_vectors.shape)

        # ----- Downsample for t-SNE (it does not scale to 300k samples) -----
        target_sample_count = 10000
        if lidar_vectors.shape[0] > target_sample_count:
            step = lidar_vectors.shape[0] // target_sample_count
            if step < 1:
                step = 1
            lidar_vectors = lidar_vectors[::step]

        print("Shape used for t-SNE:", lidar_vectors.shape)

        # ----- t-SNE to 2 dimensions -----
        tsne = TSNE(
            n_components=2,
            perplexity=30.0,
            learning_rate=200.0,
            init="random",
            random_state=42,
            verbose=1
        )

        embedding = tsne.fit_transform(lidar_vectors)

        print("t-SNE embedding shape:", embedding.shape)

        # ----- Simple 2D scatter plot -----
        plt.figure()
        plt.scatter(embedding[:, 0], embedding[:, 1], s=2)
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.title("t-SNE embedding of LiDAR scans")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    tsne_embedding = TsneEmbedding()
