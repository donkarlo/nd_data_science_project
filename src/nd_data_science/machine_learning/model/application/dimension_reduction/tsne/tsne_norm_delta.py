from nd_utility.data.storage.kind.file.pkl.pkl import Pkl
from nd_utility.os.file_system.file.file import File
from nd_utility.os.file_system.path.path import Path

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class TsneNormDelta:
    """
    Computes t-SNE on the L2 norm of LiDAR scan deltas.
    This produces a very clear separation between straight motion
    (small norm) and corner/turn motion (large norm).
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

        # ----- Compute delta vectors -----
        delta_vectors = lidar_vectors[1:] - lidar_vectors[:-1]

        # ----- L2 norm of each delta -----
        # Each delta becomes a single scalar
        norm_delta = np.linalg.norm(delta_vectors, axis=1)
        norm_delta = norm_delta.reshape(-1, 1)

        print("Norm(delta) shape:", norm_delta.shape)

        # ----- Downsample if necessary -----
        target_sample_count = 10000
        if norm_delta.shape[0] > target_sample_count:
            step = norm_delta.shape[0] // target_sample_count
            if step < 1:
                step = 1
            norm_delta = norm_delta[::step]

        print("Shape used for t-SNE:", norm_delta.shape)

        # ----- t-SNE -----
        tsne = TSNE(
            n_components=2,
            perplexity=30.0,
            learning_rate=200.0,
            init="random",
            random_state=42,
            verbose=1,
        )

        embedding = tsne.fit_transform(norm_delta)

        print("t-SNE embedding shape:", embedding.shape)

        # ----- View -----
        plt.figure()
        plt.scatter(embedding[:, 0], embedding[:, 1], s=2, c=norm_delta, cmap="viridis")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.title("t-SNE on L2 norm of LiDAR deltas")
        plt.colorbar(label="||delta||")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    tsne_norm = TsneNormDelta()
