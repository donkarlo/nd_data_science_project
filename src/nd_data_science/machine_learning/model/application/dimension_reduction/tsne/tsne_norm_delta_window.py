from nd_utility.data.storage.kind.file.pkl.pkl import Pkl
from nd_utility.os.file_system.file.file import File
from nd_utility.os.file_system.path.path import Path

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class TsneNormDeltaWindow:
    """
    Computes t-SNE on a 3-sample sliding window of L2 delta norms.
    This separates straight vs corner motion much more clearly.
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

        # ----- Delta -----
        delta_vectors = lidar_vectors[1:] - lidar_vectors[:-1]

        # ----- L2 norm(delta) -----
        norm_delta = np.linalg.norm(delta_vectors, axis=1)

        # ----- Build sliding window features -----
        # Each sample is [norm(t-2), norm(t-1), norm(t)]
        window_features = []
        for t in range(2, len(norm_delta)):
            window_features.append([
                norm_delta[t - 2],
                norm_delta[t - 1],
                norm_delta[t]
            ])
        window_features = np.array(window_features, dtype=np.float64)

        print("Sliding-window feature shape:", window_features.shape)

        # ----- Downsample -----
        target_sample_count = 8000
        if window_features.shape[0] > target_sample_count:
            step = window_features.shape[0] // target_sample_count
            if step < 1:
                step = 1
            window_features = window_features[::step]

        print("Shape used for t-SNE:", window_features.shape)

        # ----- t-SNE -----
        tsne = TSNE(
            n_components=2,
            perplexity=40.0,
            learning_rate=200.0,
            init="random",
            random_state=42,
            verbose=1,
        )
        embedding = tsne.fit_transform(window_features)

        print("t-SNE embedding shape:", embedding.shape)

        # Color by latest norm_delta value in window
        colors = window_features[:, 2]

        # ----- View -----
        plt.figure()
        plt.scatter(embedding[:, 0], embedding[:, 1], s=3,
                    c=colors, cmap="viridis")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.title("t-SNE on sliding-window norm(delta)")
        plt.colorbar(label="||delta|| (current)")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    tsne_norm_win = TsneNormDeltaWindow()
