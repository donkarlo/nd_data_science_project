from nd_utility.data.storage.kind.file.pkl.pkl import Pkl
from nd_utility.os.file_system.file.file import File
from nd_utility.os.file_system.path.path import Path
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

class Pca:
    """
    Tries to preserve teh variance of the whole robotic_group
    """
    def __init__(self):
        path = Path(
            "/robotic_group/experiements/oldest/robots/uav1/mind/memory/long_term/explicit/episodic/normal/lidar/lidar.pkl")

        os_file = File.init_from_path(path)
        pickle = Pkl(os_file, False)
        scan_sliced_values = pickle.load()
        scan_values = scan_sliced_values.get_values()

        scan_vecs = []
        for scan_value in scan_values:
            scan_vecs.append(
                scan_value
                .get_formatted_data()
                .get_vector_representation()
                .get_components()
            )

        scan_vecs = np.array(scan_vecs, dtype=np.float64)

        # ----- Clean LiDAR robotic_group -----
        max_range = 15.0

        # Replace inf / -inf / NaN with max_range
        scan_vecs[~np.isfinite(scan_vecs)] = max_range

        # Clip all distances to sensor maximum range
        scan_vecs[scan_vecs > max_range] = max_range
        scan_vecs[scan_vecs < 0.0] = 0.0

        # ----- PCA to 3 dimensions -----
        pca = PCA(n_components=3)
        scan_vecs_transformed = pca.fit_transform(scan_vecs)

        print("Original shape:", scan_vecs.shape)
        print("Transformed shape:", scan_vecs_transformed.shape)
        print("Explained variance ratio_value:", pca.explained_variance_ratio_)

        # ----- Simple 3D plot -----
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(
            scan_vecs_transformed[:, 0],
            scan_vecs_transformed[:, 1],
            scan_vecs_transformed[:, 2],
            s=1
        )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        plt.tight_layout()
        plt.show()

    def train(self) -> None:
        pass

    def test(self) -> None:
        pass

    def train(self)->None:
        pass

    def test(self)->None:
        pass

if __name__ == "__main__":
    pca = Pca()