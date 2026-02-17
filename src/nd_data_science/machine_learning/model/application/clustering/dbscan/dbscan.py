import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class DBSCANClusterer:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.model = None
        self.labels_ = None

    def fit(self, X):
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.labels_ = self.model.fit_predict(X)
        return self.labels_

    def plot_clusters(self, X):
        unique_labels = set(self.labels_)
        colors = [plt.cm.tab10(i / float(len(unique_labels))) for i in unique_labels]

        plt.figure(figsize=(8, 6))
        for k, col in zip(unique_labels, colors):
            class_member_mask = (self.labels_ == k)
            xy = X[class_member_mask]
            if k == -1:
                plt.scatter(xy[:, 0], xy[:, 1], c='k', marker='x', label='Noise')
            else:
                plt.scatter(xy[:, 0], xy[:, 1], c=[col], label=f'Cluster {k}')
        plt.title("DBSCAN Clustering (Class-Based)")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

# PublisherExample usage
if __name__ == "__main__":
    # Generate synthetic 2D robotic_group
    X, _ = make_blobs(n_samples=300, centers=[(-5, -5), (0, 0), (5, 5)],
                      cluster_std=0.8, random_state=42)

    # Create and use the clusterer
    clusterer = DBSCANClusterer(eps=1.0, min_samples=5)
    clusterer.fit(X)
    clusterer.plot_clusters(X)