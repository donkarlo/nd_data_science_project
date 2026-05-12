from nd_sociomind.experiment.parts.oldest.uav1_normal_lidar_time_ranges_modality import \
    Uav1NormalLidarTimeRangesModality

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as pyplot
import numpy


class LidarPolygonAutoDiscoveryPenalizedClusterer:
    def __init__(self, data_slice, lidar_dimension=720, start_angle=-numpy.pi, end_angle=numpy.pi,
                 far_range_threshold=2.5, replacement_distance=4.0, point_stride=4, descriptor_count=25,
                 minimum_points_for_geometry=20, minimum_corner_fraction=0.03, maximum_corner_fraction=0.70,
                 fallback_corner_quantile=0.80, minimum_corner_cluster_count=2, maximum_corner_cluster_count=8,
                 maximum_silhouette_sample_count=10000, cluster_count_penalty=0.03, random_state=42):
        self.data_slice = data_slice
        self.lidar_dimension = lidar_dimension
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.far_range_threshold = far_range_threshold
        self.replacement_distance = replacement_distance
        self.point_stride = point_stride
        self.descriptor_count = descriptor_count
        self.minimum_points_for_geometry = minimum_points_for_geometry
        self.minimum_corner_fraction = minimum_corner_fraction
        self.maximum_corner_fraction = maximum_corner_fraction
        self.fallback_corner_quantile = fallback_corner_quantile
        self.minimum_corner_cluster_count = minimum_corner_cluster_count
        self.maximum_corner_cluster_count = maximum_corner_cluster_count
        self.maximum_silhouette_sample_count = maximum_silhouette_sample_count
        self.cluster_count_penalty = cluster_count_penalty
        self.random_state = random_state

        self.time_ranges_modality = None
        self.time_ranges = None
        self.raw_lidar_ranges_time_series = None
        self.cleaned_lidar_ranges_time_series = None

        self.angle_values = None
        self.selected_indices = None
        self.x_time_series = None
        self.y_time_series = None
        self.polygon_time_series = None

        self.geometry_feature_time_series = None
        self.scaled_geometry_feature_time_series = None
        self.geometry_feature_scaler = None

        self.polygon_descriptor_time_series = None
        self.scaled_polygon_descriptor_time_series = None
        self.polygon_descriptor_scaler = None

        self.corner_score_values = None
        self.valid_ratio_values = None
        self.far_ratio_values = None

        self.wall_corner_model = None
        self.corner_mask = None
        self.straight_mask = None
        self.corner_selection_method = None

        self.corner_cluster_models = {}
        self.corner_cluster_silhouette_scores = {}
        self.corner_cluster_penalized_scores = {}
        self.selected_corner_cluster_count = None
        self.total_cluster_count = None
        self.corner_kmeans_model = None

        self.labels_ = None
        self.cluster_ids = None
        self.cluster_sizes = None
        self.representative_indices = None
        self.representative_polygons = None
        self.representative_ranges = None

    def run(self) -> numpy.ndarray:
        self.load_and_prepare()
        self.discover_clusters()
        self.extract_representative_polygons()
        self.print_summary()
        return self.labels_

    def load_and_prepare(self) -> None:
        self.time_ranges = self._load_lidar_time_ranges()
        self.raw_lidar_ranges_time_series = self.time_ranges[:, 1:]
        self.cleaned_lidar_ranges_time_series = self._clean_ranges(self.raw_lidar_ranges_time_series)

        self._prepare_angles()

        self.x_time_series, self.y_time_series = self._convert_ranges_to_cartesian(
            self.cleaned_lidar_ranges_time_series)
        self.polygon_time_series = numpy.stack((self.x_time_series, self.y_time_series), axis=2)

        self.geometry_feature_time_series = self._create_geometry_feature_time_series()
        self.scaled_geometry_feature_time_series = self._scale_geometry_features(self.geometry_feature_time_series)

        self.polygon_descriptor_time_series = self._create_fourier_polygon_descriptors(self.x_time_series,
                                                                                       self.y_time_series)
        self.scaled_polygon_descriptor_time_series = self._scale_polygon_descriptors(
            self.polygon_descriptor_time_series)

    def discover_clusters(self) -> numpy.ndarray:
        if self.scaled_geometry_feature_time_series is None:
            self.load_and_prepare()

        self.corner_mask = self._discover_corner_candidate_mask()
        self.straight_mask = numpy.logical_not(self.corner_mask)

        corner_indices = numpy.where(self.corner_mask)[0]

        if corner_indices.shape[0] < self.minimum_corner_cluster_count:
            raise ValueError("Not enough corner candidate scans were found.")

        corner_descriptors = self.scaled_polygon_descriptor_time_series[corner_indices, :]
        self.selected_corner_cluster_count = self._select_corner_cluster_count(corner_descriptors)

        self.corner_kmeans_model = KMeans(
            n_clusters=self.selected_corner_cluster_count,
            n_init=30,
            random_state=self.random_state
        )

        corner_cluster_labels = self.corner_kmeans_model.fit_predict(corner_descriptors)

        self.labels_ = numpy.zeros(self.raw_lidar_ranges_time_series.shape[0], dtype=int)
        self.labels_[corner_indices] = corner_cluster_labels + 1

        self.total_cluster_count = int(self.selected_corner_cluster_count + 1)
        self.cluster_ids, self.cluster_sizes = self._get_cluster_ids_and_sizes(self.labels_)

        return self.labels_

    def extract_representative_polygons(self) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        if self.labels_ is None:
            self.discover_clusters()

        representative_indices = []

        for cluster_id in self.cluster_ids:
            cluster_mask = self.labels_ == cluster_id
            cluster_indices = numpy.where(cluster_mask)[0]
            cluster_descriptors = self.scaled_polygon_descriptor_time_series[cluster_mask, :]

            cluster_center = numpy.mean(cluster_descriptors, axis=0)
            distance_values = numpy.linalg.norm(cluster_descriptors - cluster_center[None, :], axis=1)

            local_representative_index = int(numpy.argmin(distance_values))
            representative_index = int(cluster_indices[local_representative_index])
            representative_indices.append(representative_index)

        self.representative_indices = numpy.asarray(representative_indices, dtype=int)
        self.representative_polygons = self.polygon_time_series[self.representative_indices, :, :]
        self.representative_ranges = self.cleaned_lidar_ranges_time_series[self.representative_indices, :]

        return self.representative_indices, self.representative_polygons, self.representative_ranges

    def print_summary(self) -> None:
        if self.labels_ is None:
            raise ValueError("Labels have not been created.")

        corner_count = int(numpy.sum(self.corner_mask))
        straight_count = int(numpy.sum(self.straight_mask))

        print("LiDAR polygon automatic cluster discovery with penalized silhouette")
        print(f"sample_count = {self.labels_.shape[0]}")
        print(f"corner_selection_method = {self.corner_selection_method}")
        print(f"straight_or_wall_candidate_count = {straight_count}")
        print(f"corner_candidate_count = {corner_count}")
        print(f"selected_corner_cluster_count = {self.selected_corner_cluster_count}")
        print(f"total_cluster_count = {self.total_cluster_count}")
        print(f"cluster_count_penalty = {self.cluster_count_penalty}")
        print(f"far_range_threshold = {self.far_range_threshold}")
        print(f"replacement_distance = {self.replacement_distance}")
        print(f"descriptor_dimension = {self.scaled_polygon_descriptor_time_series.shape[1]}")
        print()

        print("Corner-cluster scores:")
        for cluster_count in sorted(self.corner_cluster_silhouette_scores.keys()):
            silhouette_value = self.corner_cluster_silhouette_scores[cluster_count]
            penalized_value = self.corner_cluster_penalized_scores[cluster_count]
            print(f"k = {cluster_count}, silhouette = {silhouette_value}, penalized = {penalized_value}")
        print()

        for cluster_id, cluster_size, representative_index in zip(self.cluster_ids, self.cluster_sizes,
                                                                  self.representative_indices):
            if int(cluster_id) == 0:
                cluster_name = "straight_or_wall_side_class"
            else:
                cluster_name = f"discovered_corner_class_{int(cluster_id)}"

            print(f"cluster_id = {int(cluster_id)}")
            print(f"cluster_name = {cluster_name}")
            print(f"cluster_size = {int(cluster_size)}")
            print(f"representative_scan_index = {int(representative_index)}")
            print("representative_polygon_shape = (720, 2)")
            print("representative_ranges_shape = (720,)")
            print()

    def plot_representative_polygons(self) -> None:
        if self.representative_polygons is None:
            self.extract_representative_polygons()

        import matplotlib.pyplot as pyplot

        cluster_count = len(self.cluster_ids)
        column_count = 3
        row_count = int(numpy.ceil(float(cluster_count) / float(column_count)))

        pyplot.figure(figsize=(15, 5 * row_count))

        for plot_index, cluster_id in enumerate(self.cluster_ids):
            polygon = self.representative_polygons[plot_index]
            closed_polygon = numpy.vstack((polygon, polygon[0:1, :]))

            axes = pyplot.subplot(row_count, column_count, plot_index + 1)
            axes.plot(closed_polygon[:, 0], closed_polygon[:, 1], linewidth=1.2)
            axes.scatter(polygon[:, 0], polygon[:, 1], s=3)

            representative_index = int(self.representative_indices[plot_index])
            cluster_size = int(self.cluster_sizes[plot_index])

            if int(cluster_id) == 0:
                title = f"Cluster {int(cluster_id)}: wall/straight\nscan={representative_index}, size={cluster_size}"
            else:
                title = f"Cluster {int(cluster_id)}: corner\nscan={representative_index}, size={cluster_size}"

            axes.set_title(title)
            axes.set_aspect("equal")
            axes.grid(True)

        pyplot.tight_layout()
        pyplot.show()

    def _load_lidar_time_ranges(self) -> numpy.ndarray:
        self.time_ranges_modality = Uav1NormalLidarTimeRangesModality(self.data_slice)
        data = self.time_ranges_modality.get_np_time_ranges()

        if data.ndim != 2:
            raise ValueError("LiDAR data must have shape (time_steps, features).")

        expected_column_count = self.lidar_dimension + 1

        if data.shape[1] != expected_column_count:
            raise ValueError(f"LiDAR data must have {expected_column_count} columns.")

        return data

    def _clean_ranges(self, lidar_ranges_time_series: numpy.ndarray) -> numpy.ndarray:
        cleaned_ranges = lidar_ranges_time_series.copy()

        invalid_mask = ~numpy.isfinite(cleaned_ranges)
        far_mask = cleaned_ranges > self.far_range_threshold
        replacement_mask = numpy.logical_or(invalid_mask, far_mask)

        cleaned_ranges[replacement_mask] = self.replacement_distance

        negative_mask = cleaned_ranges < 0.0
        cleaned_ranges[negative_mask] = 0.0

        return cleaned_ranges

    def _prepare_angles(self) -> None:
        angle_step = (self.end_angle - self.start_angle) / float(self.lidar_dimension)
        self.angle_values = self.start_angle + numpy.arange(self.lidar_dimension, dtype=float) * angle_step
        self.selected_indices = numpy.arange(0, self.lidar_dimension, self.point_stride, dtype=int)

    def _convert_ranges_to_cartesian(self, lidar_ranges_time_series: numpy.ndarray) -> tuple[
        numpy.ndarray, numpy.ndarray]:
        cosine_values = numpy.cos(self.angle_values)
        sine_values = numpy.sin(self.angle_values)

        x_time_series = lidar_ranges_time_series * cosine_values[None, :]
        y_time_series = lidar_ranges_time_series * sine_values[None, :]

        return x_time_series, y_time_series

    def _create_geometry_feature_time_series(self) -> numpy.ndarray:
        scan_count = self.raw_lidar_ranges_time_series.shape[0]

        corner_score_values = numpy.zeros(scan_count, dtype=float)
        valid_ratio_values = numpy.zeros(scan_count, dtype=float)
        far_ratio_values = numpy.zeros(scan_count, dtype=float)
        elongation_values = numpy.zeros(scan_count, dtype=float)
        radial_standard_deviation_values = numpy.zeros(scan_count, dtype=float)

        for scan_index in range(scan_count):
            scan_ranges = self.raw_lidar_ranges_time_series[scan_index]
            points, valid_ratio, far_ratio, valid_ranges = self._convert_scan_to_valid_points(scan_ranges)

            valid_ratio_values[scan_index] = valid_ratio
            far_ratio_values[scan_index] = far_ratio

            if points.shape[0] < self.minimum_points_for_geometry:
                corner_score_values[scan_index] = 0.0
                elongation_values[scan_index] = 0.0
                radial_standard_deviation_values[scan_index] = 0.0
                continue

            nonlinearity, elongation = self._calculate_nonlinearity_and_elongation(points)

            corner_score_values[scan_index] = nonlinearity
            elongation_values[scan_index] = elongation
            radial_standard_deviation_values[scan_index] = float(numpy.std(valid_ranges))

        self.corner_score_values = corner_score_values
        self.valid_ratio_values = valid_ratio_values
        self.far_ratio_values = far_ratio_values

        feature_time_series = numpy.column_stack((
            corner_score_values,
            valid_ratio_values,
            far_ratio_values,
            elongation_values,
            radial_standard_deviation_values
        ))

        feature_time_series = self._replace_invalid_values(feature_time_series)

        return feature_time_series

    def _convert_scan_to_valid_points(self, scan_ranges: numpy.ndarray) -> tuple[
        numpy.ndarray, float, float, numpy.ndarray]:
        selected_ranges = scan_ranges[self.selected_indices]
        selected_angles = self.angle_values[self.selected_indices]

        finite_mask = numpy.isfinite(selected_ranges)
        positive_mask = selected_ranges >= 0.05
        close_mask = selected_ranges <= self.far_range_threshold

        valid_mask = numpy.logical_and(finite_mask, positive_mask)
        valid_mask = numpy.logical_and(valid_mask, close_mask)

        invalid_or_far_mask = numpy.logical_or(~finite_mask, selected_ranges > self.far_range_threshold)

        valid_ratio = float(numpy.sum(valid_mask)) / float(selected_ranges.shape[0])
        far_ratio = float(numpy.sum(invalid_or_far_mask)) / float(selected_ranges.shape[0])

        valid_ranges = selected_ranges[valid_mask]
        valid_angles = selected_angles[valid_mask]

        x_values = valid_ranges * numpy.cos(valid_angles)
        y_values = valid_ranges * numpy.sin(valid_angles)

        points = numpy.column_stack((x_values, y_values))

        return points, valid_ratio, far_ratio, valid_ranges

    def _calculate_nonlinearity_and_elongation(self, points: numpy.ndarray) -> tuple[float, float]:
        centered_points = points - numpy.mean(points, axis=0)
        covariance_matrix = numpy.cov(centered_points, rowvar=False)

        eigen_values = numpy.linalg.eigvalsh(covariance_matrix)
        eigen_values = numpy.sort(eigen_values)[::-1]

        large_value = max(float(eigen_values[0]), 1e-12)
        small_value = max(float(eigen_values[1]), 1e-12)

        nonlinearity = small_value / (large_value + small_value)
        elongation = large_value / small_value

        return nonlinearity, elongation

    def _discover_corner_candidate_mask(self) -> numpy.ndarray:
        self.wall_corner_model = GaussianMixture(
            n_components=2,
            covariance_type="full",
            n_init=10,
            random_state=self.random_state
        )

        geometry_labels = self.wall_corner_model.fit_predict(self.scaled_geometry_feature_time_series)

        component_corner_scores = []

        for component_index in range(2):
            component_mask = geometry_labels == component_index
            component_score = float(numpy.mean(self.corner_score_values[component_mask]))
            component_corner_scores.append(component_score)

        corner_component = int(numpy.argmax(numpy.asarray(component_corner_scores)))
        corner_mask = geometry_labels == corner_component

        corner_fraction = float(numpy.mean(corner_mask))

        if corner_fraction < self.minimum_corner_fraction or corner_fraction > self.maximum_corner_fraction:
            threshold = float(numpy.quantile(self.corner_score_values, self.fallback_corner_quantile))
            corner_mask = self.corner_score_values >= threshold
            self.corner_selection_method = f"fallback_quantile_{self.fallback_corner_quantile}"
        else:
            self.corner_selection_method = "gmm_2_components_on_geometry_features"

        return corner_mask

    def _select_corner_cluster_count(self, corner_descriptors: numpy.ndarray) -> int:
        if corner_descriptors.shape[0] < self.minimum_corner_cluster_count:
            raise ValueError("Too few corner descriptors for clustering.")

        sampled_descriptors = self._sample_for_silhouette(corner_descriptors)

        best_cluster_count = None
        best_score = -numpy.inf

        maximum_possible_cluster_count = min(self.maximum_corner_cluster_count, sampled_descriptors.shape[0] - 1)

        for cluster_count in range(self.minimum_corner_cluster_count, maximum_possible_cluster_count + 1):
            model = KMeans(
                n_clusters=cluster_count,
                n_init=20,
                random_state=self.random_state
            )

            sampled_labels = model.fit_predict(sampled_descriptors)

            unique_label_count = len(numpy.unique(sampled_labels))

            if unique_label_count < 2:
                continue

            raw_score = float(silhouette_score(sampled_descriptors, sampled_labels))
            penalized_score = raw_score - self.cluster_count_penalty * float(cluster_count)

            self.corner_cluster_silhouette_scores[cluster_count] = raw_score
            self.corner_cluster_penalized_scores[cluster_count] = penalized_score
            self.corner_cluster_models[cluster_count] = model

            if penalized_score > best_score:
                best_score = penalized_score
                best_cluster_count = cluster_count

        if best_cluster_count is None:
            raise ValueError("Could not select a corner cluster count.")

        return int(best_cluster_count)

    def _sample_for_silhouette(self, descriptors: numpy.ndarray) -> numpy.ndarray:
        row_count = descriptors.shape[0]

        if row_count <= self.maximum_silhouette_sample_count:
            return descriptors

        random_generator = numpy.random.default_rng(self.random_state)
        sampled_indices = random_generator.choice(row_count, size=self.maximum_silhouette_sample_count, replace=False)
        sampled_descriptors = descriptors[sampled_indices, :]

        return sampled_descriptors

    def _create_fourier_polygon_descriptors(self, x_time_series: numpy.ndarray,
                                            y_time_series: numpy.ndarray) -> numpy.ndarray:
        complex_polygon_time_series = x_time_series + 1j * y_time_series
        complex_polygon_time_series = complex_polygon_time_series - numpy.mean(
            complex_polygon_time_series,
            axis=1,
            keepdims=True
        )

        fourier_time_series = numpy.fft.fft(complex_polygon_time_series, axis=1)
        selected_fourier_time_series = fourier_time_series[:, 1:self.descriptor_count + 1]

        real_parts = numpy.real(selected_fourier_time_series)
        imaginary_parts = numpy.imag(selected_fourier_time_series)

        descriptor_time_series = numpy.empty(
            (selected_fourier_time_series.shape[0], selected_fourier_time_series.shape[1] * 2),
            dtype=float
        )

        descriptor_time_series[:, 0::2] = real_parts
        descriptor_time_series[:, 1::2] = imaginary_parts
        descriptor_time_series = self._replace_invalid_values(descriptor_time_series)

        return descriptor_time_series

    def _scale_polygon_descriptors(self, descriptor_time_series: numpy.ndarray) -> numpy.ndarray:
        self.polygon_descriptor_scaler = StandardScaler()
        scaled_descriptor_time_series = self.polygon_descriptor_scaler.fit_transform(descriptor_time_series)

        return scaled_descriptor_time_series

    def _scale_geometry_features(self, feature_time_series: numpy.ndarray) -> numpy.ndarray:
        self.geometry_feature_scaler = StandardScaler()
        scaled_feature_time_series = self.geometry_feature_scaler.fit_transform(feature_time_series)

        return scaled_feature_time_series

    def _replace_invalid_values(self, values: numpy.ndarray) -> numpy.ndarray:
        cleaned_values = values.copy()

        for column_index in range(cleaned_values.shape[1]):
            column_values = cleaned_values[:, column_index]
            finite_mask = numpy.isfinite(column_values)

            if numpy.any(finite_mask):
                replacement_value = float(numpy.median(column_values[finite_mask]))
            else:
                replacement_value = 0.0

            column_values[~finite_mask] = replacement_value
            cleaned_values[:, column_index] = column_values

        return cleaned_values

    def _get_cluster_ids_and_sizes(self, labels: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
        unique_labels, label_counts = numpy.unique(labels, return_counts=True)

        cluster_ids = numpy.asarray(unique_labels, dtype=int)
        cluster_sizes = numpy.asarray(label_counts, dtype=int)

        size_order = numpy.argsort(cluster_sizes)[::-1]
        cluster_ids = cluster_ids[size_order]
        cluster_sizes = cluster_sizes[size_order]

        return cluster_ids, cluster_sizes





class ClusterPlotter:
    def __init__(self, data_slice, lidar_dimension=720, start_angle=-numpy.pi, end_angle=numpy.pi,
                 far_range_threshold=2.5, replacement_distance=4.0, point_stride=4, descriptor_count=25,
                 minimum_points_for_geometry=20, minimum_corner_fraction=0.03, maximum_corner_fraction=0.70,
                 fallback_corner_quantile=0.80, minimum_corner_cluster_count=2, maximum_corner_cluster_count=8,
                 maximum_silhouette_sample_count=10000, cluster_count_penalty=0.025, random_state=42):
        self.data_slice = data_slice
        self.lidar_dimension = lidar_dimension
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.far_range_threshold = far_range_threshold
        self.replacement_distance = replacement_distance
        self.point_stride = point_stride
        self.descriptor_count = descriptor_count
        self.minimum_points_for_geometry = minimum_points_for_geometry
        self.minimum_corner_fraction = minimum_corner_fraction
        self.maximum_corner_fraction = maximum_corner_fraction
        self.fallback_corner_quantile = fallback_corner_quantile
        self.minimum_corner_cluster_count = minimum_corner_cluster_count
        self.maximum_corner_cluster_count = maximum_corner_cluster_count
        self.maximum_silhouette_sample_count = maximum_silhouette_sample_count
        self.cluster_count_penalty = cluster_count_penalty
        self.random_state = random_state
        self.clusterer = None

    def run(self) -> None:
        self.clusterer = LidarPolygonAutoDiscoveryPenalizedClusterer(
            data_slice=self.data_slice,
            lidar_dimension=self.lidar_dimension,
            start_angle=self.start_angle,
            end_angle=self.end_angle,
            far_range_threshold=self.far_range_threshold,
            replacement_distance=self.replacement_distance,
            point_stride=self.point_stride,
            descriptor_count=self.descriptor_count,
            minimum_points_for_geometry=self.minimum_points_for_geometry,
            minimum_corner_fraction=self.minimum_corner_fraction,
            maximum_corner_fraction=self.maximum_corner_fraction,
            fallback_corner_quantile=self.fallback_corner_quantile,
            minimum_corner_cluster_count=self.minimum_corner_cluster_count,
            maximum_corner_cluster_count=self.maximum_corner_cluster_count,
            maximum_silhouette_sample_count=self.maximum_silhouette_sample_count,
            cluster_count_penalty=self.cluster_count_penalty,
            random_state=self.random_state
        )

        self.clusterer.run()
        self.clusterer.extract_representative_polygons()

    def plot_one_representative_polygon_from_each_cluster(self) -> None:
        if self.clusterer is None:
            raise ValueError("Run the clusterer first.")

        cluster_count = len(self.clusterer.cluster_ids)
        column_count = 3
        row_count = int(numpy.ceil(float(cluster_count) / float(column_count)))

        pyplot.figure(figsize=(15, 5 * row_count))

        for plot_index, cluster_id in enumerate(self.clusterer.cluster_ids):
            polygon = self.clusterer.representative_polygons[plot_index]
            closed_polygon = numpy.vstack((polygon, polygon[0:1, :]))

            axes = pyplot.subplot(row_count, column_count, plot_index + 1)
            axes.plot(closed_polygon[:, 0], closed_polygon[:, 1], linewidth=1.2)
            axes.scatter(polygon[:, 0], polygon[:, 1], s=3)

            representative_index = int(self.clusterer.representative_indices[plot_index])
            cluster_size = int(self.clusterer.cluster_sizes[plot_index])

            if int(cluster_id) == 0:
                title = f"Cluster {int(cluster_id)}: wall/straight\nscan={representative_index}, size={cluster_size}"
            else:
                title = f"Cluster {int(cluster_id)}: corner\nscan={representative_index}, size={cluster_size}"

            axes.set_title(title)
            axes.set_aspect("equal")
            axes.grid(True)

        pyplot.tight_layout()
        pyplot.show()

    def plot_all_polygons_for_each_cluster(self, maximum_polygons_per_cluster: int | None = None) -> None:
        if self.clusterer is None:
            raise ValueError("Run the clusterer first.")

        random_generator = numpy.random.default_rng(self.random_state)

        for cluster_id in self.clusterer.cluster_ids:
            cluster_mask = self.clusterer.labels_ == cluster_id
            cluster_indices = numpy.where(cluster_mask)[0]

            if maximum_polygons_per_cluster is not None and cluster_indices.shape[0] > maximum_polygons_per_cluster:
                cluster_indices = random_generator.choice(
                    cluster_indices,
                    size=maximum_polygons_per_cluster,
                    replace=False
                )

            pyplot.figure(figsize=(8, 8))

            for scan_index in cluster_indices:
                polygon = self.clusterer.polygon_time_series[scan_index]
                closed_polygon = numpy.vstack((polygon, polygon[0:1, :]))
                pyplot.plot(closed_polygon[:, 0], closed_polygon[:, 1], linewidth=0.4, alpha=0.08)

            representative_position_array = numpy.where(self.clusterer.cluster_ids == cluster_id)[0]
            representative_position = int(representative_position_array[0])

            representative_polygon = self.clusterer.representative_polygons[representative_position]
            representative_closed_polygon = numpy.vstack((representative_polygon, representative_polygon[0:1, :]))

            pyplot.plot(
                representative_closed_polygon[:, 0],
                representative_closed_polygon[:, 1],
                linewidth=2.0
            )

            cluster_size = int(self.clusterer.cluster_sizes[representative_position])

            if int(cluster_id) == 0:
                title = f"Cluster {int(cluster_id)}: wall/straight, plotted={cluster_indices.shape[0]}, size={cluster_size}"
            else:
                title = f"Cluster {int(cluster_id)}: corner, plotted={cluster_indices.shape[0]}, size={cluster_size}"

            pyplot.title(title)
            pyplot.gca().set_aspect("equal")
            pyplot.grid(True)
            pyplot.show()

    def plot_multiple_examples_from_each_cluster(self, examples_per_cluster: int = 5) -> None:
        if self.clusterer is None:
            raise ValueError("Run the clusterer first.")

        random_generator = numpy.random.default_rng(self.random_state)

        cluster_count = len(self.clusterer.cluster_ids)
        column_count = 3
        row_count = int(numpy.ceil(float(cluster_count) / float(column_count)))

        pyplot.figure(figsize=(15, 5 * row_count))

        for plot_index, cluster_id in enumerate(self.clusterer.cluster_ids):
            cluster_mask = self.clusterer.labels_ == cluster_id
            cluster_indices = numpy.where(cluster_mask)[0]

            if cluster_indices.shape[0] > examples_per_cluster:
                selected_indices = random_generator.choice(
                    cluster_indices,
                    size=examples_per_cluster,
                    replace=False
                )
            else:
                selected_indices = cluster_indices

            axes = pyplot.subplot(row_count, column_count, plot_index + 1)

            for selected_scan_index in selected_indices:
                polygon = self.clusterer.polygon_time_series[selected_scan_index]
                closed_polygon = numpy.vstack((polygon, polygon[0:1, :]))
                axes.plot(closed_polygon[:, 0], closed_polygon[:, 1], linewidth=1.0, alpha=0.9)

            cluster_size_position_array = numpy.where(self.clusterer.cluster_ids == cluster_id)[0]
            cluster_size_position = int(cluster_size_position_array[0])
            cluster_size = int(self.clusterer.cluster_sizes[cluster_size_position])

            if int(cluster_id) == 0:
                title = f"Cluster {int(cluster_id)}: wall/straight\nshown={selected_indices.shape[0]}, size={cluster_size}"
            else:
                title = f"Cluster {int(cluster_id)}: corner\nshown={selected_indices.shape[0]}, size={cluster_size}"

            axes.set_title(title)
            axes.set_aspect("equal")
            axes.grid(True)

        pyplot.tight_layout()
        pyplot.show()

    def plot_mean_polygon_from_each_cluster_with_centroid(self) -> None:
        if self.clusterer is None:
            raise ValueError("Run the clusterer first.")

        cluster_count = len(self.clusterer.cluster_ids)
        column_count = 3
        row_count = int(numpy.ceil(float(cluster_count) / float(column_count)))

        pyplot.figure(figsize=(15, 5 * row_count))

        cosine_values = numpy.cos(self.clusterer.angle_values)
        sine_values = numpy.sin(self.clusterer.angle_values)

        for plot_index, cluster_id in enumerate(self.clusterer.cluster_ids):
            cluster_mask = self.clusterer.labels_ == cluster_id
            cluster_ranges = self.clusterer.cleaned_lidar_ranges_time_series[cluster_mask, :]

            mean_ranges = numpy.mean(cluster_ranges, axis=0)

            mean_x_values = mean_ranges * cosine_values
            mean_y_values = mean_ranges * sine_values

            mean_polygon = numpy.column_stack((mean_x_values, mean_y_values))
            closed_mean_polygon = numpy.vstack((mean_polygon, mean_polygon[0:1, :]))

            centroid = self.calculate_polygon_centroid(mean_polygon)

            axes = pyplot.subplot(row_count, column_count, plot_index + 1)

            axes.plot(closed_mean_polygon[:, 0], closed_mean_polygon[:, 1], linewidth=2.0)
            axes.scatter(mean_polygon[:, 0], mean_polygon[:, 1], s=3)

            axes.scatter(
                [centroid[0]],
                [centroid[1]],
                s=100,
                marker="x",
                linewidths=2.0,
                color="red"
            )

            axes.text(
                centroid[0],
                centroid[1],
                f"  centroid\n  ({centroid[0]:.2f}, {centroid[1]:.2f})",
                fontsize=9,
                ha="left",
                va="bottom",
                color="red"
            )

            cluster_size_position_array = numpy.where(self.clusterer.cluster_ids == cluster_id)[0]
            cluster_size_position = int(cluster_size_position_array[0])
            cluster_size = int(self.clusterer.cluster_sizes[cluster_size_position])

            if int(cluster_id) == 0:
                title = (
                    f"Mean polygon - Cluster {int(cluster_id)}: wall/straight\n"
                    f"size={cluster_size}, centroid=({centroid[0]:.2f}, {centroid[1]:.2f})"
                )
            else:
                title = (
                    f"Mean polygon - Cluster {int(cluster_id)}: corner\n"
                    f"size={cluster_size}, centroid=({centroid[0]:.2f}, {centroid[1]:.2f})"
                )

            axes.set_title(title)
            axes.set_aspect("equal")
            axes.grid(True)

        pyplot.tight_layout()
        pyplot.show()

    def calculate_polygon_centroid(self, polygon: numpy.ndarray) -> numpy.ndarray:
        x_values = polygon[:, 0]
        y_values = polygon[:, 1]

        next_x_values = numpy.roll(x_values, -1)
        next_y_values = numpy.roll(y_values, -1)

        cross_values = x_values * next_y_values - next_x_values * y_values
        signed_area = 0.5 * numpy.sum(cross_values)

        if abs(signed_area) < 1e-12:
            centroid = numpy.mean(polygon, axis=0)
            return centroid

        centroid_x = numpy.sum((x_values + next_x_values) * cross_values) / (6.0 * signed_area)
        centroid_y = numpy.sum((y_values + next_y_values) * cross_values) / (6.0 * signed_area)

        centroid = numpy.asarray([centroid_x, centroid_y], dtype=float)

        return centroid

    def calculate_mean_polygon_centroids_for_clusters(self) -> dict:
        if self.clusterer is None:
            raise ValueError("Run the clusterer first.")

        cosine_values = numpy.cos(self.clusterer.angle_values)
        sine_values = numpy.sin(self.clusterer.angle_values)

        centroid_by_cluster_id = {}

        for cluster_id in self.clusterer.cluster_ids:
            cluster_mask = self.clusterer.labels_ == cluster_id
            cluster_ranges = self.clusterer.cleaned_lidar_ranges_time_series[cluster_mask, :]

            mean_ranges = numpy.mean(cluster_ranges, axis=0)

            mean_x_values = mean_ranges * cosine_values
            mean_y_values = mean_ranges * sine_values

            mean_polygon = numpy.column_stack((mean_x_values, mean_y_values))
            centroid = self.calculate_polygon_centroid(mean_polygon)

            centroid_by_cluster_id[int(cluster_id)] = centroid

        return centroid_by_cluster_id

    def print_mean_polygon_centroids_for_clusters(self) -> None:
        centroid_by_cluster_id = self.calculate_mean_polygon_centroids_for_clusters()

        print("Mean polygon centroids:")

        for cluster_id in sorted(centroid_by_cluster_id.keys()):
            centroid = centroid_by_cluster_id[cluster_id]
            print(f"cluster_id = {cluster_id}, centroid_x = {centroid[0]}, centroid_y = {centroid[1]}")


if __name__ == "__main__":
    plotter = ClusterPlotter(
        data_slice=slice(0, 50000),

        lidar_dimension=720,
        start_angle=-numpy.pi,
        end_angle=numpy.pi,

        far_range_threshold=2.5,
        replacement_distance=4.0,

        point_stride=4,
        descriptor_count=25,
        minimum_points_for_geometry=20,

        minimum_corner_fraction=0.03,
        maximum_corner_fraction=0.70,
        fallback_corner_quantile=0.80,

        minimum_corner_cluster_count=2,
        maximum_corner_cluster_count=8,
        maximum_silhouette_sample_count=10000,

        cluster_count_penalty=0.025,
        random_state=42
    )

    plotter.run()

    plotter.plot_one_representative_polygon_from_each_cluster()

    plotter.plot_multiple_examples_from_each_cluster(
        examples_per_cluster=10
    )

    plotter.plot_mean_polygon_from_each_cluster_with_centroid()