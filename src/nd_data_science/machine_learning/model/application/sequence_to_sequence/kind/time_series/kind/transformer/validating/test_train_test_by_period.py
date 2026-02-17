import numpy as np

from nd_data_science.machine_learning.model.application.sequence_to_sequence.kind.time_series.kind.transformer.architecture.architecture import \
    Architecture
from nd_data_science.machine_learning.model.application.sequence_to_sequence.kind.time_series.kind.transformer.prediction.predictor import \
    Predictor
from nd_data_science.machine_learning.model.application.sequence_to_sequence.kind.time_series.kind.transformer.kind.uncertainty.gaussian.training.config import \
    Config as TrainerConfig
from nd_data_science.machine_learning.model.application.sequence_to_sequence.kind.time_series.kind.transformer.training.training import \
    Config
from nd_data_science.machine_learning.model.application.sequence_to_sequence.validation.kind.train_test.train_test_by_periods import \
    TrainTestByPeriods
from nd_math.probability.statistic.population.sampling.kind.countable.finite.members_mentioned.numbered.sequence.sliding_window.sliding_window import SlidingWindow
from nd_utility.data.storage.kind.file.numpi.multi_valued import MultiValued as NpMultiValued
from nd_utility.os.file_system.file.file import File as OsFile
from nd_utility.os.file_system.path.file import File as FilePath


class TestTrainTestByPeriodSampling:
    def test_plot_mean_euclidean_distance_plot(self):
        file_path = FilePath(
            "/nd_sociomind/experiment/member/oldest/robotic_group/uav1/grouping/mind/memory/explicit/long_term/episodic/normal/gaussianed_quaternion_kinematic/time_position/time_position.npz"
        )
        os_file = OsFile.init_from_path(file_path)
        storage = NpMultiValued(os_file, False)
        storage.load()
        # removing the time
        one_period_members_count = 24450
        ram = storage.get_ram()[:10 * 24450, 1:]  # (T, 3)

        usable_len = (len(ram) // one_period_members_count) * one_period_members_count
        ram = ram[:usable_len]

        partition_count = len(ram) // one_period_members_count

        # shape = (partition_count, one_period_members_count, 3)
        partitioned_population = ram.reshape(partition_count, one_period_members_count, ram.shape[1])
        print(partitioned_population[0].shape)
        training_partitions = np.vstack(
            [partitioned_population[0], partitioned_population[1], partitioned_population[2], partitioned_population[3],
             partitioned_population[4], partitioned_population[5], partitioned_population[6], partitioned_population[7]])
        testing_partition = np.vstack([partitioned_population[8], partitioned_population[9]])

        print("ram.shape:", ram.shape, "ram.nbytes(MB):", ram.nbytes / 1024 / 1024)
        print("partitioned_population.shape:", partitioned_population.shape, "dtype:", partitioned_population.dtype)

        # cloud_point_math_view = PointCloud(PointGroup(ram))
        # cloud_point_math_view.render()

        # sliding window trainer_config
        sliding_window = SlidingWindow(100, 100, 10)

        # config trainer_config
        feature_dimension = ram.shape[1]
        architecture = Architecture(
            model_dimension=64,
            number_of_attention_heads=8,
            feed_forward_dimension=128,
            input_feature_dimension=feature_dimension,  # GPS without time has 3 dimensions
            output_sequence_size=sliding_window.get_input_length(),  # the length of each sliding window as the Predicting length
            output_feature_dimension=feature_dimension,  # we want 3d GPS predictions
            maximum_time_steps=2048,
            dropout_rate=0.1,
        )

        # training trainer_config
        trainer_config = TrainerConfig(
            epochs=15,
            batch_size=8,
            learning_rate=1e-3,
            shuffle=True)

        train_test = TrainTestByPeriods.init_from_one_split(training_partitions, testing_partition,
                                                            Config, architecture, trainer_config, Predictor,
                                                            sliding_window.get_overlap_size())
        train_test.render_euclidean_distance()
