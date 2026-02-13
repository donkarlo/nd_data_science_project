from nd_data_science.machine_learning.model.application.sequence_to_sequence.kind.time_series.kind.transformer.architecture.architecture import \
    Architecture
from nd_data_science.machine_learning.model.application.sequence_to_sequence.kind.time_series.kind.transformer.prediction.predictor import \
    Predictor
from nd_data_science.machine_learning.model.application.sequence_to_sequence.kind.time_series.kind.transformer.training.training import \
    Config
from nd_data_science.machine_learning.model.application.sequence_to_sequence.validation.kind.train_test.train_test_by_point_sampling import \
    TrainTestByPointSampling

from nd_math.probability.statistic.population.kind.countable.finite.member_mentioned.numbered.numbered import Numbered as NumberedPopulation
from nd_math.view.kind.point_cloud.point_cloud import PointCloud
from nd_utility.data.storage.kind.file.numpi.multi_valued import MultiValued as NpMultiValued
from nd_utility.os.file_system.file.file import File as OsFile
from nd_utility.os.file_system.path.file import File as FilePath
from nd_math.view.kind.point_cloud.point.group.group import Group as PointGroup


class TestTrainTestByPointSampling:
    def test_plot_mean_euclidean_distance_plot(self):
        file_path = FilePath(
            "/home/donkarlo/Dropbox/phd/data/experiements/oldest/robots/uav1/structure/mind/memory/explicit/long_term/episodic/normal/gaussianed_quaternion_kinematic/time_position_sequence_sliced_from_1_to_300000.npz"
        )
        os_file = OsFile.init_from_path(file_path)
        storage = NpMultiValued(os_file, False)
        storage.load()
        # removing the time
        one_cycle_count = 24450
        ram = storage.get_ram()[0:60000, 1:]

        cloud_point_math_view = PointCloud(PointGroup(ram))
        cloud_point_math_view.render()


        architecture = Architecture(
            model_dimension=128,
            number_of_attention_heads=8,
            feed_forward_dimension=256,
            input_feature_count=3,
            output_time_steps=100,
            output_feature_count=3,
            maximum_time_steps=2048,
            dropout_rate=0.1,
        )

        trainer_config = Config(epochs=5,
                                batch_size=16,
                                learning_rate=1e-3,
                                shuffle=True)

        population = NumberedPopulation(ram)
        train_test = TrainTestByPointSampling.init_from_partitionaning_ratio(Config, architecture, trainer_config, Predictor, population, 0.7)
        train_test.render_euclidean_distance()
