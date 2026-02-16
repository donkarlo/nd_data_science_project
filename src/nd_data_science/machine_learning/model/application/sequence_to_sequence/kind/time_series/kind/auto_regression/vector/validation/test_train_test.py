from nd_data_science.machine_learning.model.application.sequence_to_sequence.kind.time_series.kind.auto_regression.vector.Training.config import \
    Config
from nd_data_science.machine_learning.model.application.sequence_to_sequence.kind.time_series.kind.auto_regression.vector.Training.trainer import \
    Trainer
from nd_data_science.machine_learning.model.application.sequence_to_sequence.kind.time_series.kind.auto_regression.vector.architecture import \
    Architecture
from nd_data_science.machine_learning.model.application.sequence_to_sequence.kind.time_series.kind.auto_regression.vector.predictor import \
    Predictor
from nd_data_science.machine_learning.model.application.sequence_to_sequence.validation.kind.train_test.train_test_sliding_window_sampling import TrainTestBySlidingWindowSampling
from nd_math.probability.statistic.population.sampling.sampling import \
    Generator
from nd_math.probability.statistic.population.sampling.sampling import \
    SlidingWindow
from nd_math.number.kind.real.interval.unit.open_unit_interval import OpenUnitInterval
from nd_math.probability.statistic.population.kind.countable.finite.member_mentioned.numbered.numbered import Numbered as NumpiedPopulation
from nd_utility.data.storage.kind.file.numpi.multi_valued import MultiValued as NpMultiValued
from nd_utility.os.file_system.file.file import File as OsFile
from nd_utility.os.file_system.path.file import File as FilePath


class TestTrainTest:

    def test_with_sampling_from_random_sliding_windows_papulation_ratio(self):
        file_path = FilePath(
            "/home/donkarlo/Dropbox/phd/data/experiements/oldest/robots/uav1/structure/mind/memory/explicit/long_term/episodic/normal/gaussianed_quaternion_kinematic/time_position_sequence_sliced_from_1_to_300000.npz"
        )
        os_file = OsFile.init_from_path(file_path)
        storage = NpMultiValued(os_file, False)
        storage.load()
        # removing the time
        ram = storage.get_ram()[0:50000, 1:]

        sliding_window = SlidingWindow(100, 100, 5)
        sliding_windows_generator = Generator(ram, sliding_window)

        input_array = sliding_windows_generator.get_inputs()
        target_array = sliding_windows_generator.get_outputs()
        input_target_pairs = sliding_windows_generator.get_input_output_pairs()

        architecture = Architecture(
            feature_count=3,
            lag_order=1,
            include_intercept=True
        )

        training_config = Config(fit_method="ols",
                                regularization_strength=None,
                                select_lag_by=None)

        sample_population = NumpiedPopulation(input_target_pairs)
        open_unit_interval = OpenUnitInterval(0.7)
        train_test = TrainTestBySlidingWindowSampling.init_from_partitionaning_ratio(Trainer, architecture, training_config, Predictor,
                                                                                     sample_population, open_unit_interval)
        train_test.render_euclidean_distance()