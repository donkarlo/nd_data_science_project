from nd_data_science.machine_learning.model.application.sequence_to_sequence.kind.time_series.kind.transformer.kind.uncertainty.gaussian.architecture.architecture import \
    Architecture as ModelArchitecture
from nd_data_science.machine_learning.model.application.sequence_to_sequence.kind.time_series.kind.transformer.kind.uncertainty.gaussian.training.config import \
    Config as TrainerConfig
from nd_data_science.machine_learning.model.application.sequence_to_sequence.kind.time_series.kind.transformer.kind.uncertainty.gaussian.predicting.predicting import \
    Predicting
from nd_data_science.machine_learning.model.application.sequence_to_sequence.kind.time_series.kind.transformer.kind.uncertainty.gaussian.training.training import \
    Training
from nd_data_science.machine_learning.model.application.sequence_to_sequence.kind.time_series.kind.transformer.kind.uncertainty.gaussian.validation.train_test_by_periods import \
    TrainTestByPeriods
from nd_math.probability.statistic.population.sampling.sampler.kind.countable.finite.members_mentioned.numbered.sequence.sliding_window.generator import \
    Generator as SlidingWindowGenerator
from nd_math.probability.statistic.population.sampling.sampler.kind.countable.finite.members_mentioned.numbered.sequence.sliding_window.sliding_window import \
    SlidingWindow
from nd_utility.os.file_system.file.file import File as OsFile
from nd_utility.os.file_system.path.path import Path as FilePath
from nd_utility.data.storage.kind.file.numpi.multi_valued import MultiValued as NpMultiValued
import numpy as np


class TestTrainTestByPeriodSampling:
    def test_plot_mean_euclidean_distance_plot(self):
        file_path = FilePath(
            "//data/oldest/robotic/robotic/robotic/composite/uav1/structure/kind/mind/memory/explicit/long_term/episodic/normal/gaussianed_quaternion_kinematic/time_position/time_position.npz"
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
        training_input_target_pairs = np.vstack(
            [partitioned_population[0], partitioned_population[1], partitioned_population[2], partitioned_population[3],
             partitioned_population[4], partitioned_population[5], partitioned_population[6],
             partitioned_population[7]])
        testing_input_target_pairs = np.vstack([partitioned_population[8], partitioned_population[9]])

        training_length = len(training_input_target_pairs)
        feature_dimension = training_input_target_pairs.shape[1]

        sliding_window = SlidingWindow(100, 100, 10)
        training_input_target_sequence_pairs = SlidingWindowGenerator(training_input_target_pairs, sliding_window).get_input_output_pairs()
        testing_input_target_sequence_pairs = SlidingWindowGenerator(testing_input_target_pairs, sliding_window).get_input_output_pairs()

        model_architecture = ModelArchitecture(
            model_dimension=64,
            number_of_attention_heads=8,
            feed_forward_dimension=128,
            input_feature_count=feature_dimension,
            output_time_steps=sliding_window.get_output_length(),
            output_feature_count=feature_dimension,
            maximum_time_steps=2048,
            dropout_rate=0.1,
        )
        trainer_config = TrainerConfig(
            training_sequence_size= training_length,
            input_sequence_size = 100,
            output_sequence_size = 100,
            sequence_overlap_size = 10,
            epochs=10,
            batch_size=4,
            learning_rate=1e-3,
            shuffle=True,
        )

        trainer = Training(model_architecture, trainer_config, training_input_target_sequence_pairs)
        learned_parameters = trainer.get_learned_parameters()

        predictor = Predicting(model_architecture, learned_parameters)

        train_test = TrainTestByPeriods(predictor, training_input_target_sequence_pairs, testing_input_target_sequence_pairs)
        train_test.render_euclidean_distance()
