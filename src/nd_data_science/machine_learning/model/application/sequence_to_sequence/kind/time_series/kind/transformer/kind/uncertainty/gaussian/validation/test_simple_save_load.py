# file: example_gaussian_transformer_train_save_load_predict.py
from __future__ import annotations

from pathlib import Path

from nd_data_science.machine_learning.model.application.sequence_to_sequence.kind.time_series.kind.transformer.kind.uncertainty.gaussian.architecture.architecture import \
    Architecture
from nd_data_science.machine_learning.model.application.sequence_to_sequence.kind.time_series.kind.transformer.kind.uncertainty.gaussian.prediction.predictor import \
    Predictor
from nd_data_science.machine_learning.model.application.sequence_to_sequence.kind.time_series.kind.transformer.kind.uncertainty.gaussian.training.config import \
    Config
from nd_data_science.machine_learning.model.application.sequence_to_sequence.kind.time_series.kind.transformer.kind.uncertainty.gaussian.architecture.storage import Storage as ArchitectureYamlRepository
from nd_data_science.machine_learning.model.application.sequence_to_sequence.kind.time_series.kind.transformer.kind.uncertainty.gaussian.training.storaged import Storaged as ConfigYamlRepository
from nd_data_science.machine_learning.model.application.sequence_to_sequence.kind.time_series.kind.transformer.kind.uncertainty.gaussian.training.learned_parameter.storage import Storage as LearnedParameterNpzRepository

from nd_math.probability.statistic.population.sampling.sampler.kind.countable.finite.members_mentioned.numbered.sequence.sliding_window.sliding_window import \
    SlidingWindow
from robotic_nd.robot.structure.kind.mind.process.kind.memory.action.kind.intra.binary.segregation.segregator.kind.trace_group import \
    TraceGroup as NormalGpsTraceGroup300k


def test() -> None:
    sliding_window = SlidingWindow(100, 100, 5)

    normal_gps_trace_group = NormalGpsTraceGroup300k(False)
    training_input_target_pairs, testing_input_target_pairs = normal_gps_trace_group.get_periods_by_8_successuive_trains_and_1_test(
        sliding_window)

    feature_dimension = int(training_input_target_pairs.shape[3])

    architecture = Architecture(
        model_dimension=64,
        number_of_attention_heads=8,
        feed_forward_dimension=128,
        input_feature_count=feature_dimension,
        output_time_steps=int(sliding_window.get_output_length()),
        output_feature_count=feature_dimension,
        maximum_time_steps=2048,
        dropout_rate=0.1,
    )

    trainer_config = Config(
        epochs=2,
        batch_size=4,
        learning_rate=1e-3,
        shuffle=True,
    )

    trainer = Config(architecture, trainer_config, training_input_target_pairs)
    learned_parameters = trainer.get_learned_parameters()

    checkpoint_directory = Path("./gaussian_transformer_checkpoint")
    architecture_file_path = checkpoint_directory / "config.yaml"
    trainer_config_file_path = checkpoint_directory / "training_config.yaml"
    weights_file_path = checkpoint_directory / "weights.npz"

    ArchitectureYamlRepository().save(architecture, architecture_file_path)
    ConfigYamlRepository().save(trainer_config, trainer_config_file_path)
    LearnedParameterNpzRepository().save(learned_parameters, weights_file_path)

    loaded_architecture = ArchitectureYamlRepository().load(architecture_file_path)
    loaded_learned_parameters = LearnedParameterNpzRepository().load(weights_file_path)

    predictor = Predictor(loaded_architecture, loaded_learned_parameters)

    test_input_array = testing_input_target_pairs[5:6, 0]
    test_target_array = testing_input_target_pairs[5:6, 1]

    mu, var = predictor.get_predicted_distributions(test_input_array)
    print("target_array: ", test_target_array)
    print("mean: ", mu)
    print("var: ", var)
