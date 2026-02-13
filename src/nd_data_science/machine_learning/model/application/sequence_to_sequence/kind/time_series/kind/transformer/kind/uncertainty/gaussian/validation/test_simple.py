from nd_data_science.machine_learning.model.application.sequence_to_sequence.kind.time_series.kind.transformer.kind.uncertainty.gaussian.architecture.architecture import \
    Architecture
from nd_data_science.machine_learning.model.application.sequence_to_sequence.kind.time_series.kind.transformer.kind.uncertainty.gaussian.prediction.predictor import \
    Predictor
from nd_data_science.machine_learning.model.application.sequence_to_sequence.kind.time_series.kind.transformer.kind.uncertainty.gaussian.training.config import Config
from nd_math.probability.statistic.population.sampling.sampler.kind.countable.finite.members_mentioned.numbered.sequence.sliding_window.sliding_window import \
    SlidingWindow
from robotic_nd.robot.structure.kind.mind.process.kind.memory.action.kind.intra.binary.segregation.segregator.kind.trace_group import \
    TraceGroup as NormalGpsTraceGroup300k




sliding_window = SlidingWindow(100, 100, 5)

normal_gps_trace_group = NormalGpsTraceGroup300k(False)
(training_input_target_pairs, testing_input_target_pairs) = normal_gps_trace_group.get_periods_by_8_successuive_trains_and_1_test(
    sliding_window)
feature_dimension = training_input_target_pairs.shape[3]

architecture = Architecture(
    model_dimension=64,
    number_of_attention_heads=8,
    feed_forward_dimension=128,
    input_feature_count=feature_dimension,
    output_time_steps=sliding_window.get_output_length(),
    output_feature_count=feature_dimension,
    maximum_time_steps=2048,
    dropout_rate=0.1,
)
trainer_config = Config(
    epochs=10,
    batch_size=4,
    learning_rate=1e-3,
    shuffle=True
)




trainer = Config(architecture, trainer_config, training_input_target_pairs)
learned_parameters = trainer.get_learned_parameters()

predictor = Predictor(architecture, learned_parameters)

test_input_array = testing_input_target_pairs[5:6, 0]  # shape: (1, 100, 3)
test_target_array = testing_input_target_pairs[5:6, 1] # # shape: (1, 100, 3)
mu, var = predictor.get_predicted_distributions(test_input_array)
print("target_array: ", test_target_array)
print("mean: ", mu)
print("var: ", var)