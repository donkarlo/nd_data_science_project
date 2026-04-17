import numpy as np

from nd_datascience.machine_learning.model.application.sequence_to_sequence.kind.time_series.kind.transformer.kinds.uncertainty.gaussian.predicting.predicting import \
    Predicting
from nd_datascience.machine_learning.model.application.sequence_to_sequence.kind.time_series.kind.transformer.kinds.uncertainty.gaussian.training.training import \
    Training
from nd_datascience.machine_learning.model.application.sequence_to_sequence.kind.time_series.kind.transformer.kinds.uncertainty.gaussian.validation.train_test_by_periods import \
    TrainTestByPeriods
from nd_math.number.kind.real.interval.unit.close_unit_interval import CloseUnitInterval
from nd_math.probability.statistic.population.kind.countable.finite.member_mentioned.numbered.numbered import \
    Numbered as NumberedPopulation
from nd_math.probability.statistic.population.sampling.kind.countable.finite.members_mentioned.numbered.random.random import \
    Random
from nd_math.probability.statistic.population.sampling.kind.countable.finite.members_mentioned.numbered.sequence.sliding_window.generator import \
    Generator as SlidingWindowGenerator
from nd_math.probability.statistic.population.sampling.kind.countable.finite.members_mentioned.numbered.sequence.sliding_window.sliding_window import \
    SlidingWindow
from nd_math.probability.statistic.population.sampling.size.kind.ratio import Ratio
from nd_math.sequence.partitioning.kind.by_length.by_length import ByLength as ByLengthSequencePartitioning
from nd_robotic_ai.robot.parts.mind.parts.cognition.parts.object_level.process.parts.memory.parts.implicit.repetition_priming.kinds.event_specific_knowledge_forcasting_model_config.architecture import Architecture as RepititionPrimingArchitecture
from nd_robotic_ai.robot.parts.mind.parts.cognition.parts.object_level.process.parts.memory.parts.implicit.repetition_priming.kinds.event_specific_knowledge_forcasting_model_config.training.config import Config as RepititionPrimingTrainingConfig
from nd_sociomind.experiment.parts.oldest.uav1_300k_normal_time_position_modality import \
    Uav1300kNormalTimePositionModality
from nd_utility.data.kind.dic.dic import Dic


class TestTrainTestByPeriodSampling:
    def setup_method(self):
        # This can be discovered by autocorellation
        self._one_period_members_count = 24450
        slc = slice(0, 10 * 24450)
        self._time_positions_composite_memory = Uav1300kNormalTimePositionModality(slc)

    def test_plot_mean_euclidean_distance_plot(self):
        positions_sequence = self._time_positions_composite_memory.get_np_positions()

        partitioned_position_sequence = ByLengthSequencePartitioning(positions_sequence,
                                                                     self._one_period_members_count).get_full_length_partitions()

        partitioned_population = NumberedPopulation(partitioned_position_sequence)
        random_sampling = Random(partitioned_population, Ratio(CloseUnitInterval(0.7), len(
            partitioned_position_sequence)))
        training_sequence = np.vstack(random_sampling.get_samples())
        testing_sequence = np.vstack(random_sampling.get_complements())


        feature_dimension = training_sequence.shape[1]  # for position without time, it must give three
        training_sequence_size = training_sequence.shape[0]

        model_architecture = RepititionPrimingArchitecture(Dic({
            "input_feature_dimension": feature_dimension,
            "output_feature_dimension": feature_dimension}))
        trainer_config = RepititionPrimingTrainingConfig(Dic({
            "training_sequence_size": training_sequence_size}
        ))




        sliding_window = SlidingWindow(trainer_config.get_config_dic()["input_sequence_size"],
                                       trainer_config.get_config_dic()["output_sequence_size"],
                                       trainer_config.get_config_dic()["sequence_overlap_size"])

        # generating sliding window
        training_input_target_sequence_pairs = SlidingWindowGenerator(training_sequence,
                                                                      sliding_window).get_input_output_pairs()
        testing_input_target_sequence_pairs = SlidingWindowGenerator(testing_sequence,
                                                                     sliding_window).get_input_output_pairs()

        # training
        trainer = Training(model_architecture, trainer_config, positions_sequence)
        learned_parameters = trainer.get_learned_parameters()

        predictor = Predicting(model_architecture, learned_parameters)

        train_test = TrainTestByPeriods(predictor, training_input_target_sequence_pairs,
                                        testing_input_target_sequence_pairs)
        train_test.render_euclidean_distance()
