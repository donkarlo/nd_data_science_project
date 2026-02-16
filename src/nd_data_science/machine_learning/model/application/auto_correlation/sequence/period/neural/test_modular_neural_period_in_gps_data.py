from nd_data_science.machine_learning.model.application.auto_correlation.sequence.period.neural.architecture.architecture import \
    Architecture

from nd_data_science.machine_learning.model.application.auto_correlation.sequence.period.neural.neural_period_estimator import \
    NeuralPeriodEstimator
from nd_sociomind.experiment.kind.oldest.robotic.grouping.composite.child.group.member.uav1.structure.kind.mind.memory.explicit.long_term.episodic.auto_biographical.event_specific_knowledge.normal.sliced_from_1_to_300000.gaussianed_quaternion_kinematic.time_position.time_positions import \
    TimePositions
from nd_data_science.machine_learning.model.application.auto_correlation.sequence.period.neural.training.config import \
    Config as TrainingConfig

from nd_data_science.machine_learning.model.application.auto_correlation.sequence.period.neural.training.training import Training
from nd_data_science.machine_learning.model.application.auto_correlation.sequence.period.neural.training.learned_parameters import \
    LearnedParameters

from nd_data_science.machine_learning.model.application.auto_correlation.sequence.period.neural.predicting.predicting import Predicting

class TestModularNeuralPeriodInGpsData:
    def setup_method(self):
        self._time_positions_composite_memory = TimePositions(slice(0,50000))

    def test_clusterer(self):
        positions_sequence = self._time_positions_composite_memory.get_np_positions()

        # training phase
        architecture = Architecture(window_length=128, latent_size=16)
        training_config = TrainingConfig(batch_size=256, epochs=5)

        trainer = Training(architecture=architecture, training_config=training_config)
        learned_parameters = trainer.fit(positions_sequence)

        learned_parameters.save("/tmp/encoder_weights.npz")

        # predicting phase (later / separate process)
        architecture = Architecture(window_length=128, latent_size=16)
        learned_parameters = LearnedParameters.load("/tmp/encoder_weights.npz")

        predictor = Predicting(architecture=architecture, learned_parameters=learned_parameters, batch_size=256)
        period = predictor.estimate_period(positions_sequence, min_period=10)
        print(period)
        print("Finish")