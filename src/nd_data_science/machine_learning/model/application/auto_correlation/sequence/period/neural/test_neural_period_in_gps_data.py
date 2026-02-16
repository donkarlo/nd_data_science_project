

from nd_data_science.machine_learning.model.application.auto_correlation.sequence.period.neural.neural_period_estimator import \
    NeuralPeriodEstimator
from nd_sociomind.experiment.kind.oldest.robotic.grouping.composite.child.group.member.uav1.structure.kind.mind.memory.explicit.long_term.episodic.auto_biographical.event_specific_knowledge.normal.sliced_from_1_to_300000.gaussianed_quaternion_kinematic.time_position.time_positions import \
    TimePositions


class TestNeuralPeriodInGpsData:
    def setup_method(self):
        self._time_positions_composite_memory = TimePositions(slice(0,50000))

    def test_clusterer(self):
        estimator = NeuralPeriodEstimator(
            window_length=128,
            latent_size=16,
            batch_size=256,
            epochs=5
        )
        sequence = self._time_positions_composite_memory.get_np_positions()
        estimator.fit(self._time_positions_composite_memory.get_np_positions())
        period = estimator.estimate_period(sequence, min_period=10)

        one_period = sequence[:period]

        print("Estimated period:", period)
        print("One period shape:", one_period.shape)
