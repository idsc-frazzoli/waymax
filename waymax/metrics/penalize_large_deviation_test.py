from jax import numpy as jnp
import tensorflow as tf
from absl.testing import parameterized
import dataclasses

from waymax.metrics import penalize_large_deviation
from waymax.utils import test_utils


class PenalizeLargeDeviationMetricTest(tf.test.TestCase, parameterized.TestCase):
    def test_largedeviation(self):
        simulator_state = test_utils.simulator_state_with_offroad()
        modified_sim_trajectory = dataclasses.replace(
            simulator_state.sim_trajectory,
            x = simulator_state.sim_trajectory.x.at[0,...].set(6.0)
        )
        modified_simulator_state = dataclasses.replace(
            simulator_state,
            sim_trajectory = modified_sim_trajectory
        )
        result = penalize_large_deviation.PenalizeLargeDeviationMetric().compute(modified_simulator_state)
        self.assertEqual(result.value.shape, (1,))
        self.assertEqual(result.value, 1.0)

    def test_smalldeviation(self):
        simulator_state = test_utils.simulator_state_with_offroad()
        modified_sim_trajectory = dataclasses.replace(
            simulator_state.sim_trajectory,
            y = simulator_state.sim_trajectory.y.at[0,...].set(1.0)
        )
        modified_simulator_state = dataclasses.replace(
            simulator_state,
            sim_trajectory = modified_sim_trajectory
        )
        result = penalize_large_deviation.PenalizeLargeDeviationMetric().compute(modified_simulator_state)
        self.assertEqual(result.value.shape, (1,))
        self.assertEqual(result.value, 0.0)


if __name__ == '__main__':
  tf.test.main()