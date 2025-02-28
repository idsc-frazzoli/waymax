from jax import numpy as jnp
import tensorflow as tf
from absl.testing import parameterized
import dataclasses

from waymax.metrics import get_target
from waymax.utils import test_utils


class GetTargetMetricTest(tf.test.TestCase, parameterized.TestCase):
    def test_get_target(self):
        simulator_state = test_utils.simulator_state_with_offroad() # only xy matters here
        result = get_target.GetTargetMetric().compute(simulator_state)
        self.assertEqual(result.value.shape, (1,))
        self.assertEqual(result.value, 1.0)
    
    def test_non_terminal(self):
        simulator_state = test_utils.simulator_state_with_offroad() # only xy matters here
        modified_simulator_state = dataclasses.replace(
            simulator_state,
            timestep = -1,
        )
        result = get_target.GetTargetMetric().compute(modified_simulator_state)
        self.assertEqual(result.value.shape, (1,))
        self.assertEqual(result.value, 0.0)

    def test_not_get_target(self):
        simulator_state = test_utils.simulator_state_with_offroad() # only xy matters here
        modified_sim_trajectory = dataclasses.replace(
            simulator_state.sim_trajectory,
            x = simulator_state.sim_trajectory.x.at[0,...].set(-1.0)
        )
        modified_simulator_state = dataclasses.replace(
            simulator_state,
            sim_trajectory = modified_sim_trajectory
        )
        result = get_target.GetTargetMetric().compute(modified_simulator_state)
        self.assertEqual(result.value.shape, (1,))
        self.assertEqual(result.value, 0.0)


if __name__ == '__main__':
  tf.test.main()