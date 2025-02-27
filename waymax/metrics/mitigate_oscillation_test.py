from jax import numpy as jnp
import tensorflow as tf
from absl.testing import parameterized
import dataclasses

from waymax import datatypes
from waymax.metrics import mitigate_oscillation
from waymax.utils import test_utils


class MitigateOscillationMetricTest(tf.test.TestCase, parameterized.TestCase):
    def test_oscillation(self):
        simulator_state = test_utils.simulator_state_with_offroad() # only action_history matters here
        modified_simulator_state = dataclasses.replace(
            simulator_state,
            action_history = datatypes.SDC_actions_history(
                data=jnp.array([[1.0, 1.0],[-1.0, -1.0]]), valid=jnp.zeros((2,1), dtype=jnp.bool_)
            )
        )
        result = mitigate_oscillation.PenalizeHeadbackMetric().compute(modified_simulator_state)
        self.assertEqual(result.value.shape, ())
        self.assertEqual(result.value, -2.0)

    def test_no_oscillation(self):
        simulator_state = test_utils.simulator_state_with_offroad() # only action_history matters here
        modified_simulator_state = dataclasses.replace(
            simulator_state,
            action_history = datatypes.SDC_actions_history(
                data=jnp.array([[1.0, -2.0],[2.0, -1.0]]), valid=jnp.zeros((2,1), dtype=jnp.bool_)
            )
        )
        result = mitigate_oscillation.PenalizeHeadbackMetric().compute(modified_simulator_state)
        self.assertEqual(result.value.shape, ())
        self.assertEqual(result.value, 0.0)


if __name__ == '__main__':
  tf.test.main()