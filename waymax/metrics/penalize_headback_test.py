from jax import numpy as jnp
import tensorflow as tf
from absl.testing import parameterized
import dataclasses

from waymax.metrics import penalize_headback
from waymax.utils import test_utils


class PenalizeHeadbackMetricTest(tf.test.TestCase, parameterized.TestCase):
    def test_headback(self):
        simulator_state = test_utils.simulator_state_with_offroad() # only velocity matters here
        modified_simulator_state = dataclasses.replace(
            simulator_state,
            vel_x = simulator_state.vel_x.at[...,0].set(-1.0)
        )
        result = penalize_headback.PenalizeHeadbackMetric().compute(modified_simulator_state)
        self.assertEqual(result.value.shape, (1,))
        self.assertEqual(result.value, 1.0)

    def test_not_headback(self):
        simulator_state = test_utils.simulator_state_with_offroad() # only velocity matters here
        modified_simulator_state = dataclasses.replace(
            simulator_state,
            vel_y = simulator_state.vel_y.at[...,0].set(-1.0)
        )
        result = penalize_headback.PenalizeHeadbackMetric().compute(modified_simulator_state)
        self.assertEqual(result.value.shape, (1,))
        self.assertEqual(result.value, 0.0)


if __name__ == '__main__':
  tf.test.main()