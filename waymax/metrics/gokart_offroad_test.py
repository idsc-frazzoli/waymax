import tensorflow as tf
from absl.testing import parameterized

from gocarx.metrics.gokart_offroad import GokartOffroadMetric
from gocarx.utils.gokart_utils import create_init_state


class GokartOffroadMetricTest(tf.test.TestCase, parameterized.TestCase):
    def test_onroad(self):
        metric = GokartOffroadMetric()
        state = create_init_state(num_timesteps=100)
        result = metric.compute(state)
        # should be zero, because the car is not offroad
        self.assertEqual(result.value, 0.0)

    def test_offroad(self):
        metric = GokartOffroadMetric()
        state = create_init_state(num_timesteps=100)
        current_y = state.current_sim_trajectory.x[..., 0, 0]
        # move the car offroad
        current_y -= 2
        state.sim_trajectory.y = state.sim_trajectory.y.at[..., 0, 0].set(current_y)
        result = metric.compute(state)
        # should be negative, because the car is offroad
        self.assertEqual(result.value, 1.0)