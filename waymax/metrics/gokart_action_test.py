import tensorflow as tf
from absl.testing import parameterized
from jax import numpy as jnp

from waymax.metrics import GokartActionMetric, GokartActionRateMetric, GokartActionTVMetric
from gocarx.utils.gokart_utils import create_init_state
from waymax import datatypes


class GokartActionMetricTest(tf.test.TestCase, parameterized.TestCase):
    def test_actions(self):
        metric = GokartActionMetric()
        state = create_init_state(num_timesteps=5)
        state.history_actions = state.history_actions.set(
            datatypes.Action(data=jnp.zeros((1, 3)), valid=jnp.ones((1, 3))), 0
        )
        result = metric.compute(state)
        self.assertEqual(result.value, 0.0)

        state.history_actions = state.history_actions.set(
            datatypes.Action(data=jnp.array([0.1, 0.2, 0.3]), valid=jnp.ones((1, 3))), 0
        )
        result = metric.compute(state)
        print(result.value, 0.14)
        self.assertEqual(result.value, 0.14)

        state.history_actions = state.history_actions.set(
            datatypes.Action(data=jnp.array([-0.7654, 0.876, -0.432]), valid=jnp.ones((1, 3))), 0
        )
        result = metric.compute(state)
        # should be zero, because the velocity is zero
        self.assertEqual(result.value, 1.5398371599999998)

    def test_steering(self):
        return
        metric = GokartActionMetric(["steering_angle"])
        state = create_init_state(num_timesteps=5)
        state.history_actions = state.history_actions.set(
            datatypes.Action(data=jnp.array([0.0, 0.876, -0.432]), valid=jnp.ones((1, 3))), 0
        )
        result = metric.compute(state)
        # should be zero, because the velocity is zero
        self.assertEqual(result.value, 0.0)

        state.history_actions = state.history_actions.set(
            datatypes.Action(data=jnp.array([0.1, 0.2, 0.3]), valid=jnp.ones((1, 3))), 0
        )
        result = metric.compute(state)
        # should be zero, because the velocity is zero
        self.assertEqual(result.value, 0.0)

        state.history_actions = state.history_actions.set(
            datatypes.Action(data=jnp.array([-0.7654, 0.876, -0.432]), valid=jnp.ones((1, 3))), 0
        )
        result = metric.compute(state)
        # should be zero, because the velocity is zero
        self.assertEqual(result.value, 0.0)

    def test_throttle(self):
        return
        metric = GokartActionMetric(["AB_L", "AB_R"])
        state = create_init_state(num_timesteps=5)
        state.history_actions = state.history_actions.set(
            datatypes.Action(data=jnp.array([0.4, 0.0, 0.0]), valid=jnp.ones((1, 3))), 0
        )
        result = metric.compute(state)
        # should be zero, because the velocity is zero
        self.assertEqual(result.value, 0.0)

        state.history_actions = state.history_actions.set(
            datatypes.Action(data=jnp.array([0.1, 0.2, 0.3]), valid=jnp.ones((1, 3))), 0
        )
        result = metric.compute(state)
        # should be zero, because the velocity is zero
        self.assertEqual(result.value, 0.0)

        state.history_actions = state.history_actions.set(
            datatypes.Action(data=jnp.array([-0.7654, 0.876, -0.432]), valid=jnp.ones((1, 3))), 0
        )
        result = metric.compute(state)
        # should be zero, because the velocity is zero
        self.assertEqual(result.value, 0.0)


if __name__ == "__main__":
    tf.test.main()
