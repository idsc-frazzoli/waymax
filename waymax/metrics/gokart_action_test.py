import tensorflow as tf
from absl.testing import parameterized
from jax import numpy as jnp

from waymax.metrics import GokartActionMetric, GokartActionRateMetric, GokartTVActionMetric
from gocarx.utils.gokart_utils import create_init_state
from waymax import datatypes


class GokartActionMetricTest(tf.test.TestCase, parameterized.TestCase):
    def test_actions(self):
        metric = GokartActionMetric()
        state = create_init_state(num_timesteps=5)

        state.history_actions = state.history_actions.set(
            datatypes.Action(data=jnp.zeros((1, 3)), valid=jnp.ones((1, 3))), 0
        )
        state.history_actions = state.history_actions.set(
            datatypes.Action(data=jnp.array([0.1, 0.2, 0.3]), valid=jnp.ones((1, 3))), 1
        )
        state.history_actions = state.history_actions.set(
            datatypes.Action(data=jnp.array([-0.9654, 0.676, -0.232]), valid=jnp.ones((1, 3))), 2
        )

        result = metric.compute(state)
        self.assertEqual(result.value, 0.0)

        state.timestep = 1
        result = metric.compute(state)
        self.assertEqual(result.value, 0.0)

        state.timestep = 2
        result = metric.compute(state)
        self.assertAllClose(result.value, 0.14)

        state.timestep = 3
        result = metric.compute(state)
        self.assertAllClose(result.value, 1.442797)

    def test_steering(self):
        metric = GokartActionMetric(["steering_angle"])
        state = create_init_state(num_timesteps=5)

        state.history_actions = state.history_actions.set(
            datatypes.Action(data=jnp.array([0.0, 0.676, -0.232]), valid=jnp.ones((1, 3))), 0
        )
        state.history_actions = state.history_actions.set(
            datatypes.Action(data=jnp.array([0.1, 0.2, 0.3]), valid=jnp.ones((1, 3))), 1
        )
        state.history_actions = state.history_actions.set(
            datatypes.Action(data=jnp.array([-0.9654, 0.676, -0.232]), valid=jnp.ones((1, 3))), 2
        )

        result = metric.compute(state)
        self.assertEqual(result.value, 0.0)

        state.timestep = 1
        result = metric.compute(state)
        self.assertEqual(result.value, 0.0)

        state.timestep = 2
        result = metric.compute(state)
        self.assertAllClose(result.value, 0.01)

        state.timestep = 3
        result = metric.compute(state)
        self.assertAllClose(result.value, 0.931997)

    def test_throttle(self):
        metric = GokartActionMetric(["AB_L", "AB_R"])
        state = create_init_state(num_timesteps=5)

        state.history_actions = state.history_actions.set(
            datatypes.Action(data=jnp.array([-0.9654, 0.0, 0.0]), valid=jnp.ones((1, 3))), 0
        )
        state.history_actions = state.history_actions.set(
            datatypes.Action(data=jnp.array([0.1, 0.2, 0.3]), valid=jnp.ones((1, 3))), 1
        )
        state.history_actions = state.history_actions.set(
            datatypes.Action(data=jnp.array([-0.9654, 0.676, -0.232]), valid=jnp.ones((1, 3))), 2
        )

        result = metric.compute(state)
        self.assertEqual(result.value, 0.0)

        state.timestep = 1
        result = metric.compute(state)
        self.assertEqual(result.value, 0.0)

        state.timestep = 2
        result = metric.compute(state)
        self.assertAllClose(result.value, 0.13)

        state.timestep = 3
        result = metric.compute(state)
        self.assertEqual(result.value, 0.5108)


class GokartActionRateMetricTest(tf.test.TestCase, parameterized.TestCase):

    def test_action_rates(self):
        metric = GokartActionRateMetric()
        state = create_init_state(num_timesteps=5)

        state.history_actions = state.history_actions.set(
            datatypes.Action(data=jnp.array([0.1, 0.2, 0.3]), valid=jnp.ones((1, 3))), 0
        )
        state.history_actions = state.history_actions.set(
            datatypes.Action(data=jnp.array([0.4, -0.6, 0.7]), valid=jnp.ones((1, 3))), 1
        )
        state.history_actions = state.history_actions.set(
            datatypes.Action(data=jnp.array([-0.654, 0.038, -0.283]), valid=jnp.ones((1, 3))), 2
        )
        state.history_actions = state.history_actions.set(
            datatypes.Action(data=jnp.array([-0.103, 0.812, -0.539]), valid=jnp.ones((1, 3))), 3
        )

        result = metric.compute(state)
        self.assertEqual(result.value, 0.0)

        state.timestep = 1
        result = metric.compute(state)
        self.assertEqual(result.value, 0.0)

        state.timestep = 2
        result = metric.compute(state)
        self.assertAllClose(result.value, 0.89)

        state.timestep = 3
        result = metric.compute(state)
        self.assertAllClose(result.value, 2.4842489999999997)

        state.timestep = 4
        result = metric.compute(state)
        self.assertAllClose(result.value, 0.9682130000000002)

    def test_steering_rate(self):
        metric = GokartActionRateMetric(["steering_angle"])
        state = create_init_state(num_timesteps=5)

        state.history_actions = state.history_actions.set(
            datatypes.Action(data=jnp.array([0.1, 0.2, 0.3]), valid=jnp.ones((1, 3))), 0
        )
        state.history_actions = state.history_actions.set(
            datatypes.Action(data=jnp.array([0.4, -0.6, 0.7]), valid=jnp.ones((1, 3))), 1
        )
        state.history_actions = state.history_actions.set(
            datatypes.Action(data=jnp.array([-0.654, 0.038, -0.283]), valid=jnp.ones((1, 3))), 2
        )
        state.history_actions = state.history_actions.set(
            datatypes.Action(data=jnp.array([-0.103, 0.812, -0.539]), valid=jnp.ones((1, 3))), 3
        )

        result = metric.compute(state)
        self.assertEqual(result.value, 0.0)

        state.timestep = 1
        result = metric.compute(state)
        self.assertEqual(result.value, 0.0)

        state.timestep = 2
        result = metric.compute(state)
        self.assertAllClose(result.value, 0.09)

        state.timestep = 3
        result = metric.compute(state)
        self.assertAllClose(result.value, 1.110916)

        state.timestep = 4
        result = metric.compute(state)
        self.assertAllClose(result.value, 0.303601)

    def test_throttle_rate(self):
        metric = GokartActionRateMetric(["AB_L", "AB_R"])
        state = create_init_state(num_timesteps=5)

        state.history_actions = state.history_actions.set(
            datatypes.Action(data=jnp.array([0.1, 0.2, 0.3]), valid=jnp.ones((1, 3))), 0
        )
        state.history_actions = state.history_actions.set(
            datatypes.Action(data=jnp.array([0.4, -0.6, 0.7]), valid=jnp.ones((1, 3))), 1
        )
        state.history_actions = state.history_actions.set(
            datatypes.Action(data=jnp.array([-0.654, 0.038, -0.283]), valid=jnp.ones((1, 3))), 2
        )
        state.history_actions = state.history_actions.set(
            datatypes.Action(data=jnp.array([-0.103, 0.812, -0.539]), valid=jnp.ones((1, 3))), 3
        )

        result = metric.compute(state)
        self.assertEqual(result.value, 0.0)

        state.timestep = 1
        result = metric.compute(state)
        self.assertEqual(result.value, 0.0)

        state.timestep = 2
        result = metric.compute(state)
        self.assertAllClose(result.value, 0.8)

        state.timestep = 3
        result = metric.compute(state)
        self.assertAllClose(result.value, 1.373332)

        state.timestep = 4
        result = metric.compute(state)
        self.assertAllClose(result.value, 0.664612)


class GokartTVActionMetricTest(tf.test.TestCase, parameterized.TestCase):

    def test_tv_action(self):

        metric = GokartTVActionMetric()
        state = create_init_state(num_timesteps=5)

        state.history_actions = state.history_actions.set(
            datatypes.Action(data=jnp.array([-0.9654, 0.0, 0.0]), valid=jnp.ones((1, 3))), 0
        )
        state.history_actions = state.history_actions.set(
            datatypes.Action(data=jnp.array([0.1, 0.2, 0.3]), valid=jnp.ones((1, 3))), 1
        )
        state.history_actions = state.history_actions.set(
            datatypes.Action(data=jnp.array([-0.9654, 0.676, -0.232]), valid=jnp.ones((1, 3))), 2
        )

        result = metric.compute(state)
        self.assertEqual(result.value, 0.0)

        state.timestep = 1
        result = metric.compute(state)
        self.assertEqual(result.value, 0.0)

        state.timestep = 2
        result = metric.compute(state)
        self.assertAllClose(result.value, 0.01)

        state.timestep = 3
        result = metric.compute(state)
        self.assertAllClose(result.value, 0.824464)


if __name__ == "__main__":
    tf.test.main()
