import tensorflow as tf
from absl.testing import parameterized
from jax import numpy as jnp

from gocarx.utils.gokart_utils import init_gokart_sim_state
from waymax import datatypes
from waymax.metrics import GokartActionNormMetric, GokartActionRateNormMetric, GokartTVActionNormMetric, GokartActionOutRangeMetric


class GokartActionNormMetricTest(tf.test.TestCase, parameterized.TestCase):
    def test_actions(self):
        metric = GokartActionNormMetric()
        state = init_gokart_sim_state(num_timesteps=5)
        
        state.actions_history = state.actions_history.set_actions(
            datatypes.Action(data=jnp.array([0.0, 0.0, 0.0]), valid=jnp.ones((1, 3))), 0
        )
        state.actions_history = state.actions_history.set_actions(
            datatypes.Action(data=jnp.array([0.1, 0.2, 0.3]), valid=jnp.ones((1, 3))), 1
        )
        state.actions_history = state.actions_history.set_actions(
            datatypes.Action(data=jnp.array([-0.9654, 0.676, -0.232]), valid=jnp.ones((1, 3))), 2
        )

        state.timestep = 0
        result = metric.compute(state)
        self.assertEqual(result.value, 0.0)

        state.timestep = 1
        result = metric.compute(state)
        self.assertAllClose(result.value, 0.374165)

        state.timestep = 2
        result = metric.compute(state)
        self.assertAllClose(result.value, 1.201164)

    def test_steering(self):
        metric = GokartActionNormMetric(["steering_angle"])
        state = init_gokart_sim_state(num_timesteps=5)
        
        state.actions_history = state.actions_history.set_actions(
            datatypes.Action(data=jnp.array([0.0, 0.676, -0.232]), valid=jnp.ones((1, 3))), 0
        )
        state.actions_history = state.actions_history.set_actions(
            datatypes.Action(data=jnp.array([0.1, 0.2, 0.3]), valid=jnp.ones((1, 3))), 1
        )
        state.actions_history = state.actions_history.set_actions(
            datatypes.Action(data=jnp.array([-0.9654, 0.676, -0.232]), valid=jnp.ones((1, 3))), 2
        )
        state.actions_history = state.actions_history.set_actions(
            datatypes.Action(data=jnp.array([0.0, -0.843, 0.123]), valid=jnp.ones((1, 3))), 3
        )
        state.actions_history = state.actions_history.set_actions(
            datatypes.Action(data=jnp.array([0.382, 0.39, -0.54]), valid=jnp.ones((1, 3))), 4
        )

        state.timestep = 0
        result = metric.compute(state)
        self.assertEqual(result.value, 0.0)

        state.timestep = 1
        result = metric.compute(state)
        self.assertAllClose(result.value, 0.1)

        state.timestep = 2
        result = metric.compute(state)
        self.assertAllClose(result.value, 0.965399)

    def test_throttle(self):
        metric = GokartActionNormMetric(["acc_left", "acc_right"])
        state = init_gokart_sim_state(num_timesteps=5)
        
        state.actions_history = state.actions_history.set_actions(
            datatypes.Action(data=jnp.array([-0.9654, 0.0, 0.0]), valid=jnp.ones((1, 3))), 0
        )
        state.actions_history = state.actions_history.set_actions(
            datatypes.Action(data=jnp.array([0.1, 0.2, 0.3]), valid=jnp.ones((1, 3))), 1
        )
        state.actions_history = state.actions_history.set_actions(
            datatypes.Action(data=jnp.array([-0.9654, 0.676, -0.232]), valid=jnp.ones((1, 3))), 2
        )
        state.actions_history = state.actions_history.set_actions(
            datatypes.Action(data=jnp.array([0.54, -0.843, 0.123]), valid=jnp.ones((1, 3))), 3
        )
        state.actions_history = state.actions_history.set_actions(
            datatypes.Action(data=jnp.array([-0.28, 0.39, -0.54]), valid=jnp.ones((1, 3))), 4
        )

        state.timestep = 0
        result = metric.compute(state)
        self.assertEqual(result.value, 0.0)

        state.timestep = 1
        result = metric.compute(state)
        self.assertAllClose(result.value, 0.360555)

        state.timestep = 2
        result = metric.compute(state)
        self.assertAllClose(result.value, 0.714702)


class GokartActionRateNormMetricTest(tf.test.TestCase, parameterized.TestCase):

    def test_action_rates(self):
        metric = GokartActionRateNormMetric()
        state = init_gokart_sim_state(num_timesteps=5)
        
        state.actions_history = state.actions_history.set_actions(
            datatypes.Action(data=jnp.array([0.1, 0.2, 0.3]), valid=jnp.ones((1, 3))), 0
        )
        state.actions_history = state.actions_history.set_actions(
            datatypes.Action(data=jnp.array([0.4, -0.6, 0.7]), valid=jnp.ones((1, 3))), 1
        )
        state.actions_history = state.actions_history.set_actions(
            datatypes.Action(data=jnp.array([-0.654, 0.038, -0.283]), valid=jnp.ones((1, 3))), 2
        )
        state.actions_history = state.actions_history.set_actions(
            datatypes.Action(data=jnp.array([-0.103, 0.812, -0.539]), valid=jnp.ones((1, 3))), 3
        )
        state.actions_history = state.actions_history.set_actions(
            datatypes.Action(data=jnp.array([-0.629, 0.123, -0.654]), valid=jnp.ones((1, 3))), 4
        )
        
        state.timestep = 0
        result = metric.compute(state)
        self.assertEqual(result.value, 0.0)

        state.timestep = 1
        result = metric.compute(state)
        self.assertAllClose(result.value, 0.943398)

        state.timestep = 2
        result = metric.compute(state)
        self.assertAllClose(result.value, 1.576150)

        state.timestep = 3
        result = metric.compute(state)
        self.assertAllClose(result.value, 0.983978)

        state.timestep = 4
        result = metric.compute(state)
        self.assertAllClose(result.value, 0.874427)

    def test_steering_rate(self):
        metric = GokartActionRateNormMetric(["steering_angle"])
        state = init_gokart_sim_state(num_timesteps=5)
        
        state.actions_history = state.actions_history.set_actions(
            datatypes.Action(data=jnp.array([0.1, 0.2, 0.3]), valid=jnp.ones((1, 3))), 0
        )
        state.actions_history = state.actions_history.set_actions(
            datatypes.Action(data=jnp.array([0.4, -0.6, 0.7]), valid=jnp.ones((1, 3))), 1
        )
        state.actions_history = state.actions_history.set_actions(
            datatypes.Action(data=jnp.array([-0.654, 0.038, -0.283]), valid=jnp.ones((1, 3))), 2
        )
        state.actions_history = state.actions_history.set_actions(
            datatypes.Action(data=jnp.array([-0.103, 0.812, -0.539]), valid=jnp.ones((1, 3))), 3
        )
        state.actions_history = state.actions_history.set_actions(
            datatypes.Action(data=jnp.array([-0.812, 0.123, -0.654]), valid=jnp.ones((1, 3))), 4
        )

        state.timestep = 0
        result = metric.compute(state)
        self.assertEqual(result.value, 0.0)

        state.timestep = 1
        result = metric.compute(state)
        self.assertAllClose(result.value, 0.3)

        state.timestep = 2
        result = metric.compute(state)
        self.assertAllClose(result.value, 1.054)

        state.timestep = 3
        result = metric.compute(state)
        self.assertAllClose(result.value, 0.551)

    def test_throttle_rate(self):
        metric = GokartActionRateNormMetric(["acc_left", "acc_right"])
        state = init_gokart_sim_state(num_timesteps=5)
        
        state.actions_history = state.actions_history.set_actions(
            datatypes.Action(data=jnp.array([0.1, 0.2, 0.3]), valid=jnp.ones((1, 3))), 0
        )
        state.actions_history = state.actions_history.set_actions(
            datatypes.Action(data=jnp.array([0.4, -0.6, 0.7]), valid=jnp.ones((1, 3))), 1
        )
        state.actions_history = state.actions_history.set_actions(
            datatypes.Action(data=jnp.array([-0.654, 0.038, -0.283]), valid=jnp.ones((1, 3))), 2
        )
        state.actions_history = state.actions_history.set_actions(
            datatypes.Action(data=jnp.array([-0.103, 0.812, -0.539]), valid=jnp.ones((1, 3))), 3
        )
        state.actions_history = state.actions_history.set_actions(
            datatypes.Action(data=jnp.array([-0.812, 0.123, -0.654]), valid=jnp.ones((1, 3))), 4
        )

        state.timestep = 0
        result = metric.compute(state)
        self.assertEqual(result.value, 0.0)

        state.timestep = 1
        result = metric.compute(state)
        self.assertAllClose(result.value, 0.894427)

        state.timestep = 2
        result = metric.compute(state)
        self.assertAllClose(result.value, 1.171892)

        state.timestep = 3
        result = metric.compute(state)
        self.assertAllClose(result.value, 0.815237)


class GokartTVActionNormMetricTest(tf.test.TestCase, parameterized.TestCase):

    def test_tv_action(self):

        metric = GokartTVActionNormMetric()
        state = init_gokart_sim_state(num_timesteps=5)
        
        state.actions_history = state.actions_history.set_actions(
            datatypes.Action(data=jnp.array([-0.9654, 0.0, 0.0]), valid=jnp.ones((1, 3))), 0
        )
        state.actions_history = state.actions_history.set_actions(
            datatypes.Action(data=jnp.array([0.1, 0.2, 0.3]), valid=jnp.ones((1, 3))), 1
        )
        state.actions_history = state.actions_history.set_actions(
            datatypes.Action(data=jnp.array([-0.9654, 0.676, -0.232]), valid=jnp.ones((1, 3))), 2
        )
        state.actions_history = state.actions_history.set_actions(
            datatypes.Action(data=jnp.array([0.54, -0.843, 0.123]), valid=jnp.ones((1, 3))), 3
        )

        state.timestep = 0
        result = metric.compute(state)
        self.assertEqual(result.value, 0.0)

        state.timestep = 1
        result = metric.compute(state)
        self.assertAllClose(result.value, 0.1)

        state.timestep = 2
        result = metric.compute(state)
        self.assertAllClose(result.value, 0.908)

class GokartActionOutRangeMetricTest(tf.test.TestCase, parameterized.TestCase):
    
    def test(self):
        state = init_gokart_sim_state(num_timesteps=5)
        state.actions_history = state.actions_history.set_actions(
            datatypes.Action(data=jnp.array([-1.9654, 0.676, 1.232]), valid=jnp.ones((1, 3))), 2
        )
        
        state.timestep = 2
        
        metric = GokartActionOutRangeMetric("steering_angle")
        result = metric.compute(state)
        self.assertEqual(result.value, 0.0)
        
        metric = GokartActionOutRangeMetric("steering_angle", -1.0, 1.0)
        result = metric.compute(state)
        self.assertEqual(result.value, 1.0)
        
        metric = GokartActionOutRangeMetric("acc_left")
        result = metric.compute(state)
        self.assertEqual(result.value, 0.0)
        
        metric = GokartActionOutRangeMetric("acc_left", max_value=0.5)
        result = metric.compute(state)
        self.assertEqual(result.value, 1.0)
        
        metric = GokartActionOutRangeMetric(["acc_left", "acc_right"], -1.0)
        result = metric.compute(state)
        self.assertEqual(result.value, 0.0)
        
        metric = GokartActionOutRangeMetric(["acc_left", "acc_right"], 0.7)
        result = metric.compute(state)
        self.assertEqual(result.value, 1.0)

if __name__ == "__main__":
    tf.test.main()
