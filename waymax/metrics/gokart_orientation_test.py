import tensorflow as tf
from absl.testing import parameterized
from jax import numpy as jnp

from gocarx.utils.gokart_utils import init_gokart_sim_state
from waymax import datatypes
from waymax.metrics import GokartOrientationMetric


class GokartOrientationMetricTest(tf.test.TestCase, parameterized.TestCase):
    def test_zero_velocity(self):
        metric = GokartOrientationMetric()
        state = init_gokart_sim_state(num_timesteps=100)
        result = metric.compute(state)
        # should be zero, because the velocity is zero
        self.assertEqual(result.value, 0.0)
    
    def test_correct_orientation(self):
        metric = GokartOrientationMetric()
        state = init_gokart_sim_state(num_timesteps=100)
        # set a velocity, so that the orientation reward is not zero
        state.sim_trajectory.vel_x = state.sim_trajectory.vel_x.at[..., 0, 0].set(1)
        result = metric.compute(state)
        self.assertGreater(result.value, 0.0)

    def test_negative_velocity(self):
        metric = GokartOrientationMetric()
        state = init_gokart_sim_state(num_timesteps=100)
        # set a velocity, so that the orientation reward is not zero
        state.sim_trajectory.vel_x = state.sim_trajectory.vel_x.at[..., 0, 0].set(-1)
        result = metric.compute(state)
        self.assertLess(result.value, 0.0)

    def test_wrong_orientation(self):
        metric = GokartOrientationMetric()
        state = init_gokart_sim_state(num_timesteps=100)
        state.sim_trajectory.vel_x = state.sim_trajectory.vel_x.at[..., 0, 0].set(-1)
        # shape: (..., num_objects, timesteps=1) -> (..., num_objects)
        yaw = state.current_sim_trajectory.yaw[..., 0]

        sdc_yaw_curr = datatypes.select_by_onehot(
                yaw,
                state.object_metadata.is_sdc,
                keepdims=False,
        )
        wrong_orientation = sdc_yaw_curr + jnp.pi
        state.sim_trajectory.yaw = state.sim_trajectory.yaw.at[..., 0, 0].set(wrong_orientation)
        result = metric.compute(state)
        self.assertEqual(result.value, 0.0)


