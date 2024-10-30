import dataclasses
from pprint import pprint

from jax import numpy as jnp
import tensorflow as tf

from absl.testing import parameterized

from waymax import config as _config, datatypes
from waymax.dynamics.tricycle_model import TricycleModel
from waymax.env import GokartRacingEnvironment
from waymax.metrics.gokart_metric import GokartProgressMetric, GokartOrientationMetric
from waymax.utils.gokart_config import GoKartGeometry, PajieckaParams, TricycleParams
from waymax.utils.gokart_utils import create_init_state

class GokartProgressMetricTest(tf.test.TestCase, parameterized.TestCase):
    def test_progress_without_stepping(self):
        metric = GokartProgressMetric()
        state = create_init_state(num_timesteps=100)
        result = metric.compute(state)
        self.assertEqual(result.value, 0.0)


    def test_progress(self):
        metric = GokartProgressMetric()
        state = create_init_state(num_timesteps=100)
        dynamics_model = TricycleModel(gk_geometry=GoKartGeometry(), model_params=TricycleParams(),
                                        paj_params=PajieckaParams(), dt=0.1, normalize_actions=True, )

        env = GokartRacingEnvironment(
                dynamics_model=dynamics_model,
                config=dataclasses.replace(
                        _config.EnvironmentConfig(),
                        max_num_objects=1,
                        init_steps=1  # => state.timestep = 0
                ),
        )
        # steering, left acceleration, right acceleration
        raw_action = jnp.array([0.0, 0.1, 0.1])
        action = datatypes.Action(data=raw_action, valid=jnp.array([True]))

        _, state = env.reset(state)
        # set the initial velocity to 2
        state.sim_trajectory.vel_x = state.sim_trajectory.vel_x.at[..., 0, 0].set(2)
        _, new_state, _, _, _ = env.step(state, action=action)
        result = metric.compute(new_state)
        self.assertGreaterEqual(result.value, 0.0)

class GokartOrientationMetricTest(tf.test.TestCase, parameterized.TestCase):
    def test_zero_velocity(self):
        metric = GokartOrientationMetric()
        state = create_init_state(num_timesteps=100)
        result = metric.compute(state)
        # should be zero, because the velocity is zero
        self.assertEqual(result.value, 0.0)
    
    def test_correct_orientation(self):
        metric = GokartOrientationMetric()
        state = create_init_state(num_timesteps=100)
        # set a velocity, so that the orientation reward is not zero
        state.sim_trajectory.vel_x = state.sim_trajectory.vel_x.at[..., 0, 0].set(1)
        result = metric.compute(state)
        self.assertGreater(result.value, 0.0)

    def test_negative_velocity(self):
        metric = GokartOrientationMetric()
        state = create_init_state(num_timesteps=100)
        # set a velocity, so that the orientation reward is not zero
        state.sim_trajectory.vel_x = state.sim_trajectory.vel_x.at[..., 0, 0].set(-1)
        result = metric.compute(state)
        self.assertLess(result.value, 0.0)

    def test_wrong_orientation(self):
        metric = GokartOrientationMetric()
        state = create_init_state(num_timesteps=100)
        state.sim_trajectory.vel_x = state.sim_trajectory.vel_x.at[..., 0, 0].set(1)
        # shape: (..., num_objects, timesteps=1) -> (..., num_objects)
        yaw = state.current_sim_trajectory.yaw[..., 0]

        sdc_yaw_curr = datatypes.select_by_onehot(
                yaw,
                state.object_metadata.is_sdc,
                keepdims=False,
        )
        wrong_orientation = sdc_yaw_curr + jnp.pi
        print(f"wrong_orientation: {wrong_orientation}")
        state.sim_trajectory.yaw = state.sim_trajectory.yaw.at[..., 0, 0].set(wrong_orientation)
        result = metric.compute(state)
        self.assertLess(result.value, 0.0)

if __name__ == '__main__':
  tf.test.main()