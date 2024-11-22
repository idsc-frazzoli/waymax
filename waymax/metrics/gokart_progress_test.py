import dataclasses
from pprint import pprint

from jax import numpy as jnp
import tensorflow as tf

from absl.testing import parameterized

from gocarx.metrics.gokart_progress import GokartProgressMetric
from waymax import config as _config, datatypes
from gocarx.dynamics.tricycle_model import TricycleModel
from gocarx.env import GokartRacingEnvironment
from gocarx.dynamics.gokart_config import GoKartGeometry, PajieckaParams, TricycleParams
from gocarx.utils.gokart_utils import create_init_state

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

        state = env.reset(state)
        # set the initial velocity to 2
        state.sim_trajectory.vel_x = state.sim_trajectory.vel_x.at[..., 0, 0].set(2)
        new_state = env.step(state, action=action)
        result = metric.compute(new_state)
        self.assertGreaterEqual(result.value, 0.0)

    def test_progress_in_wrong_direction(self):
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

        state = env.reset(state)
        # set the initial velocity to -2, so the car is moving backwards
        state.sim_trajectory.vel_x = state.sim_trajectory.vel_x.at[..., 0, 0].set(-2)
        new_state = env.step(state, action=action)
        result = metric.compute(new_state)
        self.assertEqual(result.value, 0.0)

    def test_progress_when_completing_lap(self):
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

        state = env.reset(state)
        state.sim_trajectory.vel_x = state.sim_trajectory.vel_x.at[..., 0, 0].set(6)
        current_x = state.current_sim_trajectory.x[..., 0, 0]
        current_x -= 0.3 # a little before the end of the lap
        state.sim_trajectory.x = state.sim_trajectory.x.at[..., 0, 0].set(current_x)
        new_state = env.step(state, action=action)
        result = metric.compute(new_state)
        self.assertGreater(result.value, 0.5)
