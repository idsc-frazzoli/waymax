import dataclasses
from pprint import pprint

from jax import numpy as jnp

from waymax import config as _config, datatypes
from waymax.dynamics.tricycle_model import TricycleModel
from waymax.env import GokartRacingEnvironment
from waymax.metrics.gokart_progress import GokartProgressMetric
from waymax.utils.gokart_config import GoKartGeometry, PajieckaParams, TricycleParams
from waymax.utils.gokart_utils import create_init_state


def test_progress_without_stepping():
    metric = GokartProgressMetric()
    state = create_init_state(num_timesteps=100)
    res = metric.compute(state)
    pprint(res)


def test_progress():
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
    raw_action = jnp.array([0.0, 0.1, 0.1, ])
    action = datatypes.Action(data=raw_action, valid=jnp.array([True]))

    new_state = env.step(state, action=action)
    res = metric.compute(new_state)
    pprint(res)
