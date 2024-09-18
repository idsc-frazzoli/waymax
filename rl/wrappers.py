import functools
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union


import chex
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
from waymax.env import PlanningGoKartSimState
from waymax.env import GokartRacingEnvironment
from waymax import datatypes

class JaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)
    
@struct.dataclass
class LogEnvState:
    env_state: PlanningGoKartSimState
    episode_returns: float  # Sum of rewards in the current episode
    episode_lengths: int
    returned_episode_returns: float # Sum of rewards in the returned/terminated episode
    returned_episode_lengths: int

class WaymaxLogWrapper(JaxWrapper):
    def __init__(self, env: GokartRacingEnvironment):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, env_state: PlanningGoKartSimState, rng: jax.Array | None = None):
        obs, env_state = self._env.reset(env_state, rng)
        state = LogEnvState(env_state, 0.0, 0, 0.0, 0)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: LogEnvState, action: datatypes.Action):
        # Take a step in the environment
        obs, env_state, reward, done, info = self._env.step(state.env_state, action)

        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] =  done
          
        return obs, state, reward, done, info
