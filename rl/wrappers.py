import functools
from functools import partial
import re
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
    episode_progression_returns: float
    episode_orientation_returns: float  
    episode_offroad_returns: float
    episode_lengths: int
    returned_episode_returns: float # Sum of rewards in the returned/terminated episode
    returned_progression_returns: float
    returned_orientation_returns: float
    returned_offroad_returns: float
    returned_episode_lengths: int

class WaymaxLogWrapper(JaxWrapper):
    def __init__(self, env: GokartRacingEnvironment):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, env_state: PlanningGoKartSimState, rng: jax.Array | None = None):
        obs, env_state = self._env.reset(env_state, rng)
        state = LogEnvState(env_state, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0,0.0, 0.0, 0)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: LogEnvState, action: datatypes.Action):
        # Take a step in the environment
        obs, env_state, reward, done, info = self._env.step(state.env_state, action)

        new_episode_return = state.episode_returns + reward
        new_episode_progression_return = state.episode_progression_returns + info["progression_reward"]
        new_episode_orientation_return = state.episode_orientation_returns + info["orientation_reward"]
        new_episode_offroad_return = state.episode_offroad_returns + info["offroad_reward"]
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_progression_returns=new_episode_progression_return * (1 - done),
            episode_orientation_returns=new_episode_orientation_return * (1 - done),
            episode_offroad_returns=new_episode_offroad_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_progression_returns=state.returned_progression_returns * (1 - done)
            + new_episode_progression_return * done,
            returned_orientation_returns=state.returned_orientation_returns * (1 - done)
            + new_episode_orientation_return * done,
            returned_offroad_returns=state.returned_offroad_returns * (1 - done)
            + new_episode_offroad_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_progression_returns"] = state.returned_progression_returns
        info["returned_orientation_returns"] = state.returned_orientation_returns
        info["returned_offroad_returns"] = state.returned_offroad_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] =  done
          
        return obs, state, reward, done, info
    
@struct.dataclass
class NormalizeVecObsEnvState:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    env_state: PlanningGoKartSimState


class NormalizeVecObservation(JaxWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        state = NormalizeVecObsEnvState(
            mean=jnp.zeros_like(obs),
            var=jnp.ones_like(obs),
            count=1e-4,
            env_state=state,
        )
        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecObsEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            env_state=state.env_state,
        )

        return (obs - state.mean) / jnp.sqrt(state.var + 1e-8), state

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )

        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecObsEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            env_state=env_state,
        )
        return (
            (obs - state.mean) / jnp.sqrt(state.var + 1e-8),
            state,
            reward,
            done,
            info,
        )


@struct.dataclass
class NormalizeVecRewEnvState:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    return_val: float
    env_state: PlanningGoKartSimState


class NormalizeVecReward(JaxWrapper):
    def __init__(self, env, gamma):
        super().__init__(env)
        self.gamma = gamma

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        batch_count = obs.shape[0]
        state = NormalizeVecRewEnvState(
            mean=0.0,
            var=1.0,
            count=1e-4,
            return_val=jnp.zeros((batch_count,)),
            env_state=state,
        )
        return obs, state

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        return_val = state.return_val * self.gamma * (1 - done) + reward

        batch_mean = jnp.mean(return_val, axis=0)
        batch_var = jnp.var(return_val, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecRewEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            return_val=return_val,
            env_state=env_state,
        )
        return obs, state, reward / jnp.sqrt(state.var + 1e-8), done, info
