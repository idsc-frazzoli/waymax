from typing import Optional

import jax
from jax import Array, numpy as jnp
from flax import struct

from gocarx.rl.ppo.structures import TimestepCarry
from waymax.datatypes.gokart_obs import GokartObservation
from waymax.env.abstract_environment import AbstractEnvironment
from waymax.env.planning_agent_environment import PlanningGoKartSimState
from waymax import datatypes

class EnvWrapper:
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._wrapped_env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._wrapped_env, name)
    
@struct.dataclass
class NormalizeGokartObsEnvState:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    # env_state: datatypes.SimulatorState   # PlanningGoKartSimState???
    
class NormalizeGokartObservationWrapper(EnvWrapper):
    
    def __init__(self, wrapped_env):
        """Constructs the Brax-like wrapper over a Waymax environment.

        Args:
          wrapped_env: Waymax-like environment to wrap with the Brax interface
        """
        super().__init__(wrapped_env)

    def observe(self, state: datatypes.SimulatorState, rng: Array) -> GokartObservation:
        obs = self._wrapped_env.observe(state, rng)
        
        norm_state = NormalizeGokartObsEnvState(
            mean=jnp.zeros_like(obs.flatten()),
            var=jnp.ones_like(obs.flatten()),
            count=1e-4,
            # env_state=state,  # PlanningGoKartSimState???
        )

        batch_mean = jnp.mean(obs.flatten(), axis=0)
        batch_var = jnp.var(obs.flatten(), axis=0)
        batch_count = obs.flatten().shape[0]

        delta = batch_mean - norm_state.mean
        tot_count = norm_state.count + batch_count

        new_mean = norm_state.mean + delta * batch_count / tot_count
        m_a = norm_state.var * norm_state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * norm_state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        norm_state = NormalizeGokartObsEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            # env_state=state,
        )
        
        obs_norm_flattened = (obs.flatten() - norm_state.mean) / jnp.sqrt(norm_state.var + 1e-8)
        
        # fixme like this it stays invariant to changes of obs dim in
        # distance to edge, but not anything else
        obs_norm = GokartObservation(
            vel_x=jnp.array([obs_norm_flattened[0]]),  # doing this for shape
            vel_y=jnp.array([obs_norm_flattened[1]]),
            vel_r=jnp.array([obs_norm_flattened[2]]),
            dir_diff=jnp.array([obs_norm_flattened[3]]),
            dist_to_edge=obs_norm_flattened[4:],
        )

        return obs_norm