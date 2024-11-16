import jax
from jax import Array, numpy as jnp
from typing import Optional

from gocarx.rl.ppo.structures import TimestepCarry
from waymax.datatypes.gokart_obs import GokartObservation
from waymax.env.abstract_environment import AbstractEnvironment
# from waymax.env.planning_agent_environment import PlanningGoKartSimState
from waymax import datatypes

class EnvWrapper:
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._wrapped_env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._wrapped_env, name)

class DomainRandomizationWrapper(EnvWrapper):
    """
    Brax-like interface wrapper for the Waymax environment.
    Differently from the original implementation, this wrapper supports rng arguments for stepping and resetting.
    """

    def __init__(self, wrapped_env: AbstractEnvironment) -> None:
        """Constructs the Brax-like wrapper over a Waymax environment.

        Args:
          wrapped_env: Waymax-like environment to wrap with the Brax interface
        """
        # domain rando config
        self.sigma_vx = 0.173  # longitudinal vel.
        self.sigma_vy = 0.139  # lateral vel.
        self.sigma_r = 0.044  # angular vel
        self.sigma_yaw = 0.024  # orientation
        self.sigma_xy = 0.160  # position
        super().__init__(wrapped_env)
        
    def observe(self, state: datatypes.SimulatorState, rng: Array) -> GokartObservation:
        """Generate Gaussian noise with JAX
        
        Gaussian properties: consider centered gaussian, i.e. mu = 0, and some sigma defined in
        environment object.
        """
        obs = self._wrapped_env.observe(state)
        
        sigma_dist = self.sigma_xy * jnp.ones(shape=(11,))

        obs.vel_x += jax.random.normal(rng, shape=(1,)) * self.sigma_vx
        obs.vel_y += jax.random.normal(rng, shape=(1,)) * self.sigma_vy
        obs.vel_r += jax.random.normal(rng, shape=(1,)) * self.sigma_r
        obs.dir_diff += jax.random.normal(rng, shape=(1,)) * self.sigma_yaw
        obs.dist_to_edge += jax.random.normal(rng, shape=sigma_dist.shape) * sigma_dist
        return obs
    
    # def reset(self, state: datatypes.SimulatorState, rng: Optional[jax.Array] = None) -> TimestepCarry:
    #     return self._wrapped_env.reset(state, rng)
    
    # def step(self, timestep: TimestepCarry, action: datatypes.Action, rng: jax.Array) -> TimestepCarry:
    #     return self._wrapped_env.step(timestep, action, rng)
    
    # def reward(self, state: datatypes.SimulatorState, action: datatypes.Action) -> jax.Array:
    #     return self._wrapped_env.reward(state, action)