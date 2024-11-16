import dataclasses

import jax
import jax.numpy as jnp

from gocarx.rl.ppo.wrappers import BraxLikeWrapper

from waymax.config import LinearCombinationRewardConfig, EnvironmentConfig
from waymax.dynamics.tricycle_model import TricycleModel
from waymax.env.gokart_environment import GokartRacingEnvironment
from waymax.datatypes.gokart_obs import GokartObservation
from waymax.utils.gokart_config import GoKartGeometry, PajieckaParams, TricycleParams

def test_observation_flatten():
    dynamics_model = TricycleModel(
            gk_geometry=GoKartGeometry(),
            model_params=TricycleParams(),
            paj_params=PajieckaParams(),
            dt=0.1,
            normalize_actions=True,
    )
    rewards = LinearCombinationRewardConfig(
        rewards={"gokart_offroad": -1, "gokart_progress": 0.5, "gokart_orientation": 0.1}
    )
    env_config = dataclasses.replace(
            EnvironmentConfig(), rewards=rewards, max_num_objects=1, init_steps=1
    )
    
    env = GokartRacingEnvironment(dynamics_model=dynamics_model,config=env_config)
    env = BraxLikeWrapper(env)
    
    obs = GokartObservation(
        vel_x=jnp.zeros(shape=(100,1)),  # doing this for shape
        vel_y=jnp.zeros(shape=(100,1)),
        vel_r=jnp.zeros(shape=(100,1)),
        dir_diff=jnp.zeros(shape=(100,1)),
        dist_to_edge=jnp.zeros(shape=(100,11)),
    )
    
    obs_flatten = obs.flatten()
    
    # Ensure the noisy observation is not the same as the original observation
    assert isinstance(obs_flatten, jax.Array), f"Expected a jax.Array, but got {type(obs_flatten)}"

if __name__ == "__main__":
    test_observation_flatten()