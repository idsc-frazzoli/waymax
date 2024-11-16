import dataclasses

import jax
import jax.numpy as jnp

from gocarx.rl.ppo.wrappers import BraxLikeWrapper


from waymax.config import LinearCombinationRewardConfig, EnvironmentConfig
from waymax.dynamics.tricycle_model import TricycleModel
from waymax.env.gokart_environment import GokartRacingEnvironment
from waymax.datatypes.gokart_obs import GokartObservation
from waymax.utils.gokart_config import GoKartGeometry, PajieckaParams, TricycleParams
from waymax.env.wrappers.domain_randomization_wrapper import DomainRandomizationWrapper
from waymax.utils.gokart_utils import create_init_state

def test_dr_wrapper():
    rng = jax.random.PRNGKey(1)
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
    env = DomainRandomizationWrapper(env)
    env = BraxLikeWrapper(env)
    
    state = create_init_state()
    obs = GokartObservation(
        vel_x=jnp.zeros(shape=(100,1)),  # doing this for shape
        vel_y=jnp.zeros(shape=(100,1)),
        vel_r=jnp.zeros(shape=(100,1)),
        dir_diff=jnp.zeros(shape=(100,1)),
        dist_to_edge=jnp.zeros(shape=(100,11)),
    )
    
    obs_original = obs.flatten().copy()
    obs_noisy = env.observe(state, rng)
    
    # Ensure the object is of type MyClass
    assert isinstance(obs_noisy, GokartObservation), "Object is not of type GokartObservation"
    # Ensure the noisy observation is not the same as the original observation
    assert not jnp.array_equal(obs_original, obs_noisy.flatten()), "Observations are the same, but they should not be."

if __name__ == "__main__":
    test_dr_wrapper()