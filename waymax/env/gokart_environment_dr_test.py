import dataclasses

import jax
import jax.numpy as jnp

from waymax.config import LinearCombinationRewardConfig, EnvironmentConfig
from waymax.dynamics.tricycle_model import TricycleModel
from waymax.env.gokart_environment_dr import calculate_distances_to_boundary, GokartRacingDREnvironment
from waymax.utils.gokart_config import GoKartGeometry, PajieckaParams, TricycleParams, TrackControlPoints
from waymax.utils.gokart_utils import generate_racing_track
from waymax.datatypes.gokart_obs import GokartObservation

car_pos = jnp.array([30.626804, 20.0801])
car_orientation = jnp.array(-0.10766554)  # jnp.pi/4 # radians
num_rays = 8  # Number of rays to cast
max_distance = 0.1  # Maximum perpendicular distance to consider for filtering points
# for new version of generate_racing_track
track_cntrl_points = TrackControlPoints()
roadgraph_points, x_center, y_center, cumulative_length = generate_racing_track(
    track_cntrl_points.x, track_cntrl_points.y, track_cntrl_points.r
)
edge_points = roadgraph_points.xy[..., 2000:, :]


def test_calculate_dist_to_boundaries():
    res = calculate_distances_to_boundary(car_pos, car_orientation, edge_points, num_rays, max_distance)
    # todo something with res


def test_gokart_env():
    dynamics_model = TricycleModel(
        gk_geometry=GoKartGeometry(),
        model_params=TricycleParams(),
        paj_params=PajieckaParams(),
        dt=0.1,
        normalize_actions=True,
    )
    rewards = LinearCombinationRewardConfig(rewards={"offroad": -1.0, "progress": -1.0, "log_divergence": -1.0})
    env_config = dataclasses.replace(EnvironmentConfig(), rewards=rewards, max_num_objects=1, init_steps=1)
    env = GokartRacingDREnvironment(dynamics_model=dynamics_model, config=env_config)
    env

    # todo something with env
    
    
def test_domain_rando():
    dynamics_model = TricycleModel(
        gk_geometry=GoKartGeometry(),
        model_params=TricycleParams(),
        paj_params=PajieckaParams(),
        dt=0.1,
        normalize_actions=True,
    )
    rewards = LinearCombinationRewardConfig(rewards={"gokart_offroad": -1.0, "gokart_progress": -1.0})
    env_config = dataclasses.replace(EnvironmentConfig(), rewards=rewards, max_num_objects=1, init_steps=1)
    env = GokartRacingDREnvironment(dynamics_model=dynamics_model, config=env_config)
    rng = jax.random.PRNGKey(1)
    
    obs = GokartObservation(
        vel_x=jnp.zeros(shape=(100,1)),  # doing this for shape
        vel_y=jnp.zeros(shape=(100,1)),
        vel_r=jnp.zeros(shape=(100,1)),
        dir_diff=jnp.zeros(shape=(100,1)),
        dist_to_edge=jnp.zeros(shape=(100,11)),
    )
    obs_original = obs.flatten().copy()
    obs_noisy = env.apply_domain_rando(obs, rng)
    
    # Ensure the object is of type MyClass
    assert isinstance(obs_noisy, GokartObservation), "Object is not of type GokartObservation"
    # Ensure the noisy observation is not the same as the original observation
    assert not jnp.array_equal(obs_original, obs_noisy.flatten()), "Observations are the same, but they should not be."

if __name__ == '__main__':
    test_domain_rando()
    