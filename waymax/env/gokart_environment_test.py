import dataclasses

import jax.numpy as jnp

from waymax.config import LinearCombinationRewardConfig, EnvironmentConfig
from waymax.dynamics.tricycle_model import TricycleModel
from waymax.env.gokart_environment import calculate_distances_to_boundary, GokartRacingEnvironment
from waymax.utils.gokart_config import GoKartGeometry, PajieckaParams, TricycleParams, TrackControlPoints
from waymax.utils.gokart_utils import generate_racing_track

car_pos = jnp.array([30.626804, 20.0801])
car_orientation = jnp.array(-0.10766554)  # jnp.pi/4 # radians
num_rays = 8  # Number of rays to cast
max_distance = 0.1  # Maximum perpendicular distance to consider for filtering points
# for new version of generate_racing_track
track_cntrl_points = TrackControlPoints()
roadgraph_points, x_center, y_center, cumulative_length = generate_racing_track(
        track_cntrl_points.x,
        track_cntrl_points.y,
        track_cntrl_points.r)
edge_points = roadgraph_points.xy[..., 2000:, :]


def test_calculate_dist_to_boundaries():
    res = calculate_distances_to_boundary(
            car_pos, car_orientation, edge_points, num_rays, max_distance)
    # todo something with res


def test_gokart_env():
    dynamics_model = TricycleModel(
            gk_geometry=GoKartGeometry(),
            model_params=TricycleParams(),
            paj_params=PajieckaParams(),
            dt=0.1,
            normalize_actions=True,
    )
    rewards = LinearCombinationRewardConfig(
            rewards={"offroad": -1.0, "progress": -1.0, "log_divergence": -1.0}
    )
    env_config = dataclasses.replace(EnvironmentConfig(),
                                     rewards=rewards,
                                     max_num_objects=1,
                                     init_steps=1)
    env = GokartRacingEnvironment(
            dynamics_model=dynamics_model,
            config=env_config
    )

    # todo something with env
