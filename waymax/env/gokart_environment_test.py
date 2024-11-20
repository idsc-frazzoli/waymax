import dataclasses
from pprint import pprint

import jax.numpy as jnp

from waymax.config import LinearCombinationRewardConfig, EnvironmentConfig, MetricsConfig
from waymax.dynamics.tricycle_model import TricycleModel
from waymax.env.gokart_environment import calculate_distances_to_boundary, GokartRacingEnvironment
from waymax.utils.gokart_config import GoKartGeometry, PajieckaParams, TricycleParams, TrackControlPoints
from waymax.utils.gokart_utils import generate_racing_track, create_init_state


def test_calculate_dist_to_boundaries():
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
    metrics_config = MetricsConfig(metrics_to_run=("gokart_offroad", "gokart_progress", "gokart_orientation"))
    rewards = LinearCombinationRewardConfig(
            rewards={"gokart_offroad": -1, "gokart_progress": 0.5, "gokart_orientation": 0.1}
    )
    env_config = dataclasses.replace(
            EnvironmentConfig(), metrics=metrics_config, rewards=rewards, max_num_objects=1, init_steps=1
    )
    env = GokartRacingEnvironment(
            dynamics_model=dynamics_model,
            config=env_config
    )
    state = create_init_state()
    metrics_dict = env.metrics(state)
    pprint(metrics_dict)
    # reward = env.reward(state, jnp.array([0.1, 0.2, 0.3]))
    # pprint(reward)
    # todo something with env
