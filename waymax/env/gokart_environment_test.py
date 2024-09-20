from env.gokart_environment import calculate_distances_to_boundary

from jax import numpy as jnp

from utils.gokart_config import TrackControlPoints
from utils.gokart_utils import generate_racing_track

car_pos = jnp.array([30.626804, 20.0801])
car_orientation = -0.10766554  #jnp.pi/4 # radians
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
    distances, hit_points, debug_values = calculate_distances_to_boundary(
        car_pos, car_orientation, edge_points, num_rays, max_distance)
