# Copyright 2023 The Waymax Authors.
#
# Licensed under the Waymax License Agreement for Non-commercial Use
# Use (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#     https://github.com/waymo-research/waymax/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Waymax environment for tasks relating to Planning for the ADV."""

from typing import Sequence

import chex
from dm_env import specs
import jax
import jax.numpy as jnp
from waymax import config as _config
from waymax import datatypes
from waymax import dynamics as _dynamics
from waymax import metrics
from waymax import rewards
from waymax.agents import actor_core
from waymax.env import abstract_environment
from waymax.env import base_environment as _env
from waymax.env import typedefs as types
from waymax.env import PlanningAgentEnvironment, PlanningAgentSimulatorState
from waymax.utils import geometry


class GokartRacingEnvironment(PlanningAgentEnvironment):

  # TODO: Move to the new sim agent interface when available.
  def __init__(
      self,
      dynamics_model: _dynamics.DynamicsModel,
      config: _config.EnvironmentConfig,
      sim_agent_actors: Sequence[actor_core.WaymaxActorCore] = (),
      sim_agent_params: Sequence[actor_core.Params] = (),
  ) -> None:
    super().__init__(dynamics_model, config, sim_agent_actors, sim_agent_params)

  def observe(self, state: PlanningAgentSimulatorState) -> types.Observation:
    """Computes the observation for the given simulation state.

    Here we assume that the default observation is just the simulator state. We
    leave this for the user to override in order to provide a user-specific
    observation function. A user can use this to move some of their model
    specific post-processing into the environment rollout in the actor nodes. If
    they want this post-processing on the accelerator, they can keep this the
    same and implement it on the learner side. We provide some helper functions
    at datatypes.observation.py to help write your own observation functions.

    Args:
      state: Current state of the simulator of shape (...).

    Returns:
      Simulator state as an observation without modifications of shape (...).
    """
    # shape: (..., num_objects, timesteps=1, 2) -> (..., num_objects, 2)
    pos_xy = state.current_sim_trajectory.xy[..., 0, :]

    # shape: (...,2)
    sdc_xy_curr = datatypes.select_by_onehot(
        pos_xy,
        state.object_metadata.is_sdc,
        keepdims=False,
    )

    # shape: (..., num_objects, timesteps=1) -> (..., num_objects)
    yaw = state.current_sim_trajectory.yaw[..., 0]

    sdc_yaw_curr = datatypes.select_by_onehot(
        yaw,
        state.object_metadata.is_sdc,
        keepdims=False,
    )

    # Shape: (..., num_objects, num_timesteps=1)
    obj_valid_curr = datatypes.dynamic_slice(
        state.sim_trajectory.valid,
        state.timestep,
        1,
        axis=-1,
    )
    # Shape: (...)
    sdc_valid_curr = datatypes.select_by_onehot(
        obj_valid_curr[..., 0],
        state.object_metadata.is_sdc,
        keepdims=False,
    )

    is_road_edge = datatypes.is_road_edge(state.roadgraph_points.types)
    edge_points = state.roadgraph_points.xy[is_road_edge & state.roadgraph_points.valid]
    if len(sdc_xy_curr.shape) == 1: # no batch dimension
      distance_to_edge = calculate_distances_to_boundary(sdc_xy_curr, sdc_yaw_curr, edge_points)
    else:
      distance_to_edge = jax.vmap(calculate_distances_to_boundary, in_axes=(0, 0, 0))(sdc_xy_curr, sdc_yaw_curr, edge_points)
      
    return state
  def check_termination(self, state: PlanningAgentSimulatorState) -> jnp.ndarray:
    """Checks if the episode should terminate.

    Args:
      state: Current state of the simulator of shape (...).

    Returns:
      A boolean array of shape (...) indicating if the episode should terminate.
    """
    metric_dict = self.metrics(state)
    is_offroad = metric_dict["offroad"].squeeze(-1)
    if is_offroad == 1.0:
       self.reset(state)
    # return jnp.zeros(state.shape, dtype=jnp.bool_)
  

def calculate_distances_to_boundary(car_position, car_yaw, boundary_points, num_rays=8, max_distance=0.15):
    """
    calculate distances to boundary in different directions
    
    Args:
    car_position: car position (x, y)  shape (2,)
    car_yaw: car orientation in radians
    boundary_points: boundary points of the track    shape (N, 2)
    num_rays: number of rays to cast
    
    Returns:
    distances: distance to boundary in different directions    shape (num_rays,)
    hit_points: points of intersections of rays and boundary    shape (num_rays, 2)
    """

    angles = jnp.linspace(-jnp.pi/2, jnp.pi/2, num_rays)

    ray_directions = jnp.array([jnp.cos(car_yaw + angles), jnp.sin(car_yaw + angles)]) # (2, num_rays)

    relative_positions = boundary_points - car_position # (N, 2)
    projections = jnp.dot(relative_positions, ray_directions) # (N, num_rays)
    distances_to_points = jnp.linalg.norm(relative_positions, axis=1, keepdims=True)  # (N, 1)
    perpendicular_distances = jnp.sqrt(distances_to_points**2 - projections**2)  # (N, num_rays)
    
    valid_mask = (projections > 0) & (perpendicular_distances < max_distance)  
    valid_projections = jnp.where(valid_mask, projections, jnp.inf)  # (N, num_rays)

    distances = jnp.min(valid_projections, axis=0) # (num_rays,)
    hit_points = car_position + ray_directions.T * distances[:, None] # (num_rays, 2)
    
    return distances, hit_points