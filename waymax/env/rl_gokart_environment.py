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
      distance_to_edge = calculate_edge_distances(sdc_xy_curr, sdc_yaw_curr, edge_points)
    else:
      distance_to_edge = jax.vmap(calculate_edge_distances, in_axes=(0, 0, 0))(sdc_xy_curr, sdc_yaw_curr, edge_points)
      
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
  

def point_to_line_distance(point, line_point, direction):
    """
    Calculate the perpendicular distance from a point to a line.
    Args:
      point: Array of (x, y) coordinates of the point.
      line_point: Array of (x, y) coordinates of a point on the line.
      direction: Array of (dx, dy) representing the direction of the line.
    Return: 
      Tuple of perpendicular distance and projection length from the point to the line.
    """
    # Vector from the line point to the point
    vector = point - line_point
    
    # Project the vector onto the direction
    projection_length = jnp.dot(vector, direction) / jnp.linalg.norm(direction)
    projection = projection_length * direction / jnp.linalg.norm(direction)
    
    # Calculate the perpendicular vector
    perpendicular = vector - projection
    
    # Return the norm of the perpendicular vector (distance) and the projection length
    return jnp.linalg.norm(perpendicular), projection_length

def calculate_edge_distances(car_position, car_orientation, edge_points, num_rays: int = 8, max_distance: float=0.5):
    """
    Calculate approximate distances from the car to the edge points in specified directions.
    
    Args:
      car_position: Tuple of (x, y) coordinates of the car.
      car_orientation: Angle in radians representing the car's orientation.
      edge_points: Array of shape (n, 2) representing the edge points.
      angles: List of angles (in radians) relative to the car's orientation to cast rays.
      max_distance: Maximum distance to consider for filtering points.
    Return: 
      List of approximate distances from the car to the edges along the specified angles.
    """
    car_pos = jnp.array(car_position)
    edge_points = jnp.array(edge_points)

    # Generate angles based on the number of rays
    angles = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, num=num_rays)
    
    def distance_for_angle(angle):
        direction = jnp.array([jnp.cos(car_orientation + angle), jnp.sin(car_orientation + angle)])
        
        # Vectorize the point_to_line_distance function
        vectorized_point_to_line_distance = jax.vmap(point_to_line_distance, in_axes=(0, None, None))
        perpendicular_distances, projection_lengths = vectorized_point_to_line_distance(edge_points, car_pos, direction)
        
        # Filter points within the max distance and in front of the car
        valid_indices = (perpendicular_distances <= max_distance) & (projection_lengths >= 0)
        
        if not jnp.any(valid_indices):
            return jnp.inf
        
        valid_points = edge_points[valid_indices]
        
        # Calculate distances from the car to the valid edge points
        valid_distances = jnp.linalg.norm(valid_points - car_pos, axis=1)
        
        return jnp.min(valid_distances)
    
    # Vectorize the distance calculation for all angles
    distance_for_angles = jax.vmap(distance_for_angle)
    distances = distance_for_angles(angles)
    
    # Convert distances back to a Python list and replace jnp.inf with None
    distances = [None if d == jnp.inf else d for d in distances]
    
    return distances