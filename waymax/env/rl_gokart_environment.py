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

import copy
from re import T
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
from waymax.env import PlanningAgentEnvironment
from waymax.utils import geometry
import dataclasses
from functools import partial


@chex.dataclass
class PlanningGoKartSimState(datatypes.GoKartSimState):
  """Simulator state for the planning agent environment.

  Attributes:
    sim_agent_actor_states: State of the sim agents that are being run inside of
      the environment `step` function. If sim agents state is provided, this
      will be updated. The list of sim agent states should be as long as and in
      the same order as the number of sim agents run in the environment.
  """

  sim_agent_actor_states: Sequence[actor_core.ActorState] = ()

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
    self.metrics_config = dataclasses.replace(_config.MetricsConfig(),metrics_to_run=("offroad", "sdc_progression"))
    self.reward_config = _config.LinearCombinationRewardConfig(rewards={'offroad': -1.0, 'sdc_progression': 10.0})
    self.reward_fn = rewards.LinearCombinationReward(self.reward_config)

  def observe(self, state: PlanningGoKartSimState) -> types.Observation:
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
      sdc_xy_curr: cuurent position of the self-driving car in the global coordinate system
      sdc_vel_curr: current velocity of the self-driving car in the go-kart coordinate system
      dir_diff: difference between the current orientation of the self-driving car and the orientation of the nearest point on the track
      distance_to_edge: distance to the track boundary in different directions
    """
    # shape: (..., num_objects, timesteps=1, 2) -> (..., num_objects, 2)
    pos_xy = state.current_sim_trajectory.xy[..., 0, :]
    vel_xy = state.current_sim_trajectory.vel_xy[..., 0, :]

    # shape: (...,2)
    sdc_xy_curr = datatypes.select_by_onehot(
        pos_xy,
        state.object_metadata.is_sdc,
        keepdims=False,
    )
    sdc_vel_curr = datatypes.select_by_onehot(
        vel_xy,
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

    # Distance from the current sdc position to all the points on sdc_paths (here centerline)
    # Shape: (..., num_paths, num_points_per_path) our case: num_paths=1
    dist_raw = jnp.linalg.norm(
        state.sdc_paths.xy - jnp.expand_dims(sdc_xy_curr, axis=(-2, -3)),
        axis=-1,
        keepdims=False,
    )
    # Only consider valid on-route paths.
    dist = jnp.where(state.sdc_paths.valid & state.sdc_paths.on_route, dist_raw, jnp.inf)
    # Only consider valid SDC states. # shape: (..., num_paths, num_points_per_path)
    dist = jnp.where(
        jnp.expand_dims(sdc_valid_curr, axis=(-1, -2)), dist, jnp.inf
    )
    
    idx = jnp.argmin(dist, axis=-1, keepdims=True)  # (..., num_paths=1, 1)

    # use the direction of the nearest sdc_path point as referece direction
    dir_ref = jnp.take_along_axis(state.sdc_paths.dir_xy, idx[..., None], axis=-2)  # (..., num_paths=1, 1, 2)
    dir_ref = jnp.squeeze(dir_ref, axis=(-2,-3))  # (...,2)

    # dir_ref = jnp.arctan2(dir_ref[:, 1], dir_ref[:, 0])  # (...,)
    dir_ref = jnp.arctan2(dir_ref[1], dir_ref[0])  # (...,)
    dir_diff = sdc_yaw_curr - dir_ref  # (...,)
    dir_diff = dir_diff[..., None] # (..., 1)

    # is_road_edge = datatypes.is_road_edge(state.roadgraph_points.types)
    # edge_points = state.roadgraph_points.xy[is_road_edge & state.roadgraph_points.valid] # not allowed for jit, dynamic indexing !!!
    # edge_points = get_edge_points(state.roadgraph_points.xy, state.roadgraph_points.types)
    # indices = jnp.where(is_road_edge)[0]   # jnp.where with single argument not compatible with jit!!!
    # edge_points = state.roadgraph_points.xy[indices, :]
    # indices = jnp.where(is_road_edge, size=state.roadgraph_points.types.size, fill_value=-1)[0]

    edge_points = state.roadgraph_points.xy[1000:,:] # for testing, need to find a better way to get the edge points
    if len(sdc_xy_curr.shape) == 1: # no batch dimension
      # edge_points = state.roadgraph_points.xy[is_road_edge]
      distance_to_edge, _ = calculate_distances_to_boundary(sdc_xy_curr, sdc_yaw_curr, edge_points)
    else:
      distance_to_edge, _ = jax.vmap(calculate_distances_to_boundary, in_axes=(0, 0, 0))(sdc_xy_curr, sdc_yaw_curr, edge_points)
    
    obs = jnp.concatenate([sdc_vel_curr, dir_diff, distance_to_edge], axis=-1) ## add information of the track? + yaw rate
    return obs
  
  def check_termination(self, state: PlanningGoKartSimState) -> jnp.ndarray:
    """Checks if the episode should terminate.

    Args:
      state: Current state of the simulator of shape (...).

    Returns:
      A boolean array of shape (...) indicating if the episode should terminate.
      reset the episode if the self-driving car is off-road or the episode is done
    """
    metric_dict = self.metrics(state)
    is_offroad = metric_dict["offroad"].value.astype(jnp.bool)
    # if is_offroad | state.is_done:
    #    self.reset(state)
    condition = jnp.logical_or(is_offroad, state.is_done)

    # Reset the simulator state if the condition is met.  currently not implemented!!!
    # jnp.where returns a tuple of indices, so we need to extract the first element.
    reset_idx = jnp.where(condition)[0]

    if jnp.any(condition):
      return True
    else:
      return False
    # return reset_idx

  def reset(self, state: PlanningGoKartSimState, rng: jax.Array | None = None):
    """Resets the simulator state.

    Args:
      state: Current state of the simulator of shape (...).
      rng: Optional random number generator for stochastic environments.

    Returns:
      A new state of the simulator after resetting.
    """
    state = super().reset(state)
    obs  = self.observe(state)
    return state, obs
  
  def step(self, state: PlanningGoKartSimState, action: datatypes.Action) -> PlanningGoKartSimState:
    """Advances simulation by one timestep using the dynamics model.

    Args:
      state: The current state of the simulator of shape (...).
      action: The action to apply, of shape (..., num_objects). The
        actions.valid field is used to denote which objects are being controlled
        - objects whose valid is False will fallback to default behavior
        specified by self.dynamics.
      rng: Optional random number generator for stochastic environments.

    Returns:
      The next simulation state after taking an action of shape (...).
    """
    last_state = copy.deepcopy(state)
    last_metric_dict = metrics.run_metrics(last_state, self.metrics_config)
    state = super().step(state, action)
    metric_dict = metrics.run_metrics(state, self.metrics_config)
    progression_reward = 10 * (metric_dict["sdc_progression"].value - last_metric_dict["sdc_progression"].value)
    obs = self.observe(state)
    # done = self.check_termination(state)
    done = False # for testing
    info = {}
    return obs, state, progression_reward, done, info

  

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


def get_edge_points(roadgraph_points_pos, roadgraph_points_types) -> jnp.ndarray:
    is_road_edge = datatypes.is_road_edge(roadgraph_points_types)
    indices = jnp.where(is_road_edge)
    edge_points = roadgraph_points_pos[indices[0], indices[1], :]
    if len(roadgraph_points_pos.shape) == 1:
        edge_points = edge_points.reshape((-1, 2))
    else:
        edge_points = edge_points.reshape((roadgraph_points_pos.shape[0], -1, 2))
    return edge_points