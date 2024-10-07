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
import dataclasses
from doctest import debug
from typing import Sequence

import beartype
import chex
import jax
import jax.numpy as jnp
from jax import Array
from jax.experimental import checkify
from jaxtyping import Float, jaxtyped

from waymax import config as _config, datatypes, dynamics as _dynamics, metrics, rewards
from waymax.agents import actor_core
from waymax.env import typedefs as types, PlanningAgentEnvironment

from waymax.utils.gokart_utils import TrackControlPoints, generate_racing_track

typechecker = beartype.beartype


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
        self._state_dynamics = _dynamics.GoKartStateDynamics()
        self.metrics_config = dataclasses.replace(_config.MetricsConfig(),
                                                  metrics_to_run=("offroad", "sdc_progression"))
        reward_config = _config.LinearCombinationRewardConfig(rewards={'offroad': -1.0, 'sdc_progression': 10.0})
        self.reward_fn = rewards.LinearCombinationReward(reward_config)
        self._current_position = None
        self._current_yaw = None
        self._current_velocity = None

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
          sdc_xy_curr: current position of the self-driving car in the global coordinate system
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

        dir_ref = self.get_ref_direction(state)  # (...,2)
        dir_ref = jnp.arctan2(dir_ref[..., 1], dir_ref[..., 0])  # (...,)
        dir_diff = sdc_yaw_curr - dir_ref  # (...,)
        dir_diff = dir_diff[..., None]  # (..., 1)

        # is_road_edge = datatypes.is_road_edge(state.roadgraph_points.types)
        # edge_points = state.roadgraph_points.xy[is_road_edge & state.roadgraph_points.valid] # not allowed for jit, dynamic indexing !!!
        # edge_points = get_edge_points(state.roadgraph_points.xy, state.roadgraph_points.types)
        # indices = jnp.where(is_road_edge)[0]   # jnp.where with single argument not compatible with jit!!!
        # edge_points = state.roadgraph_points.xy[indices, :]
        # indices = jnp.where(is_road_edge, size=state.roadgraph_points.types.size, fill_value=-1)[0]

        edge_points = state.roadgraph_points.xy[..., 2000:,
                      :]  # TODO: for testing, need to find a better way to get the edge points
        # jax.debug.breakpoint()
        # for debugging
        # track_cntrl_points = TrackControlPoints()
        # roadgraph_points, x_center, y_center, cumulative_length = generate_racing_track(
        #         track_cntrl_points.x,
        #         track_cntrl_points.y,
        #         track_cntrl_points.r)
        # edge_points = roadgraph_points.xy[2000:, :]
        # assert jnp.array_equal(edge_points, edge_points_test) 

        if len(sdc_xy_curr.shape) == 1:  # no batch dimension
            # edge_points = state.roadgraph_points.xy[is_road_edge]
            assert len(edge_points.shape) == 2
        #     jax.debug.print("Shape of edge_points: {}", edge_points.shape)
        #     jax.debug.print("left edge: {}", edge_points[:5]) 
        #     jax.debug.print("left edge: {}", edge_points[1994:1999]) 
        #     jax.debug.print("right edge: {}", edge_points[2000:2005])
        #     jax.debug.print("right edge: {}", edge_points[-5:])  
        #     jax.debug.print("sdc_xy_curr: {}", sdc_xy_curr)
            distance_to_edge, _, debug_value = calculate_distances_to_boundary(sdc_xy_curr, sdc_yaw_curr, edge_points)
        #     jax.debug.print("distance_to_edge: {}", distance_to_edge)
        #     jax.debug.breakpoint()
        else:
            distance_to_edge, _, _ = jax.vmap(calculate_distances_to_boundary, in_axes=(0, 0, 0))(sdc_xy_curr,
                                                                                                  sdc_yaw_curr,
                                                                                                  edge_points)

        obs = jnp.concatenate(
                [sdc_xy_curr, jnp.array([sdc_yaw_curr]), sdc_vel_curr, dir_diff, distance_to_edge, debug_value],
                axis=-1)  ## add information of the track? + yaw rate
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
        # reset_idx = jnp.where(condition)[0]

        # if jnp.any(condition):
        #   return True
        # else:
        #   return False
        return condition

    def reset(self, state: PlanningGoKartSimState, rng: jax.Array | None = None):
        """Resets the simulator state.

        Args:
          state: Current state of the simulator of shape (...).
          rng: Optional random number generator for stochastic environments.

        Returns:
          A new state of the simulator after resetting.
        """
        chex.assert_equal(
                self.config.max_num_objects, state.log_trajectory.num_objects
        )

        # Fills with invalid values (i.e. -1.) and False.
        sim_traj_uninitialized = datatypes.fill_invalid_trajectory(
                state.log_trajectory
        )
        state_uninitialized = state.replace(
                timestep=jnp.array(-1), sim_trajectory=sim_traj_uninitialized
        )
        state = datatypes.update_state_by_log(
                state_uninitialized, self.config.init_steps
        )
        state = PlanningGoKartSimState(**state)
        if rng is not None:
            keys = jax.random.split(rng, len(self._sim_agent_actors))
        else:
            keys = [None] * len(self._sim_agent_actors)
        init_actor_states = [
            actor_core.init(key, state)
            for key, actor_core in zip(keys, self._sim_agent_actors)
        ]
        state = state.replace(sim_agent_actor_states=init_actor_states)
        obs = self.observe(state)
        return obs, state

    def step(self, state: PlanningGoKartSimState, action: datatypes.Action):
        """Advances simulation by one timestep using the dynamics model.

        Args:
        state: The current state of the simulator of shape (...).
        action: The action to apply, of shape (..., num_objects). 
        rng: Optional random number generator for stochastic environments.

        Returns:
        The next simulation state after taking an action of shape (...).
        """
        # compute reward, currently only progression reward is implemented
        last_state = copy.deepcopy(state)
        last_pos_xy = state.current_sim_trajectory.xy[..., 0, :]
        # shape: (...,2)
        last_sdc_xy = datatypes.select_by_onehot(
                last_pos_xy,
                state.object_metadata.is_sdc,
                keepdims=False,
        )
        
        state = super().step(state, action)
        # current_pos_xy = state.current_sim_trajectory.xy[..., 0, :]
        # # shape: (...,2)
        # current_sdc_xy = datatypes.select_by_onehot(
        #     current_pos_xy,
        #     state.object_metadata.is_sdc,
        #     keepdims=False,
        # )
        # movement_vector = current_sdc_xy - last_sdc_xy
        dir_ref = self.get_ref_direction(state)
        # last_metric_dict = metrics.run_metrics(last_state, self.metrics_config)
        # metric_dict = metrics.run_metrics(state, self.metrics_config)
        # progression_reward = 1000 * (metric_dict["sdc_progression"].value - last_metric_dict["sdc_progression"].value)
        # # progression_reward = self._compute_progression_reward(last_state, state)
        # progression_reward = jnp.where(
        #   jnp.dot(movement_vector, dir_ref) > 0,
        #   progression_reward,
        #   0) # no reward if the self-driving car is moving in the wrong direction (TODO:signed progression reward)
        reward = self.compute_reward(last_state, state, dir_ref)
        obs = self.observe(state)
        done = self.check_termination(state)
        # reward = jnp.where(done, reward, reward+0.05) # encourage the self-driving car to stay on the track
        reward = jnp.where(done & jnp.logical_not(state.is_done), reward - 5, reward) # penalize the self-driving car for going off-road
        obs, state = self.post_step(state, obs, done)
        # done = False # for testing
        info = {}
        return jax.lax.stop_gradient(obs), jax.lax.stop_gradient(state), reward, done, info

    def post_step(self, state: PlanningGoKartSimState, obs: jnp.ndarray, done) -> PlanningGoKartSimState:
        """reset the environment if the self-driving car is off-road or the episode is done

        Args:
          state: The current state of the simulator
          obs: observations
          dones: A boolean indicating if the episode should terminate

        Returns:
          The simulation state after post-step processing.
        """
        # states_re, obs_re = self.reset(state)

        # # reset environments based on termination
        # state = jax.tree_map(
        #     lambda x, y: jax.lax.select(done, x, y), states_re, state
        # )
        # obs = jax.tree_map(
        #     lambda x, y: jax.lax.select(done, x, y), obs_re, obs
        # )

        obs, state = jax.lax.cond(
                done,
                lambda _: self.reset(state),
                lambda _: (obs, state),
                operand=None
        )
        return obs, state

    def compute_reward(self, last_state: PlanningGoKartSimState, state: PlanningGoKartSimState, dir_ref) -> jnp.ndarray:
        """Computes the reward for the given simulation state.

        Args:
        
        state: Current state of the simulator of shape (...).

        Returns:
        The reward for the given simulation state.
        """
        # progression_reward = self._compute_progression_reward(last_state, state)
        orientation_reward = self._compute_orientation_reward(state, dir_ref)
        # reward = progression_reward + orientation_reward
        return orientation_reward
        # return reward

    def _compute_progression_reward(self, last_state: PlanningGoKartSimState,
                                    state: PlanningGoKartSimState) -> jnp.ndarray:
        """Computes the progression reward.

        Args:
          last_state: The last state of the simulator.
          state: The current state of the simulator.

        Returns:
          The progression reward.
        """
        sdc_paths = state.sdc_paths
        if sdc_paths is None:
            raise ValueError(
                    'SimulatorState.sdc_paths required to compute the route progression '
                    'metric.'
            )

        # Shape: (..., num_objects, num_timesteps=1, 2)
        obj_xy_last = datatypes.dynamic_slice(
                last_state.sim_trajectory.xy,
                last_state.timestep,
                1,
                axis=-2,
        )
        obj_xy_curr = datatypes.dynamic_slice(
                state.sim_trajectory.xy,
                state.timestep,
                1,
                axis=-2,
        )

        # Shape: (..., 2)
        sdc_xy_last = datatypes.select_by_onehot(
                obj_xy_last[..., 0, :],
                last_state.object_metadata.is_sdc,
                keepdims=False,
        )
        sdc_xy_curr = datatypes.select_by_onehot(
                obj_xy_curr[..., 0, :],
                state.object_metadata.is_sdc,
                keepdims=False,
        )

        # sdc_xy_start = datatypes.select_by_onehot(
        #     state.log_trajectory.xy[..., 0, :],
        #     state.object_metadata.is_sdc,
        #     keepdims=False,
        # )
        # sdc_xy_end = datatypes.select_by_onehot(
        #     state.log_trajectory.xy[..., -1, :],
        #     state.object_metadata.is_sdc,
        #     keepdims=False,
        # )

        # # Shape: (..., num_objects, num_timesteps=1)
        # obj_valid_curr = datatypes.dynamic_slice(
        #     state.sim_trajectory.valid,
        #     simulator_state.timestep,
        #     1,
        #     axis=-1,
        # )
        # # Shape: (...)
        # sdc_valid_curr = datatypes.select_by_onehot(
        #     obj_valid_curr[..., 0],
        #     simulator_state.object_metadata.is_sdc,
        #     keepdims=False,
        # )

        # Shape: (..., num_paths, num_points_per_path)
        dist = jnp.linalg.norm(
                sdc_paths.xy - jnp.expand_dims(sdc_xy_curr, axis=(-2, -3)),
                axis=-1,
                keepdims=False,
        )
        # # Only consider valid on-route paths.
        # dist = jnp.where(sdc_paths.valid & sdc_paths.on_route, dist_raw, jnp.inf)
        # # Only consider valid SDC states.
        # dist = jnp.where(
        #     jnp.expand_dims(sdc_valid_curr, axis=(-1, -2)), dist, jnp.inf
        # )
        dist_path = jnp.min(dist, axis=-1,
                            keepdims=True)  # (..., num_paths, 1) find the nearest point to the car on each path
        idx = jnp.argmin(dist_path, axis=-2, keepdims=True)  # (..., 1, 1) find the index of the nearest path
        min_dist_path = jnp.min(dist, axis=(-1, -2))  # (...) find the minimum distance to the nearest path

        # Shape: (..., max(num_points_per_path))
        ref_path = jax.tree_util.tree_map(
                lambda x: jnp.take_along_axis(x, indices=idx, axis=-2)[..., 0, :],
                sdc_paths,
        )

        def get_arclength_for_pts(xy: jax.Array, path: datatypes.Paths):
            # Shape: (..., max(num_points_per_path))
            dist_raw = jnp.linalg.norm(
                    xy[..., jnp.newaxis, :] - path.xy, axis=-1, keepdims=False
            )
            dist = jnp.where(path.valid, dist_raw, jnp.inf)
            idx = jnp.argmin(dist, axis=-1, keepdims=True)
            # (..., )
            return jnp.take_along_axis(path.arc_length, indices=idx, axis=-1)[..., 0]

        # start_dist = get_arclength_for_pts(sdc_xy_start, ref_path)
        # end_dist = get_arclength_for_pts(sdc_xy_end, ref_path)
        last_dist = get_arclength_for_pts(sdc_xy_last, ref_path)
        curr_dist = get_arclength_for_pts(sdc_xy_curr, ref_path)

        # progress = jnp.where(
        #     end_dist == start_dist,
        #     FULL_PROGRESS_VALUE,
        #     (curr_dist - start_dist) / (end_dist - start_dist),
        # )
        progress = curr_dist - last_dist
        valid = jnp.isfinite(min_dist_path)
        progress = jnp.where(valid, progress, 0.0)
        return progress

    def _compute_orientation_reward(self, state: PlanningGoKartSimState, dir_ref: jnp.ndarray) -> jnp.ndarray:
        """Computes the orientation reward. TODO more detialed

        Args:
        state: The current state of the simulator.

        Returns:
        The orientation reward.
        """
        # shape: (..., num_objects, timesteps=1, 2) -> (..., num_objects, 2)
        vel_xy = state.current_sim_trajectory.vel_xy[..., 0, :]

        # shape: (...,2)
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
        yaw_vec = jnp.array([jnp.cos(sdc_yaw_curr), jnp.sin(sdc_yaw_curr)])  # (..., 2)
        orientation_reward = jnp.dot(yaw_vec, dir_ref)  # (...,)
        orientation_reward *= jnp.dot(sdc_vel_curr, dir_ref)  # (...,)
        orientation_reward *= 0.05
        orientation_reward = jnp.clip(orientation_reward, -0.05, 0.05)
        return orientation_reward

    def get_ref_direction(self, state: PlanningGoKartSimState) -> jnp.ndarray:
        """Get the reference direction of the self-driving car

        Args:
          state: The current state of the simulator

        Returns:
          The reference direction of the self-driving car
        """
        # shape: (..., num_objects, timesteps=1, 2) -> (..., num_objects, 2)
        pos_xy = state.current_sim_trajectory.xy[..., 0, :]

        # shape: (...,2)
        sdc_xy_curr = datatypes.select_by_onehot(
                pos_xy,
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
        dir_ref = jnp.squeeze(dir_ref, axis=(-2, -3))  # (...,2)

        return dir_ref


@jaxtyped(typechecker=typechecker)
def calculate_distances_to_boundary(
        car_position: Float[Array, "2"],
        car_yaw: Float[Array, ""],
        boundary_points: Float[Array, "N 2"],
        num_rays: int = 8,
        max_distance: float = 0.1):
    """
    calculate distances to boundary in different directions
    
    Args:
    car_position: car position (x, y)
    car_yaw: car orientation in radians
    boundary_points: boundary points of the track shape (N, 2)
    num_rays: number of rays to cast
    
    Returns:
    distances: distance to boundary in different directions    shape (num_rays,)
    hit_points: points of intersections of rays and boundary    shape (num_rays, 2)
    """

    # checked_fn = checkify.checkify(check_greater)
    # jax.debug.breakpoint()    
    angles = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, num_rays)
    rotated_angles = car_yaw + angles
    ray_directions: Float[Array, "2 nRays"] = jnp.array([jnp.cos(rotated_angles), jnp.sin(rotated_angles)])

    boundary2car: Float[Array, "N 2"] = boundary_points - car_position
    projections: Float[Array, "N nRays"] = jnp.dot(boundary2car, ray_directions)
    boundary2car_dist: Float[Array, "N 1"] = jnp.linalg.norm(boundary2car, axis=1, keepdims=True)  # (N, 1)
    # error, out = checked_fn(distances_to_points**2 - projections**2, 0)
    # error.throw()
    # perpendicular_distances = jnp.sqrt(distances_to_points**2 - projections**2)  # (N, num_rays)
    # TODO reproduce the error, check boundary points
    boundary2car_dist_repeated: Float[Array, "N nRays"] = jnp.repeat(boundary2car_dist, repeats=num_rays, axis=1)  # (N, 8)
    boundary2rays: Float[Array, "N nRays"] = (boundary2car_dist_repeated + 1e-6) ** 2 - projections ** 2
    # tested: min(abs(boundary2car_dist_repeated) - abs(projections)) ~= -9.536e-07 numerical error???
    debug_values1 = jnp.min(abs(boundary2car_dist_repeated) - abs(projections))
    debug_values2 = jnp.min(boundary2rays)

    perpendicular_distances = jnp.sqrt(jnp.maximum(boundary2rays, 0))
    # TODO log the value!!! (N, num_rays) CAR OUTSIDE THE TRACK

    valid_mask = (projections > 0) & (perpendicular_distances < max_distance)
    valid_projections = jnp.where(valid_mask, projections, jnp.inf)  # (N, num_rays)

    distances: Float[Array, "nRays"] = jnp.min(valid_projections, axis=0)  # (num_rays,)
#     jax.debug.breakpoint() 
    hit_points: Float[Array, "nRays 2"] = car_position + ray_directions.T * distances[:, None]  # (num_rays, 2)

    return distances, hit_points, jnp.array([debug_values1, debug_values2])


def get_edge_points(roadgraph_points_pos, roadgraph_points_types) -> Float[Array, "N 2"]:
    is_road_edge = datatypes.is_road_edge(roadgraph_points_types)
    indices = jnp.where(is_road_edge)
    edge_points = roadgraph_points_pos[indices[0], indices[1], :]
    if len(roadgraph_points_pos.shape) == 1:
        edge_points = edge_points.reshape((-1, 2))
    else:
        edge_points = edge_points.reshape((roadgraph_points_pos.shape[0], -1, 2))
    return edge_points


def check_greater(a: Array, b: Array):
    condition = jnp.all(a > b)
    checkify.check(condition, f"Assertion failed: is not greater than {b}")
