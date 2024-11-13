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

"""Gokart environment for tasks relating to Planning for the ADV."""

import dataclasses
from typing import Sequence
from typing import Optional

import beartype
import chex
import jax
import jax.numpy as jnp
from dm_env.specs import BoundedArray
from jax import Array
from jax.experimental import checkify
from jaxtyping import Float, jaxtyped

from waymax import config as _config, datatypes, dynamics as _dynamics, rewards
from waymax.agents import actor_core
from waymax.env import typedefs as types, PlanningAgentEnvironment
from waymax.utils.geometry import rotation_matrix, wrap_yaws
from waymax.datatypes.observation import Observation, ObjectPose2D

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

    def __init__(
        self,
        dynamics_model: _dynamics.DynamicsModel,
        config: _config.EnvironmentConfig,
        sim_agent_actors: Sequence[actor_core.WaymaxActorCore] = (),
        sim_agent_params: Sequence[actor_core.Params] = (),
    ) -> None:
        super().__init__(dynamics_model, config, sim_agent_actors, sim_agent_params)
        self._state_dynamics = _dynamics.GoKartStateDynamics()
        self.metrics_config = dataclasses.replace(
            _config.MetricsConfig(), metrics_to_run=("gokart_offroad", "gokart_progress", "gokart_orientation")
        )
        reward_config = _config.LinearCombinationRewardConfig(
            rewards={"gokart_offroad": 5, "gokart_progress": 1.0, "gokart_orientation": 0.05}
        )
        self._reward_function = rewards.LinearCombinationReward(reward_config)

    def observation_spec(self) -> BoundedArray:
        # todo add observation information (should not be from ppo config)
        # create obs type for teh gokart environment
        dim = 15
        minimum = -jnp.array([jnp.inf] * dim)
        maximum = jnp.array([jnp.inf] * dim)
        specs = BoundedArray((15,), jnp.float32, minimum, maximum)
        return specs

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

        dir_ref, nearest_index = self._get_ref_direction(state)  # (...,num,2)
        dir_ref = wrap_yaws(jnp.arctan2(dir_ref[..., 1], dir_ref[..., 0]))  # (...,num)
        # dir_diff = sdc_yaw_curr - dir_ref  # (...,)

        dir_diff = wrap_yaws(dir_ref - sdc_yaw_curr)  # (...,num)
        # jax.debug.breakpoint()
        # future_track, _ = get_future_track(state, sdc_xy_curr, sdc_yaw_curr, nearest_index)

        yaw_rate = state.current_sim_trajectory.yaw_rate[..., 0]

        sdc_yaw_rate_curr = datatypes.select_by_onehot(
            yaw_rate,
            state.object_metadata.is_sdc,
            keepdims=False,
        )
        sdc_yaw_rate_curr = jnp.array([sdc_yaw_rate_curr])

        # TODO: for testing, need to find a better way to get the edge points
        edge_points = state.roadgraph_points.xy[..., 2000:, :]

        # todo move to always have a batch dimention?
        if len(sdc_xy_curr.shape) == 1:  # no batch dimension
            assert len(edge_points.shape) == 2
            distance_to_edge, _, debug_value = calculate_distances_to_boundary(sdc_xy_curr, sdc_yaw_curr, edge_points)
        else:
            distance_to_edge, _, _ = jax.vmap(calculate_distances_to_boundary, in_axes=(0, 0, 0))(
                sdc_xy_curr, sdc_yaw_curr, edge_points
            )

        obs = jnp.concatenate(
            [sdc_vel_curr, sdc_yaw_rate_curr, dir_diff, distance_to_edge], axis=-1
        )  ## add information of the track? + yaw rate  #future_track.ravel()
        # sdc_xy_curr, jnp.array([sdc_yaw_curr]), , debug_value
        return obs

    def reset(self, state: PlanningGoKartSimState, rng: jax.Array | None = None) -> PlanningGoKartSimState:
        """Resets the simulator state.

        Args:
          state: Current state of the simulator of shape (...).
          rng: Optional random number generator for stochastic environments.

        Returns:
          A new state of the simulator after resetting.
        """
        chex.assert_equal(self.config.max_num_objects, state.log_trajectory.num_objects)

        # Fills with invalid values (i.e. -1.) and False.
        sim_traj_uninitialized = datatypes.fill_invalid_trajectory(state.log_trajectory)
        state_uninitialized = state.replace(timestep=jnp.array(-1), sim_trajectory=sim_traj_uninitialized)
        state = datatypes.update_state_by_log(state_uninitialized, self.config.init_steps)
        state = PlanningGoKartSimState(**state)
        if rng is not None:
            # random initial position and velocity
            keys = jax.random.split(rng, len(self._sim_agent_actors))
            rng_p, rng_v = jax.random.split(rng)
            num_init_points = state.sdc_paths.num_points_per_path
            valid_index = jnp.arange(0, 2000, 400)
            # init_index = jax.random.randint(rng, (), 0, num_init_points)
            # init_index = jax.random.randint(rng_p, (), 0, num_init_points)
            init_index = jax.random.choice(rng, valid_index)
            init_pos = state.sdc_paths.xy[..., 0, init_index, :]
            init_orint = state.sdc_paths.dir_xy[..., 0, init_index, :]
            init_yaw = jnp.arctan2(init_orint[..., 1], init_orint[..., 0])
            init_velocity = jax.random.uniform(rng_v, (), minval=0.0, maxval=1)
            state.sim_trajectory.x = state.sim_trajectory.x.at[..., 0, 0].set(init_pos[0])
            state.sim_trajectory.y = state.sim_trajectory.y.at[..., 0, 0].set(init_pos[1])
            state.sim_trajectory.yaw = state.sim_trajectory.yaw.at[..., 0, 0].set(init_yaw)
            # state.sim_trajectory.vel_x = state.sim_trajectory.vel_x.at[..., 0, 0].set(init_velocity)

        else:
            keys = [None] * len(self._sim_agent_actors)
        init_actor_states = [actor_core.init(key, state) for key, actor_core in zip(keys, self._sim_agent_actors)]
        state = state.replace(sim_agent_actor_states=init_actor_states)
        return state

    def action_spec(self) -> BoundedArray:
        data_spec = self.dynamics.action_spec()  # rank 1
        return data_spec

    def step(
        self, state: PlanningGoKartSimState, action: datatypes.Action, rng: jax.Array | None = None
    ) -> PlanningGoKartSimState:
        """
        Advances simulation by one timestep using the dynamics model.

        Args:
        state: The current state of the simulator of shape (...).
        action: The action to apply, of shape (..., num_objects).
        rng: Optional random number generator for stochastic environments.

        Returns:
        The next simulation state after taking an action of shape (...).
        """
        new_state: PlanningGoKartSimState = super().step(state, action, rng)
        return new_state

    #     # compute reward, currently only progression reward is implemented
    #     # last_state = copy.deepcopy(state)
    #
    #
    #     # dir_ref, _ = self.get_ref_direction(state)
    #     obs = self.observe(state)
    #     done = self.check_termination(state)
    #     # reward, reward_dict = self.compute_reward(last_state, state, dir_ref, done)
    #     reward, reward_dict = self.compute_reward(state, action)
    #     obs, state = self.post_step(state, obs, done, rng)
    #     # done = False # for testing
    #     info = reward_dict
    #     return jax.lax.stop_gradient(obs), jax.lax.stop_gradient(state), reward, done, info

    def termination(self, state: PlanningGoKartSimState) -> jax.Array:
        """reset the environment if the self-driving car is off-road or the episode is done

        Args:
          state: The current state of the simulator

        Returns:
          Boolean array indicating if the episode should terminate
        """
        # fixme can be optimized to not recompute all the metrics
        metric_dict = self.metrics(state)
        is_offroad = metric_dict["gokart_offroad"].value.astype(jnp.bool)
        condition = jnp.logical_or(is_offroad, state.is_done)
        return condition.squeeze()

    def _get_ref_direction(self, state: PlanningGoKartSimState, num=1) -> jnp.ndarray:
        """Get the reference direction of the self-driving car
        take the direction of the nearest point on the track as the reference direction

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
        dist = jnp.where(jnp.expand_dims(sdc_valid_curr, axis=(-1, -2)), dist, jnp.inf)
        # index of the nearest point on the reference path
        idx = jnp.argmin(dist, axis=-1, keepdims=True)  # (..., num_paths=1, 1)

        if num > 1:
            n = jnp.int32(state.roadgraph_points.shape[0] / 3)
            idx = (idx + jnp.arange(0, 10 * num, 10)) % n  # (..., num_paths=1, num)
        # use the direction of the nearest sdc_path point as referece direction
        dir_ref = jnp.take_along_axis(state.sdc_paths.dir_xy, idx[..., None], axis=-2)  # (..., num_paths=1, num, 2)
        dir_ref = jnp.squeeze(dir_ref, axis=-3)  # (...,num, 2)

        return dir_ref, idx[0]


@jaxtyped(typechecker=typechecker)
def calculate_distances_to_boundary(
    car_position: Float[Array, "2"],
    car_yaw: Float[Array, ""],
    boundary_points: Float[Array, "N 2"],
    num_rays: int = 11,
    max_distance: float = 0.1,
):
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
    #     jax.debug.print("car_position: {}", car_position)
    #     jax.debug.print("car_yaw: {}", car_yaw)

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
    boundary2car_dist_repeated: Float[Array, "N nRays"] = jnp.repeat(
        boundary2car_dist, repeats=num_rays, axis=1
    )  # (N, 8)
    boundary2rays: Float[Array, "N nRays"] = (boundary2car_dist_repeated + 1e-6) ** 2 - projections**2
    # tested: min(abs(boundary2car_dist_repeated) - abs(projections)) ~= -9.536e-07 numerical error???
    debug_values1 = jnp.min(abs(boundary2car_dist_repeated) - abs(projections))
    debug_values2 = jnp.min(boundary2rays)
    #     jax.debug.print("debug_values1: {}", debug_values1)
    #     jax.debug.print("debug_values2: {}", debug_values2)

    perpendicular_distances = jnp.sqrt(jnp.maximum(boundary2rays, 0))
    # TODO log the value!!! (N, num_rays) CAR OUTSIDE THE TRACK

    valid_mask = (projections > 0) & (perpendicular_distances < max_distance)
    valid_projections = jnp.where(valid_mask, projections, jnp.inf)  # (N, num_rays)

    distances: Float[Array, "nRays"] = jnp.min(valid_projections, axis=0)  # (num_rays,)
    #     jax.debug.breakpoint()
    hit_points: Float[Array, "nRays 2"] = car_position + ray_directions.T * distances[:, None]  # (num_rays, 2)

    #     jax.debug.print("distances: {}", distances)
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


def get_future_track(state: PlanningGoKartSimState, car_pos, car_orientation, nearest_index, num_points=60):
    """
    Get the reference path (centerline) ahead of the car (60 points ~= 6m) # TODO based on velocity??
    """
    roadgraph_points = state.roadgraph_points.xy
    n = jnp.int32(roadgraph_points.shape[0] / 3)
    # idxs = (jnp.arange(nearest_index, nearest_index + num_points) % n)
    idxs = (nearest_index + jnp.arange(num_points)) % n
    # track_points = roadgraph_points[idxs, :]
    track_points = jnp.take_along_axis(roadgraph_points, idxs[:, None], axis=0)
    relative_track = track_points - car_pos  # TODO consider the orientation of the car
    r_matrix = rotation_matrix(car_orientation)
    relative_track_local = relative_track @ r_matrix
    return relative_track_local, track_points
