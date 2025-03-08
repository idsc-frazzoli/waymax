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

"""All customized data structures for Waymax.

Note the shape of a data classes here is defined as the most common prefix shape
among all attributes.
The validate function is implemented separately instead of as __post_init__, to
have better support with jax utils.
"""

from typing import Any, Optional, Generic

import chex
import jax
import jax.numpy as jnp

from waymax import config
from waymax.datatypes import array, action, object_state, operations, roadgraph, route, traffic_lights, dynamics_parameters
from waymax.datatypes.object_state import TrajectoryType

ArrayLike = jax.typing.ArrayLike
PyTree = array.PyTree


@chex.dataclass
class SimulatorState(Generic[TrajectoryType]):
    """A dataclass holding the simulator state, all data in global coordinates.

    Attributes:
      sim_trajectory: Simulated trajectory for all objects of shape (...,
        num_objects, num_timesteps). The number of timesteps is the same as in the
        log, but future trajectory points that have not been simulated will be
        marked invalid.
      log_trajectory: Logged trajectory for all objects of shape (...,
        num_objects, num_timesteps).
      log_traffic_light: Logged traffic light information for the entire run
        segment of shape (..., num_traffic_lights, num_timesteps).
      object_metadata: Metadata for all objects of shape (..., num_objects).
      timestep: The current simulation timestep index of shape (...). Note that
        sim_trajectory at `timestep` is the last executed step by the simulator.
      sdc_paths: Paths for SDC, representing where the SDC can drive of shape
        (..., num_paths, num_points_per_path).
      roadgraph_points: A optional RoadgraphPoints holding subsampled roadgraph
        points of shape (..., num_points).
    """

    sim_trajectory: TrajectoryType
    # TODO Support testset, i.e. no log_trajectory for all steps.
    log_trajectory: TrajectoryType
    log_traffic_light: traffic_lights.TrafficLights
    object_metadata: object_state.ObjectMetadata
    timestep: jax.typing.ArrayLike
    sdc_paths: Optional[route.Paths] = None
    roadgraph_points: Optional[roadgraph.RoadgraphPoints] = None
    dynamics_params: Optional[dynamics_parameters.GokartDynamicsParams] = None

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape is defined as the most common prefix shape of all attributes."""
        # Here, shape is equivalent to batch dimensions, and can be ().
        return self.object_metadata.shape[:-1]

    @property
    def batch_dims(self) -> tuple[int, ...]:
        """Batch dimensions."""
        return self.shape

    @property
    def num_objects(self) -> int:
        """The number of objects included in this trajectory per example."""
        return self.object_metadata.num_objects

    @property
    def is_done(self) -> bool:
        """Returns whether the simulation is at the end of the logged history."""
        return jnp.array(  # pytype: disable=bad-return-type  # jnp-type
                (self.timestep + 1) >= self.log_trajectory.num_timesteps, bool
        )

    @property
    def remaining_timesteps(self) -> int:
        """Returns the number of remaining timesteps in the episode."""
        return jnp.array(
                self.log_trajectory.num_timesteps - self.timestep - 1, int
        )  # pytype: disable=bad-return-type  # jnp-type

    @property
    def current_sim_trajectory(self) -> TrajectoryType:
        """Returns the trajectory corresponding to the current sim state."""
        return operations.dynamic_slice(self.sim_trajectory, self.timestep, 1, axis=-1)

    @property
    def previous_sim_trajectory(self) -> TrajectoryType:
        """Returns the trajectory corresponding to the previous sim state."""
        timestep = jnp.max(self.timestep - 1, 0)
        return operations.dynamic_slice(self.sim_trajectory, timestep, 1, axis=-1)

    def select_previous_sim_trajectory(self, n_previous_steps: int, slice_size: int = 1) -> TrajectoryType:
        """Returns the trajectory corresponding to the n_previous_steps sim state."""
        def _raise_if_negative(n_previous_steps):
            if n_previous_steps < 0:
                raise ValueError(f"n_previous_steps ({n_previous_steps}) must be non-negative.")
        jax.debug.callback(_raise_if_negative, n_previous_steps)
        timestep = jnp.maximum(self.timestep - n_previous_steps, 0)
        return operations.dynamic_slice(self.sim_trajectory, timestep, slice_size, axis=-1)

    @property
    def current_log_trajectory(self) -> TrajectoryType:
        """Returns the trajectory corresponding to the current sim state."""
        return operations.dynamic_slice(self.log_trajectory, self.timestep, 1, axis=-1)

    def __eq__(self, other: Any) -> bool:
        return operations.compare_all_leaf_nodes(self, other)

    def validate(self):
        """Validates shape and type."""
        data = [
            self.sim_trajectory,
            self.log_trajectory,
            self.log_traffic_light,
            self.object_metadata,
            self.timestep,
        ]
        if self.roadgraph_points is not None:
            data.append(self.roadgraph_points)
        chex.assert_equal_shape_prefix(data, len(self.shape))


@chex.dataclass
class GoKartSimState(SimulatorState[object_state.GokartTrajectory]):
    """
    A dataclass holding the simulator state for the gokart environment.
    """
    actions_history: Optional[action.GokartAction] = None
    sdc_paths: Optional[route.GoKartPaths] = None

    @property
    def current_action_history(self) -> action.GokartAction:
        """Returns the actions corresponding to the current sim state."""
        return self.select_previous_actions_history(0)

    @property
    def previous_action_history(self) -> action.GokartAction:
        """Returns the trajectory corresponding to the previous sim state."""
        return self.select_previous_actions_history(1)

    def select_previous_actions_history(self, n_previous_steps: int, slice_size: int = 1) -> action.GokartAction:
        """Returns the trajectory corresponding to the n_previous_steps sim state."""
        def _raise_if_negative(n_previous_steps):
            if n_previous_steps < 0:
                raise ValueError(f"n_previous_steps ({n_previous_steps}) must be non-negative.")
        jax.debug.callback(_raise_if_negative, n_previous_steps)
        timestep = jnp.maximum(self.timestep - n_previous_steps, 0)
        return operations.dynamic_slice(self.actions_history, timestep, slice_size, axis=-1)

    def __eq__(self, other: Any) -> bool:
        return operations.compare_all_leaf_nodes(self, other)


def update_state_by_log(state: SimulatorState | GoKartSimState, num_steps: int) -> SimulatorState | GoKartSimState:
    """Advances SimulatorState by num_steps using logged data."""
    # TODO jax runtime check num_steps > state.remaining_timesteps
    return state.replace(
            timestep=state.timestep + num_steps,
            sim_trajectory=operations.update_by_slice_in_dim(
                    inputs=state.sim_trajectory,
                    updates=state.log_trajectory,
                    inputs_start_idx=state.timestep + 1,
                    slice_size=num_steps,
                    axis=-1,
            ),
    )


def get_control_mask(metadata: object_state.ObjectMetadata, obj_type: config.ObjectType) -> jax.Array:
    """Returns binary mask for selected object type.

    Args:
      metadata: An ObjectMetadata, having shape (..., num_objects).
      obj_type: Represents which type of objects should be selected.

    Returns:
      A binary mask with shape (..., num_objects).
    """

    if obj_type == config.ObjectType.SDC:
        is_controlled = metadata.is_sdc
    elif obj_type == config.ObjectType.MODELED:
        is_controlled = metadata.is_modeled
    elif obj_type == config.ObjectType.VALID:
        is_controlled = metadata.is_valid
    else:
        raise ValueError(f"Invalid ObjectType {obj_type}")
    return is_controlled
