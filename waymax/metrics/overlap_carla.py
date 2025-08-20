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

"""Metrics relating to overlaps."""
import jax
from jax import numpy as jnp

from waymax import datatypes
from waymax.metrics import abstract_metric
from waymax.utils import geometry


class OverlapMetric(abstract_metric.AbstractMetric):
  """Overlap metric.

  This metric returns 1.0 if an object's bounding box is overlapping with
  that of another object.
  """

  @jax.named_scope('OverlapMetric.compute')
  def compute(
      self, simulator_state: datatypes.SimulatorState
  ) -> abstract_metric.MetricResult:
    # sim_traj = datatypes.select_by_onehot(simulator_state.sim_trajectory, simulator_state.object_metadata.is_sdc)   # Select only the SDC.
    # sim_traj = jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), sim_traj)     # Expand to add batch dimension.

    # current_object_state = datatypes.dynamic_slice(
    #     sim_traj,
    #     simulator_state.timestep,
    #     1,
    #     -1,
    # )

    current_object_state = datatypes.dynamic_slice(
        simulator_state.sim_trajectory,
        simulator_state.timestep,
        1,
        -1,
    )

    return self.compute_overlap(current_object_state, simulator_state.object_metadata.is_sdc)

  def compute_overlap_new(
      self, current_traj: datatypes.Trajectory
  ) -> abstract_metric.MetricResult:
    """Computes the overlap metric.

    Args:
      current_traj: Trajectory object containing current states of shape (...,
        num_objects, num_timesteps=1).

    Returns:
      A (..., num_objects) MetricResult.
    """
    traj_5dof = current_traj.stack_fields(['x', 'y', 'length', 'width', 'yaw'])
    
    # Shape: (..., num_objects, num_objects)
    print("DEBUG: Shape of traj_5dof:", traj_5dof.shape)

    pairwise_overlap = geometry.compute_pairwise_overlaps(traj_5dof[..., 0, :])

    print("DEBUG: Shape of pairwise_overlap:", pairwise_overlap.shape)

    # Remove overlaps with invalid objects
    # This is a no-op, but explicitly writing this since we want to
    # broadcast logical_and across agents, but the last dimension by default
    # corresponds to time.
    valid = current_traj.valid[..., 0:1]  # Shape: (..., num_objects, 1)
    pairwise_overlap = jnp.logical_and(pairwise_overlap, valid)
    num_overlap = jnp.sum(pairwise_overlap, axis=-2)
    overlap_indication = (num_overlap > 0).astype(jnp.float32)

    metric = abstract_metric.MetricResult.create_and_validate(
        overlap_indication, valid[..., 0]
    )

    print("***")
    print("DEBUG: Shape of metric:", metric.shape)
    print("DEBUG: Shape of metric:", metric.value.shape)
    print("DEBUG: Shape of valid:", metric.valid.shape)

    # Unbatch the metric to remove the batch dimension.
    metric = metric.replace(
        value=jnp.squeeze(metric.value, axis=-1),
        valid=jnp.squeeze(metric.valid, axis=-1),
    )

    print("DEBUG: Shape of metric after unbatching:", metric.shape)
    print("DEBUG: Shape of metric:", metric.value.shape)
    print("DEBUG: Shape of valid:", metric.valid.shape)
    print("***")

    return metric


  def compute_overlap(
      self, current_traj: datatypes.Trajectory, is_sdc: jnp.ndarray
  ) -> abstract_metric.MetricResult:
    """Computes the overlap metric.

    Args:
      current_traj: Trajectory object containing current states of shape (...,
        num_objects, num_timesteps=1).

    Returns:
      A (..., num_objects) MetricResult.
    """
    traj_5dof = current_traj.stack_fields(['x', 'y', 'length', 'width', 'yaw'])
    # Shape: (..., num_objects, num_objects)
    pairwise_overlap = geometry.compute_pairwise_overlaps(traj_5dof[..., 0, :])

    # Remove overlaps with invalid objects
    # This is a no-op, but explicitly writing this since we want to
    # broadcast logical_and across agents, but the last dimension by default
    # corresponds to time.
    valid = current_traj.valid[..., 0:1]  # Shape: (..., num_objects, 1)
    pairwise_overlap = jnp.logical_and(pairwise_overlap, valid)
    num_overlap = jnp.sum(pairwise_overlap, axis=-2)
    overlap_indication = (num_overlap > 0).astype(jnp.float32)

    metric = abstract_metric.MetricResult.create_and_validate(
        overlap_indication, valid[..., 0]
    )

    # Shape (num_objects,): (2,)

    print("DEBUG: Shape of metric:", metric.shape)
    print("DEBUG: Shape of value:", metric.value.shape)
    print("DEBUG: Shape of valid:", metric.valid.shape)

    # I only want to extract the overlap of the SDC object --> from (2,) to ()
    sdc_overlap = datatypes.select_by_onehot(overlap_indication, is_sdc)
    sdc_valid = datatypes.select_by_onehot(valid[..., 0], is_sdc)

    metric = abstract_metric.MetricResult.create_and_validate(
        sdc_overlap, sdc_valid
    )

    print("* DEBUG: Shape of SDC metric:", metric.shape)
    print("* DEBUG: Shape of SDC value:", metric.value.shape)
    print("* DEBUG: Shape of SDC valid:", metric.valid.shape)

    return metric