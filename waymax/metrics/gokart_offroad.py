from typing import Optional

import jax
from jax import numpy as jnp

from waymax import datatypes
from waymax.metrics import abstract_metric
from waymax.metrics.roadgraph import is_offroad, compute_signed_distance_object_to_nearest_road_edge_point


class GokartOffroadMetric(abstract_metric.AbstractMetric):
    """Offroad metric.

    This metric returns 1.0 if the object is offroad.
    """

    def __init__(self, safety_margin: float = 0.0):
        """Initializes the offroad metric.

        Args:
            safety_margin: the gokart is considered offroad if its distance to the closest boundary
                is equal or less than this value.
        """
        assert isinstance(safety_margin, (float, int))
        self._safety_margin = safety_margin

    @jax.named_scope("GokartOffroadMetric.compute")
    def compute(self, state: datatypes.SimulatorState) -> abstract_metric.MetricResult:
        """Computes the offroad metric.

        Args:
          state: Updated simulator state to calculate metrics for. Will
            compute the offroad metric for timestep `state.timestep`.

        Returns:
          An array containing the metric result of the same shape as the input
            trajectories. The shape is (..., num_objects).
        """
        current_object_state = datatypes.dynamic_slice(
            state.sim_trajectory,
            state.timestep,
            1,
            -1,
        )
        offroad = is_offroad(current_object_state, state.roadgraph_points, self._safety_margin)
        valid = jnp.ones_like(offroad, dtype=jnp.bool_)
        metric = abstract_metric.MetricResult.create_and_validate(offroad.astype(jnp.float32), valid)

        return metric.replace(value=jnp.squeeze(metric.value, axis=-1), valid=jnp.squeeze(metric.valid, axis=-1))


class GokartDistanceToBoundsMetric(abstract_metric.AbstractMetric):
    """Distance to bounds metric.

    This metric returns the distance of the objects from the closest boundary (edge).
    If the object is offroad, the value can be forced to be a specific value (e.g. -1).
    given when the object is offroad (without considering the safety_margin)."""

    def __init__(self, offroad_value: Optional[float] = None):
        """Initializes the offroad metric.

        Args:
            offroad_value: additional reward given when the object is offroad (without
                considering the safety_margin).
        """
        assert isinstance(offroad_value, (float, int))
        assert offroad_value is None or offroad_value < 0
        self.offroad_value = offroad_value

    @jax.named_scope("GokartDistanceToBoundsMetric.compute")
    def compute(self, state: datatypes.SimulatorState) -> abstract_metric.MetricResult:
        """Computes the distance to bounds metric. The minimum distance to the boundary is used.

        Args:
          state: Updated simulator state to calculate metrics for. Will
            compute the offroad metric for timestep `state.timestep`.

        Returns:
          An array containing the metric result of the same shape as the input
            trajectories. The shape is (..., num_objects).
        """
        current_object_state = datatypes.dynamic_slice(
            state.sim_trajectory,
            state.timestep,
            1,
            -1,
        )
        distances = - compute_signed_distance_object_to_nearest_road_edge_point(
                current_object_state, state.roadgraph_points
        )
        # todo verify dimension here
        # If the value is negative, it means that the actor is offroad
        metric_value = jax.lax.cond(
            self.offroad_value is None,
            lambda x: jnp.where(distances<0, jnp.ones_like(distances)*self.offroad_value, distances),
            lambda x: x,
        )
        # todo select object of interest

        valid = jnp.ones_like(metric_value, dtype=jnp.bool_)
        metric = abstract_metric.MetricResult.create_and_validate(metric_value.astype(jnp.float32), valid)

        return metric.replace(value=jnp.squeeze(metric.value, axis=-1), valid=jnp.squeeze(metric.valid, axis=-1))
