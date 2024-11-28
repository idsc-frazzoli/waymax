import jax
from jax import numpy as jnp

from waymax import datatypes
from waymax.metrics import abstract_metric, MetricResult
from waymax.metrics.roadgraph import is_offroad


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
        self.safety_margin = safety_margin

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
        offroad = is_offroad(current_object_state, state.roadgraph_points, self.safety_margin)
        valid = jnp.ones_like(offroad, dtype=jnp.bool_)
        metric = abstract_metric.MetricResult.create_and_validate(offroad.astype(jnp.float32), valid)

        return metric.replace(value=jnp.squeeze(metric.value, axis=-1), valid=jnp.squeeze(metric.valid, axis=-1))
