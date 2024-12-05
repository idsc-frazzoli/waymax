from turtle import distance
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


class GokartDistanceToBoundsMetric(abstract_metric.AbstractMetric):
    """Distance to bounds metric.

    This metric returns 0 if the object is farther than safety_margin from the boundary, and
    (1-distance_bound/safety_margin)**2 if it is closer, with a maximum value of 1 if the object is
    at the boundary. Moreover, an additional reward can be given when the object is offroad
    (without considering the safety_margin).  """

    def __init__(self, safety_margin: float = 0.0, additional_offroad_reward: float = 0.0):
        """Initializes the offroad metric.

        Args:
            safety_margin: the metric will be 0 if the object is farther than this distance from the boundary.
                Otherwise, it will be (1-distance_bound/safety_margin)**2, with a maximum value of 1 if the object is
                at the boundary.
            additional_offroad_reward: additional reward given when the object is offroad (without
                considering the safety_margin).
        """
        assert isinstance(safety_margin, (float, int))
        assert isinstance(additional_offroad_reward, (float, int))
        self.safety_margin = safety_margin
        self.additional_offroad_reward = additional_offroad_reward

    @jax.named_scope("GokartOffroadMetric.compute")
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
        signed_distances = is_offroad(current_object_state, state.roadgraph_points, self.safety_margin, return_mask=False)
        offroad = jnp.any(signed_distances > 0.0, axis=-1)
        # If the value is negative, it means that the actor is on the correct side of the road, if it is positive, it is
        # considered `offroad`.
        min_distance = jnp.expand_dims(jnp.min(jnp.abs(signed_distances.clip(max=0.0))), axis=0)
        metric_value = jax.lax.cond(
            min_distance[0] < self.safety_margin,
            lambda x: ((1 - x/self.safety_margin) ** 2),
            lambda x: jnp.zeros_like(x),
            min_distance,
        ) + offroad * self.additional_offroad_reward
        valid = jnp.ones_like(metric_value, dtype=jnp.bool_)
        metric = abstract_metric.MetricResult.create_and_validate(metric_value.astype(jnp.float32), valid)

        return metric.replace(value=jnp.squeeze(metric.value, axis=-1), valid=jnp.squeeze(metric.valid, axis=-1))