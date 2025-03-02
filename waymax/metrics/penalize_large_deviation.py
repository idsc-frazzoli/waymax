import jax
from jax import numpy as jnp

from waymax import datatypes
from waymax.metrics import abstract_metric


class PenalizeLargeDeviationMetric(abstract_metric.AbstractMetric):
    """A toy and not so realistic metric for testing.

    This metric computes the L2 distance between the controlled object's XY
    location and its nearest position in the logged history, and returns 1.0
    if the L2 distance exceeds a given threshold.
    """

    def __init__(self, threshold: float = 2.2):
        assert isinstance(threshold, (float, int))
        assert threshold>0
        self._threshold = threshold

    @jax.named_scope('PenalizeLargeDeviationMetric.compute')
    def compute(
        self, simulator_state: datatypes.SimulatorState
    ) -> abstract_metric.MetricResult:
        current_object_state = datatypes.dynamic_slice(
            simulator_state.sim_trajectory,
            simulator_state.timestep,
            1,
            -1,
        )
        log_state = simulator_state.log_trajectory

        dist = jnp.linalg.norm(current_object_state.xy - log_state.xy, axis=-1)
        min_dist = jnp.min(dist, axis=-1)
        result = (min_dist>self._threshold).astype(jnp.float32)
        valid = current_object_state.valid[...,0]

        return abstract_metric.MetricResult.create_and_validate(
            result, valid
        )
