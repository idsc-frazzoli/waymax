import jax
from jax import numpy as jnp

from waymax import datatypes
from waymax.metrics import abstract_metric


class GetTargetMetric(abstract_metric.AbstractMetric):
    """Get target metric.

    This metric returns 1.0 if the controlled object finish a scenario
    without collisions and reach the final position of the logged history.
    """

    def __init__(self, threshold: float = 2.5):
        """
        Args:
            threshold: if the L2 distance between the controlled object's
                final position and the logged history's final position does
                not exceed this defined threshold, 1.0 is returned.
        """
        assert isinstance(threshold, (float, int))
        assert threshold>0
        self._threshold = threshold

    @jax.named_scope('GetTargetMetric.compute')
    def compute(
        self, simulator_state: datatypes.SimulatorState
    ) -> abstract_metric.MetricResult:
        
        current_object_state = datatypes.dynamic_slice(
            simulator_state.sim_trajectory,
            simulator_state.timestep,
            1,
            -1,
        )
        current_log_state = datatypes.dynamic_slice(
            simulator_state.log_trajectory,
            simulator_state.timestep,
            1,
            -1,
        )

        not_terminal_valid = jnp.ones_like(current_object_state.valid, dtype=jnp.bool_)
        not_terminal_result = jnp.zeros_like(not_terminal_valid, dtype=jnp.float32)

        final_dist = jnp.linalg.norm(current_object_state.xy - current_log_state.xy, axis=-1)
        terminal_result = (final_dist<=self._threshold).astype(jnp.float32)
        terminal_valid = current_object_state.valid & current_log_state.valid        
        
        result = jax.lax.cond(
            simulator_state.is_done,
            lambda _: terminal_result[..., 0],
            lambda _: not_terminal_result[..., 0],
            operand=None,
        )
        valid = jax.lax.cond(
            simulator_state.is_done,
            lambda _: terminal_valid[..., 0],
            lambda _: not_terminal_valid[..., 0],
            operand=None,
        )
        return abstract_metric.MetricResult.create_and_validate(
            result, valid
        )
