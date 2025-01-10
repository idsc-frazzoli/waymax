import jax
from jax import numpy as jnp

from waymax import datatypes
from waymax.metrics import abstract_metric


class MitigateOscillationMetric(abstract_metric.AbstractMetric):
    """Mitigate oscillation metric.

    This metric returns the product of the recent two actions if
    they have opposite signs.
    """

    @jax.named_scope('MitigateOscillationMetric.compute')
    def compute(
        self, simulator_state: datatypes.SimulatorState
    ) -> abstract_metric.MetricResult:
        # TODO (tian): not support multi-agent yet
        init_flag = (simulator_state.timestep==0)

        action = jnp.squeeze(
            simulator_state.current_action_history.data, axis=-2
        )
        prev_action = jax.lax.cond(
            init_flag,
            lambda _: jnp.zeros_like(action, dtype=jnp.float32),
            lambda _: jnp.squeeze(
                simulator_state.previous_action_history.data, axis=-2
            ),
            operand=None,
        )

        result = jnp.sum(
            jnp.minimum(action*prev_action, 0), axis=-1
        ).astype(jnp.float32)
        valid = jnp.ones_like(result, dtype=jnp.bool_)

        return abstract_metric.MetricResult.create_and_validate(
            result, valid
        )
