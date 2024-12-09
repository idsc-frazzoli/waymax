import jax
from jax import numpy as jnp

from waymax import datatypes
from waymax.metrics import abstract_metric, MetricResult


class GokartStateMetric(abstract_metric.AbstractMetric):
    """State metric.

    This metric returns a l kernel of a state of the gokart.
    """

    def __init__(self, state_name: str, l_ord: int = 2):
        """Initializes the state metric.

        Args:
            l_ord: The order of the kernel to compute. Default is 2.
        """
        assert isinstance(state_name, str)
        assert isinstance(l_ord, int)
        self.state_name = state_name
        self.l_ord = l_ord

    @jax.named_scope("GokartStateMetric.compute")
    def compute(self, simulator_state: datatypes.GoKartSimState) -> MetricResult:
        """Computes a state metric.

        Args:
          simulator_state: Updated simulator state to calculate metrics for a specific state. Will
            compute the state metric for timestep `simulator_state.timestep`.

        Returns:
          An array containing the metric result of the same shape as the input
            trajectories. The shape is (..., num_objects).
        """

        state_attr_curr = getattr(simulator_state.current_sim_trajectory, self.state_name)[..., 0, :]
        reward = MetricResult.create_and_validate(
            jnp.pow(jnp.abs(state_attr_curr), self.l_ord),
            jnp.ones(simulator_state.num_objects, dtype=jnp.bool_),
        )

        return reward.replace(value=jnp.squeeze(reward.value, axis=-1), valid=jnp.squeeze(reward.valid, axis=-1))

class GokartStateOutRangeMetric(abstract_metric.AbstractMetric):
    """State metric.

    This metric returns 1.0 if the state of the gokart is out of the given range.
    """

    def __init__(self, state_name: str, min_value: float = -jnp.inf, max_value: float = jnp.inf):
        """Initializes the state metric.

        Args:
            l_ord: The order of the norm to compute. Default is 2.
        """
        assert isinstance(state_name, str)
        assert isinstance(min_value, (float, int))
        assert isinstance(max_value, (float, int))
        assert min_value < max_value
        self.state_name = state_name
        self.min = min_value
        self.max = max_value

    @jax.named_scope("GokartStateOutRangeMetric.compute")
    def compute(self, simulator_state: datatypes.GoKartSimState) -> MetricResult:
        """Computes a state metric.

        Args:
          simulator_state: Updated simulator state to calculate metrics for a specific state. Will
            compute the state metric for timestep `simulator_state.timestep`.

        Returns:
          An array containing the metric result of the same shape as the input
            trajectories. The shape is (..., num_objects).
        """
        state_attr_curr = getattr(simulator_state.current_sim_trajectory, self.state_name)[..., 0, :]
        reward = MetricResult.create_and_validate(
            jnp.logical_or(jnp.less(state_attr_curr, self.min), jnp.greater(state_attr_curr, self.max)).astype(
                jnp.float32
            ),
            jnp.ones(simulator_state.num_objects, dtype=jnp.bool_),
        )

        return reward.replace(value=jnp.squeeze(reward.value, axis=-1), valid=jnp.squeeze(reward.valid, axis=-1))