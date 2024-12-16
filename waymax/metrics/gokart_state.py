from typing import Sequence, Union

import jax
from jax import numpy as jnp

from waymax import datatypes
from waymax.metrics import abstract_metric, MetricResult


class GokartStateNormMetric(abstract_metric.AbstractMetric):
    """State metric.

    This metric returns the l-norm of the state of the gokart
    """

    def __init__(self, state_names: Union[str, Sequence[str]], ord: int = 2):
        """Initializes the state metric.

        Args:
            ord: The order of the norm to compute. Default is 2.
        """
        assert isinstance(state_names, (Sequence, str))
        if isinstance(state_names, str):
            state_names = [state_names, ]
        assert all(isinstance(state_name, str) for state_name in state_names)
        assert isinstance(ord, int)
        self._state_names: Sequence[str] = state_names
        self._ord: int = ord

    @jax.named_scope("GokartStateNormMetric.compute")
    def compute(self, simulator_state: datatypes.GoKartSimState) -> MetricResult:
        """Computes a state metric.

        Args:
          simulator_state: Updated simulator state to calculate metrics for a specific state. Will
            compute the state metric for timestep `simulator_state.timestep`.

        Returns:
          An array containing the metric result of the same shape as the input
            trajectories. The shape is (..., num_objects).
        """

        reward = MetricResult.create_and_validate(
            jnp.linalg.norm(simulator_state.current_sim_trajectory.stack_fields(self._state_names)[..., 0, :],
                            self._ord, axis=-1).squeeze(),
            jnp.ones(simulator_state.num_objects, dtype=jnp.bool_).squeeze(-1),
        )

        return reward

class GokartStateOutRangeMetric(abstract_metric.AbstractMetric):
    """State metric.

    This metric returns 1.0 if the state of the gokart is out of the given range.
    """

    def __init__(self, state_names: Union[str, Sequence[str]], min_value: float = -jnp.inf, max_value: float = jnp.inf):
        """Initializes the state metric.

        Args:
            state_names (Union[str, Sequence[str]]): The names of the states to compute the metric for.
            min_value (float): The minimum value of the states.
            max_value (float): The maximum value of the states.
        """
        assert isinstance(state_names, (str, Sequence))
        assert isinstance(min_value, (float, int))
        assert isinstance(max_value, (float, int))
        assert min_value < max_value
        if isinstance(state_names, str):
            state_names = [state_names]
        assert all(isinstance(state_name, str) for state_name in state_names)
        self._state_names: Sequence[str] = state_names
        self._min: float = min_value
        self._max: float = max_value

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
        state_attr = simulator_state.current_sim_trajectory.stack_fields(self._state_names)[..., 0, :]
        reward = MetricResult.create_and_validate(
            jnp.any(jnp.logical_or(jnp.less(state_attr, self._min), jnp.greater(state_attr, self._max))).astype(
                jnp.float32
            ).squeeze(),
            jnp.ones(simulator_state.num_objects, dtype=jnp.bool_).squeeze(-1),
        )

        return reward