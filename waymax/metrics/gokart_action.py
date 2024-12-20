from typing import Optional, Sequence, Union

import jax
from jax import numpy as jnp

from waymax import datatypes
from waymax.metrics import abstract_metric, MetricResult


class GokartActionNormMetric(abstract_metric.AbstractMetric):
    """Action metric.

    This metric returns the l-norm of the action taken by the gokart at time t.
    """

    def __init__(self, action_names: Optional[Union[str, Sequence[str]]] = None, ord: int = 2):
        """Initializes the action metric.

        Args:
            action_names: The names of the actions to compute the metric for. If None, the metric is computed for all actions.
            ord: The order of the norm to compute. Default is 2.
        """
        assert isinstance(action_names, (type(None), Sequence, str))
        if isinstance(action_names, str):
            action_names = [action_names]
        if action_names is not None:
            assert all(isinstance(action_name, str) for action_name in action_names)
        else:
            action_names = datatypes.GokartAction.action_fields
        assert isinstance(ord, int)
        self._action_names: Sequence[str] = action_names
        self._ord: int = ord

    @jax.named_scope("GokartActionNormMetric.compute")
    def compute(self, simulator_state: datatypes.GoKartSimState) -> MetricResult:
        """Computes the action metric.

        Args:
          simulator_state: Updated simulator state to calculate metrics for. Will
            compute the action metric for timestep `simulator_state.timestep`.

        Returns:
          An array containing the metric result of the same shape as the input
            trajectories. The shape is (..., num_objects).
        """

        reward = MetricResult.create_and_validate(
            jnp.linalg.norm(simulator_state.current_action_history.stack_fields(self._action_names)[:, 0, :],
                            self._ord, axis=-1).squeeze(),
            jnp.ones(simulator_state.num_objects, dtype=jnp.bool_).squeeze(-1),
        )

        return reward


class GokartActionRateNormMetric(abstract_metric.AbstractMetric):
    """Action metric.

    This metric returns the l-Minkowski-Distance of the action taken by the gokart at time t and t-1
    """

    def __init__(self, action_names: Optional[Union[str, Sequence[str]]] = None, ord: int = 2):
        """Initializes the action metric.

        Args:
            action_names: The names of the actions to compute the metric for. If None, the metric is computed for all actions.
            ord: The order of the metric to compute. Default is 2.
        """
        assert isinstance(action_names, (type(None), Sequence, str))
        if isinstance(action_names, str):
            action_names = [action_names]
        if action_names is not None:
            assert all(isinstance(action_name, str) for action_name in action_names)
        else:
            action_names = datatypes.GokartAction.action_fields
        assert isinstance(ord, int)
        self._action_names: Sequence[str] = action_names
        self._ord: int = ord

    @jax.named_scope("GokartActionRateNormMetric.compute")
    def compute(self, simulator_state: datatypes.GoKartSimState) -> MetricResult:
        """Computes the action rate metric.

        Args:
          simulator_state: Updated simulator state to calculate metrics for. Will
            compute the action rate metric for timestep `simulator_state.timestep`.

        Returns:
          An array containing the metric result of the same shape as the input
            trajectories. The shape is (..., num_objects).
        """
        
        curr_action_history = simulator_state.current_action_history.stack_fields(self._action_names)
        prev_action_history = simulator_state.previous_action_history.stack_fields(self._action_names)
        rate = curr_action_history - prev_action_history
                
        reward = MetricResult.create_and_validate(
            jax.lax.cond(
                simulator_state.timestep > jnp.zeros_like(simulator_state.timestep),
                lambda x: jnp.linalg.norm(x, self._ord, axis=-1).squeeze(),
                lambda x: 0.0,
                rate[..., 0, :],
            ),
            jnp.ones(simulator_state.num_objects, dtype=jnp.bool_).squeeze(-1),
        )

        return reward


class GokartTVActionNormMetric(abstract_metric.AbstractMetric):
    """TV metric.

    This metric returns the l-norm of the TV action taken by the gokart
    
    TV (torque vectoring) is the difference between the right and left wheel accelerations:
    TV = acc_right - acc_left
    """

    def __init__(self, ord: int = 2):
        """Initializes the action metric.

        Args:
            ord: The order of the norm to compute. Default is 2.
        """
        assert isinstance(ord, int)
        self._ord: int = ord

    @jax.named_scope("GokartTVActionNormMetric.compute")
    def compute(self, simulator_state: datatypes.GoKartSimState) -> MetricResult:
        """Computes the action metric.

        Args:
          simulator_state: Updated simulator state to calculate metrics for. Will
            compute the TV action metric for timestep `simulator_state.timestep`.

        Returns:
          An array containing the metric result of the same shape as the input
            trajectories. The shape is (..., num_objects).
        """

        tv = simulator_state.current_action_history.torque_vectoring
        
        reward = MetricResult.create_and_validate(jnp.linalg.norm(tv, self._ord, axis=-1).squeeze(),
            jnp.ones(simulator_state.num_objects, dtype=jnp.bool_).squeeze(-1),
        )

        return reward


class GokartActionOutRangeMetric(abstract_metric.AbstractMetric):
    """Action metric.

    This metric returns 1.0 if the action of the gokart is out of the given range.
    """

    def __init__(self, action_names: Optional[Union[str, Sequence[str]]] = None, min_value: float = -jnp.inf, max_value: float = jnp.inf):
        """Initializes the action metric.

        Args:
            action_names (Union[str, Sequence[str]]): The names of the actions to compute the metric for.
            min_value (float): The minimum value of the actions.
            max_value (float): The maximum value of the actions.
        """
        assert isinstance(action_names, (type(None), str, Sequence))
        assert isinstance(min_value, (float, int))
        assert isinstance(max_value, (float, int))
        assert min_value < max_value            
        if isinstance(action_names, str):
            action_names = [action_names]
        if action_names is not None:
            assert all(isinstance(action_name, str) for action_name in action_names)
        else:
            action_names = datatypes.GokartAction.action_fields
        self._action_names: Sequence[str] = action_names
        self._min: float = min_value
        self._max: float = max_value

    @jax.named_scope("GokartActionOutRangeMetric.compute")
    def compute(self, simulator_state: datatypes.GoKartSimState) -> MetricResult:
        """Computes an action metric.

        Args:
          simulator_state: Updated simulator state to calculate metrics for a specific state. Will
            compute the state metric for timestep `simulator_state.timestep`.

        Returns:
          An array containing the metric result of the same shape as the input
            trajectories. The shape is (..., num_objects).
        """
        action_attr = simulator_state.current_action_history.stack_fields(self._action_names)[..., 0, :]
        reward = MetricResult.create_and_validate(
            jnp.any(jnp.logical_or(jnp.less(action_attr, self._min), jnp.greater(action_attr, self._max))).astype(
                jnp.float32
            ).squeeze(),
            jnp.ones(simulator_state.num_objects, dtype=jnp.bool_).squeeze(-1),
        )

        return reward