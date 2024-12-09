from typing import Optional, Sequence

import jax
from jax import numpy as jnp

from waymax import datatypes
from waymax.metrics import abstract_metric, MetricResult


class GokartActionMetric(abstract_metric.AbstractMetric):
    """Action metric.

    This metric returns the l-power of the l-norm of the action taken by the gokart
    (||A||_l)^l = [\sum_i abs(a_i)^l]
    """

    def __init__(self, action_names: Optional[Sequence[str]] = None, l_ord: int = 2):
        """Initializes the action metric.

        Args:
            action_names: The names of the actions to compute the metric for. If None, the metric is computed for all actions.
            l_ord: The order of the metric to compute. Default is 2.
        """
        import numpy as np
        np.linalg.norm(jnp.array([1, 2, 3]), ord=2)
        assert isinstance(action_names, (type(None), Sequence))
        assert not isinstance(action_names, str), "action_names should be a sequence of strings"
        if action_names is not None:
            assert all(isinstance(action_name, str) for action_name in action_names)
        assert isinstance(l_ord, int)
        self._action_names = action_names
        self._l_ord = l_ord

    @jax.named_scope("GokartActionMetric.compute")
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
            jax.lax.cond(
                simulator_state.timestep > jnp.zeros_like(simulator_state.timestep),
                lambda x: jnp.sum(jnp.pow(jnp.abs(x), self._l_ord)),
                lambda x: 0.0,
                simulator_state.prev_actions(self._action_names, 1),
            ),
            jnp.ones(simulator_state.num_objects, dtype=jnp.bool_).squeeze(-1),
        )

        return reward


class GokartActionRateMetric(abstract_metric.AbstractMetric):
    """Action metric.

    This metric returns the l-power of the l-Minkowski-Distance of the action taken by the gokart at time t and t-1
    (||A_t-A_{t-1}||_l)^l = [\sum_i abs(a_{i,t}-a{i,t-1})^l]
    """

    def __init__(self, action_names: Optional[Sequence[str]] = None, l_ord: int = 2):
        """Initializes the action metric.

        Args:
            action_names: The names of the actions to compute the metric for. If None, the metric is computed for all actions.
            l_ord: The order of the metric to compute. Default is 2.
        """
        assert isinstance(action_names, (type(None), Sequence))
        assert not isinstance(action_names, str), "action_names should be a sequence of strings"
        if action_names is not None:
            assert all(isinstance(action_name, str) for action_name in action_names)
        assert isinstance(l_ord, int)
        self._action_names = action_names
        self._l_ord = l_ord

    @jax.named_scope("GokartActionRateMetric.compute")
    def compute(self, simulator_state: datatypes.GoKartSimState) -> MetricResult:
        """Computes the action rate metric.

        Args:
          simulator_state: Updated simulator state to calculate metrics for. Will
            compute the action rate metric for timestep `simulator_state.timestep`.

        Returns:
          An array containing the metric result of the same shape as the input
            trajectories. The shape is (..., num_objects).
        """

        reward = MetricResult.create_and_validate(
            jax.lax.cond(
                simulator_state.timestep > jnp.ones_like(simulator_state.timestep),
                lambda x: jnp.sum(jnp.pow(jnp.abs(x[..., 1, :] - x[..., 0, :]), self._l_ord)),
                lambda x: 0.0,
                simulator_state.prev_actions(self._action_names, 2),
            ),
            jnp.ones(simulator_state.num_objects, dtype=jnp.bool_).squeeze(-1),
        )

        return reward


class GokartTVActionMetric(abstract_metric.AbstractMetric):
    """TV metric.

    This metric returns the l-power of the l-norm of the TV action taken by the gokart
    (||A||_l)^l = [\sum_i abs(a_i)^l]
    
    TV (torque vectoring) is the difference between the right and left wheel accelerations:
    TV = acc_right - acc_left
    """

    def __init__(self, l_ord: int = 2):
        """Initializes the action metric.

        Args:
            action_idxs: The indices of the actions to compute the metric for. If None, the metric is computed for all actions.
            l_ord: The order of the metric to compute. Default is 2.
        """
        assert isinstance(l_ord, int)
        self._l_ord = l_ord

    @jax.named_scope("GokartTVActionMetric.compute")
    def compute(self, simulator_state: datatypes.GoKartSimState) -> MetricResult:
        """Computes the action metric.

        Args:
          simulator_state: Updated simulator state to calculate metrics for. Will
            compute the TV action metric for timestep `simulator_state.timestep`.

        Returns:
          An array containing the metric result of the same shape as the input
            trajectories. The shape is (..., num_objects).
        """

        prev_action = simulator_state.prev_actions(["acc_left", "acc_right"], 1)
        reward = MetricResult.create_and_validate(
            jax.lax.cond(
                simulator_state.timestep > jnp.zeros_like(simulator_state.timestep),
                lambda x: jnp.pow(jnp.abs(x), self._l_ord),
                lambda x: 0.0,
                (prev_action[..., 1] - prev_action[..., 0]).squeeze(),
            ),
            jnp.ones(simulator_state.num_objects, dtype=jnp.bool_).squeeze(-1),
        )

        return reward
