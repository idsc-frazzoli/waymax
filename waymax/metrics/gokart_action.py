from typing import Optional, List, Sequence
import jax
from jax import numpy as jnp
from jaxtyping import AbstractArray
from prometheus_client import Metric

from waymax import datatypes
from waymax.metrics import abstract_metric, MetricResult


class GokartActionMetric(abstract_metric.AbstractMetric):
    """Action metric.

    This metric returns a l kernel of the action taken by the gokart.
    """

    def __init__(self, action_names: Optional[Sequence[str]] = None, l_ord: int = 2):
        """Initializes the action metric.

        Args:
            action_names: The names of the actions to compute the metric for. If None, the metric is computed for all actions.
            l_ord: The order of the kernel to compute. Default is 2.
        """
        assert isinstance(action_names, (type(None), Sequence))
        assert not isinstance(action_names, str), "action_names should be a sequence of strings"
        if action_names is not None:
            assert all(isinstance(action_name, str) for action_name in action_names)
        assert isinstance(l_ord, int)
        if action_names is not None:
            self.action_names = action_names
        else:
            self.action_names = datatypes.GokartActionHistory.controllable_fields
        self.l_ord = l_ord

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
        print("history_actions", simulator_state.history_actions)
        print("simulator_state.last_action", simulator_state.last_action)
        reward = MetricResult.create_and_validate(
            jax.lax.cond(
                simulator_state.timestep > jnp.zeros_like(simulator_state.timestep),
                lambda x: jnp.sum(jnp.pow(jnp.abs(x), self.l_ord)),
                lambda x: 0.0,
                simulator_state.last_action.stack_fields(self.action_names)[..., 0, :],
            ),
            jnp.ones(simulator_state.num_objects, dtype=jnp.bool_).squeeze(-1),
        )

        return reward


class GokartActionRateMetric(abstract_metric.AbstractMetric):
    """Action metric.

    This metric returns a l kernel of the action rate taken by the gokart.
    """

    def __init__(self, action_names: Optional[Sequence[str]] = None, l_ord: int = 2):
        """Initializes the action metric.

        Args:
            action_names: The names of the actions to compute the metric for. If None, the metric is computed for all actions.
            l_ord: The order of the kernel to compute. Default is 2.
        """
        assert isinstance(action_names, (type(None), Sequence))
        assert not isinstance(action_names, str), "action_names should be a sequence of strings"
        if action_names is not None:
            assert all(isinstance(action_name, str) for action_name in action_names)
        assert isinstance(l_ord, int)
        self.action_names = (
            action_names if action_names is not None else datatypes.GokartActionHistory.controllable_fields
        )
        self.l_ord = l_ord

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

        action_attrs = simulator_state.last_N_actions(2).stack_fields(self.action_names)[..., :2, :]
        # jax.debug.print("ts {}, action_attrs {}", simulator_state.timestep, action_attrs)
        reward = MetricResult.create_and_validate(
            jax.lax.cond(
                simulator_state.timestep > jnp.ones_like(simulator_state.timestep),
                lambda x: jnp.sum(jnp.pow(jnp.abs(x[1] - x[0]), self.l_ord)),
                lambda x: 0.0,
                action_attrs,
            ),
            jnp.ones(simulator_state.num_objects, dtype=jnp.bool_).squeeze(-1),
        )

        return reward


class GokartActionTVMetric(abstract_metric.AbstractMetric):
    """TV metric.

    This metric returns a l norm of the TV taken by the gokart.
    TV (torque vectoring) is the difference between the right and left wheel accelerations:
    TV = AB_R - AB_L
    """

    def __init__(self, l_ord: int = 2):
        """Initializes the action metric.

        Args:
            action_idxs: The indices of the actions to compute the metric for. If None, the metric is computed for all actions.
            l_ord: The order of the norm to compute. Default is 2.
        """
        assert isinstance(l_ord, int)
        self.l_ord = l_ord

    @jax.named_scope("GokartActionTVMetric.compute")
    def compute(self, simulator_state: datatypes.GoKartSimState) -> MetricResult:
        """Computes the action metric.

        Args:
          simulator_state: Updated simulator state to calculate metrics for. Will
            compute the TV action metric for timestep `simulator_state.timestep`.

        Returns:
          An array containing the metric result of the same shape as the input
            trajectories. The shape is (..., num_objects).
        """

        TV_attr = simulator_state.last_action.TV[..., 0]
        reward = MetricResult.create_and_validate(
            jax.lax.cond(
                simulator_state.timestep > jnp.zeros_like(simulator_state.timestep),
                lambda x: jnp.pow(jnp.abs(x), self.l_ord).squeeze(-1),
                lambda x: 0.0,
                TV_attr,
            ),
            jnp.ones(simulator_state.num_objects, dtype=jnp.bool_).squeeze(-1),
        )

        return reward
