from typing import Optional, List, Sequence
import jax
from jax import numpy as jnp

from waymax import datatypes
from waymax.metrics import abstract_metric, MetricResult


class GokartActionNormMetric(abstract_metric.AbstractMetric):
    """Action metric.

    This metric returns a l norm of the action taken by the gokart.
    """

    def __init__(self, action_idxs: Optional[Sequence[int]] = None, l_ord: int = 2):
        """Initializes the action metric.

        Args:
            action_idxs: The indices of the actions to compute the metric for. If None, the metric is computed for all actions.
            l_ord: The order of the norm to compute. Default is 2.
        """
        assert isinstance(action_idxs, (type(None), Sequence))
        if action_idxs is not None:
            assert all(isinstance(action_idx, int) for action_idx in action_idxs)
        assert isinstance(l_ord, int)
        self.action_idxs = action_idxs if action_idxs is not None else slice(None)
        self.l_ord = l_ord

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
            jnp.linalg.norm(simulator_state.history_actions[0, self.action_idxs],
                            ord=self.l_ord, axis=-1, keepdims=True),
            jnp.ones(simulator_state.num_objects, dtype=jnp.bool_),
        )

        return reward.replace(value=jnp.squeeze(reward.value, axis=-1), valid=jnp.squeeze(reward.valid, axis=-1))
    
    
class GokartActionKernelMetric(GokartActionNormMetric):
    """Action metric.

    This metric returns a l kernel of the action taken by the gokart.
    """

    def __init__(self, action_idxs: Optional[Sequence[int]] = None, l_ord: int = 2):
        super().__init__(action_idxs, l_ord)

    @jax.named_scope("GokartActionKernelMetric.compute")
    def compute(self, simulator_state: datatypes.GoKartSimState) -> MetricResult:
        """Computes the action metric.

        Args:
          simulator_state: Updated simulator state to calculate metrics for. Will
            compute the action metric for timestep `simulator_state.timestep`.

        Returns:
          An array containing the metric result of the same shape as the input
            trajectories. The shape is (..., num_objects).
        """

        reward = super().compute(simulator_state)
        reward.replace(value=jnp.power(jnp.abs(reward.value), self.l_ord))

        return reward


class GokartActionRateNormMetric(abstract_metric.AbstractMetric):
    """Action metric.

    This metric returns a l norm of the action rate taken by the gokart.
    """

    def __init__(self, action_idxs: Optional[Sequence[int]] = None, l_ord: int = 2):
        """Initializes the action metric.

        Args:
            action_idxs: The indices of the actions to compute the metric for. If None, the metric is computed for all actions.
            l_ord: The order of the norm to compute. Default is 2.
        """
        assert isinstance(action_idxs, (type(None), Sequence))
        if action_idxs is not None:
            assert all(isinstance(action_idx, int) for action_idx in action_idxs)
        assert isinstance(l_ord, int)
        self.action_idxs = action_idxs if action_idxs is not None else slice(None)
        self.l_ord = l_ord

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

        reward = MetricResult.create_and_validate(
            jnp.linalg.norm(
                simulator_state.history_actions[0, self.action_idxs] - 
                simulator_state.history_actions[1, self.action_idxs],
                ord=self.l_ord,
                axis=-1,
                keepdims=True,
            ),
            jnp.ones(simulator_state.num_objects, dtype=jnp.bool_),
        )

        return reward.replace(value=jnp.squeeze(reward.value, axis=-1), valid=jnp.squeeze(reward.valid, axis=-1))

class GokartActionRateKernelMetric(GokartActionRateNormMetric):
    """Action metric.
    
    This metric returns a l kernel of the action rate taken by the gokart.
    """
    
    def __init__(self, action_idxs: Optional[Sequence[int]] = None, l_ord: int = 2):
        super().__init__(action_idxs, l_ord)
        
    @jax.named_scope("GokartActionRateKernelMetric.compute")
    def compute(self, simulator_state: datatypes.GoKartSimState) -> MetricResult:
        """Computes the action rate metric.
        
        Args:
          simulator_state: Updated simulator state to calculate metrics for. Will
            compute the action rate metric for timestep `simulator_state.timestep`.
        
        Returns:
          An array containing the metric result of the same shape as the input
            trajectories. The shape is (..., num_objects).
        """
        
        reward = super().compute(simulator_state)
        reward.replace(value=jnp.power(jnp.abs(reward.value), self.l_ord))
        
        return reward
    
    
class GokartActionTVKernelMetric(abstract_metric.AbstractMetric):
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

    @jax.named_scope("GokartActionTVKernelMetric.compute")
    def compute(self, simulator_state: datatypes.GoKartSimState) -> MetricResult:
        """Computes the action metric.

        Args:
          simulator_state: Updated simulator state to calculate metrics for. Will
            compute the TV action metric for timestep `simulator_state.timestep`.

        Returns:
          An array containing the metric result of the same shape as the input
            trajectories. The shape is (..., num_objects).
        """

        reward = MetricResult.create_and_validate(
            jnp.power(jnp.abs(jnp.array([simulator_state.history_actions[0, 2] - 
                                         simulator_state.history_actions[0, 1]])), self.l_ord),
            jnp.ones(simulator_state.num_objects, dtype=jnp.bool_),
        )

        return reward.replace(value=jnp.squeeze(reward.value, axis=-1), valid=jnp.squeeze(reward.valid, axis=-1))
