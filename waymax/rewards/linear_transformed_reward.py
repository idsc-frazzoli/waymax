import jax

from waymax import datatypes, metrics
from waymax.config import LinearTransformedRewardConfig, LinearCombinationRewardConfig
from waymax.rewards import LinearCombinationReward
import jax.numpy as jnp

class LinearTransformedReward(LinearCombinationReward):
  """Reward function that performs a linear combination of metrics.
  With the additional possibility of applying a custom transform to each metric.
  """

  def __init__(self, config: LinearTransformedRewardConfig):
    super().__init__(LinearCombinationRewardConfig(config.rewards))
    assert all(r in config.rewards  for r in config.transform)
    self._transform = config.transform

  def compute(
      self,
      simulator_state: datatypes.SimulatorState,
      action: datatypes.Action,
      agent_mask: jax.Array,
  ) -> jax.Array:
    """Computes the reward as a linear combination of metrics.

    Args:
      simulator_state: State of the Waymax environment.
      action: Action taken to control the agent(s) (..., num_objects,
        action_space).
      agent_mask: Binary mask indicating which agent inputs are valid (...,
        num_objects).

    Returns:
      An array of rewards, where there is one reward per agent
      (..., num_objects).
    """
    del action  # unused
    all_metrics = metrics.run_metrics(simulator_state, self._metrics_config)

    reward = jnp.zeros_like(agent_mask)
    for reward_metric_name, reward_weight in self._config.rewards.items():
      metric_all_agents = all_metrics[reward_metric_name].masked_value()
      metric = metric_all_agents * agent_mask
      reward += self._transform[reward_metric_name](metric) * reward_weight
    return reward