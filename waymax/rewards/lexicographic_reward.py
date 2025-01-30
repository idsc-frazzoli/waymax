"""Reward functions with lexicographic order."""
import jax
import jax.numpy as jnp

from waymax import datatypes, metrics
from waymax.config import LexicographicRewardConfig, LinearCombinationRewardConfig
from waymax.rewards import LinearCombinationReward


class LexicographicReward(LinearCombinationReward):
  """Reward function that includes a lexicographic order for all metrics."""

  def __init__(self, config: LexicographicRewardConfig):
    super().__init__(LinearCombinationRewardConfig(config.rewards))
    
    metrics_with_hierarchy = {}
    for hierarchy, rule in enumerate(config.hierarchy):
      for metric in rule:
        metrics_with_hierarchy[metric] = hierarchy
    assert all(r in metrics_with_hierarchy for r in config.rewards)

    self._metrics_with_hierarchy = metrics_with_hierarchy
    self._num_hierarchies = len(config.hierarchy)

  def compute(
      self,
      simulator_state: datatypes.SimulatorState,
      action: datatypes.Action,
      agent_mask: jax.Array,
  ) -> jax.Array:
    """
    Formats the reward as a vector according to the hierarchy, where
    each element is a linear combination of metrics.

    Args:
      simulator_state: State of the Waymax environment.
      action: Action taken to control the agent(s) (..., num_objects,
        action_space).
      agent_mask: Binary mask indicating which agent inputs are valid (...,
        num_objects).

    Returns:
      An array of rewards, where there is one reward vector per agent
      (..., num_objects, num_hierarchies) and the most important hierarchy
      is at the beginning.
    """
    del action  # unused
    all_metrics = metrics.run_metrics(simulator_state, self._metrics_config)

    reward = jnp.zeros(agent_mask.shape+(self._num_hierarchies,), dtype=jnp.float32)
    for reward_metric_name, reward_weight in self._config.rewards.items():
      metric_all_agents = all_metrics[reward_metric_name].masked_value()
      metric = metric_all_agents * agent_mask
      reward = reward.at[...,self._metrics_with_hierarchy[reward_metric_name]].add(metric * reward_weight)
    return reward
