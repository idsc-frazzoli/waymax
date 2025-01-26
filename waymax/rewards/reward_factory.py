from typing import Type

from waymax import config as _config
from waymax.rewards import AbstractRewardFunction, LinearCombinationReward
from waymax.rewards.linear_transformed_reward import LinearTransformedReward
from waymax.rewards.lexicographic_reward import LexicographicReward

REWARDS_CONFIG2REWARD: dict[Type[_config.LinearCombinationRewardConfig], Type[AbstractRewardFunction]] = {
    _config.LinearCombinationRewardConfig: LinearCombinationReward,
    _config.LinearTransformedRewardConfig: LinearTransformedReward,
    _config.LexicographicRewardConfig: LexicographicReward,
}



def get_reward_function_from_config(config: _config.LinearCombinationRewardConfig) -> AbstractRewardFunction:
    """Returns the reward function based on the config."""
    reward_class = REWARDS_CONFIG2REWARD.get(type(config))
    if reward_class is None:
        raise ValueError(f"Unsupported reward config: {config}")
    return reward_class(config)