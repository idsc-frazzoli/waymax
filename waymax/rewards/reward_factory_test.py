from pprint import pprint


def test_check_registry():
    from .reward_factory import REWARDS_CONFIG2REWARD
    pprint(REWARDS_CONFIG2REWARD)