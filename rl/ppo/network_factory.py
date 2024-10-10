from rl.ppo.config import PPOconfig
from rl.ppo.structures import ActorCritic


def get_network(config: PPOconfig) -> ActorCritic:
    if config.ENV_NAME == "gokart":
        network = ActorCritic(action_dim=3, activation=config.ACTIVATION)
    elif config.ENV_NAME == "waymax":
        network = ActorCritic(action_dim=2, activation=config.ACTIVATION)
    else:
        raise ValueError(f"Unsupported environment {config.ENV_NAME}")
    return network
