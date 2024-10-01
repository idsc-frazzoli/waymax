import dataclasses

import jax
import wandb

from rl.ppo.config import VizConfig, PPOconfig
from rl.ppo.ppo_continuous_action import make_train

if __name__ == "__main__":

    config = PPOconfig()
    # TODO: (tian) check if replaced
    dataclasses.replace(config, ENV_NAME="CarRacing-v0", )
    viz_cfg = VizConfig()
    wandb.init(project="RLwaymax-PPO", config=dataclasses.asdict(config))

    # TODO: (tian) check new way for generating random numbers
    rng = jax.random.PRNGKey(config.SEED)
    # train_jit = jax.jit(make_train(config, viz_cfg))
    # out = train_jit(rng)
    train_function = make_train(config, viz_cfg)
    out = train_function(rng)
    wandb.finish()