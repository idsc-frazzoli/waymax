import dataclasses

import jax
import wandb

from rl.ppo.config import VizConfig, PPOconfig
from rl.ppo.ppo_continuous_action import make_train

if __name__ == "__main__":

    config = PPOconfig()
    viz_cfg = VizConfig()
    wandb.init(project="GokartRL-PPO", config=dataclasses.asdict(config))

    rng = jax.random.PRNGKey(config.SEED)
    # train_jit = jax.jit(make_train(config, viz_cfg))
    # out = train_jit(rng)
    train_function = make_train(config, viz_cfg)
    out = train_function(rng)
    wandb.finish()