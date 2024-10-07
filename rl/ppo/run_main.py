import dataclasses
from turtle import st

import jax
import wandb

from rl.ppo.config import VizConfig, PPOconfig
from rl.ppo.ppo_continuous_action import make_train

jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_disable_jit", True)
jax.config.update("jax_traceback_filtering", "off")

if __name__ == "__main__":
    config = PPOconfig()
    viz_cfg = VizConfig()
    # 1. Run training (modes "online", "offline" or "disabled")
    wandb.init(
            project="GokartRL-PPO",
            config=dataclasses.asdict(config),
            mode="online")
    # wandb.define_metric("reward/*", step_metric="train_step")
    # wandb.define_metric("loss/*", step_metric="train_step")
    # wandb.define_metric("debug/*", step_metric="train_step")
    # wandb.define_metric("eval/*", step_metric="eval_step")
    rng = jax.random.PRNGKey(config.SEED)
    # train_jit = jax.jit(make_train(config, viz_cfg))
    # out = train_jit(rng)
    train_function = make_train(config, viz_cfg)
    out = train_function(rng)
    wandb.finish()
