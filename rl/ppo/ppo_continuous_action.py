import dataclasses
import os
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import wandb
from flax.training.train_state import TrainState
from jax import Array
from jaxtyping import Float

from rl.ppo.config import PPOconfig
from rl.ppo.structures import ActorCritic, Transition
from rl.wrappers import WaymaxLogWrapper
from waymax import config as _config, datatypes, visualization
from waymax.agents import actor_core
from waymax.dynamics.tricycle_model import TricycleModel
from waymax.env import GokartRacingEnvironment
from waymax.utils.gokart_config import GoKartGeometry, TricycleParams, PajieckaParams
from waymax.utils.gokart_utils import create_init_state, create_batch_init_state

# from wrappers import (
#     LogWrapper,
#     BraxGymnaxWrapper,
#     VecEnv,
#     NormalizeVecObservation,
#     NormalizeVecReward,
#     ClipAction,
# )
from rl.wrappers import WaymaxLogWrapper, NormalizeVecObservation, NormalizeVecReward

import waymax
from waymax.env import GokartRacingEnvironment
from waymax import config as _config, datatypes
from waymax.dynamics.tricycle_model import TricycleModel
from waymax.utils.gokart_utils import create_init_state, create_batch_init_state
from waymax.utils.gokart_config import GoKartGeometry, TricycleParams, PajieckaParams
from waymax.agents import actor_core
from waymax import visualization

import pandas as pd
from datetime import datetime
import os

from waymax.dynamics.bicycle_model import InvertibleBicycleModel
from waymax.dynamics import StateDynamics
from waymax.env import PlanningAgentEnvironment
from waymax import agents

# TODO:
# 1. env.reset() returns a tuple of (observation, env_state)
# 2. env.step() returns a tuple of (observation, env_state, reward, done, info)
# 3. immediately reset the environment when offroad?    make it optional，not implemented yet
# 4. vectorize the environment 
# 5. check the implementation of the dynamics model
# 6. might need to improve the observe function
# 7. might need to improve the reward function
# 8. reset the environment individually
# 9. implement the debug modules

jax.config.update("jax_debug_nans", True)

df_log = pd.DataFrame(columns=["loss/total_loss", "loss/value_loss", "loss/loss_actor", "loss/entropy"])


# df_matrix = pd.DataFrame(columns=["matrix/obs"])

def make_train(config: PPOconfig, viz_cfg):
    config.NUM_UPDATES = (
            config.TOTAL_TIMESTEPS // config.NUM_STEPS // config.NUM_ENVS
    )
    config.MINIBATCH_SIZE = (
            config.NUM_ENVS * config.NUM_STEPS // config.NUM_MINIBATCHES
    )

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = f"data_log_{current_time}.csv"
    log_file_matrix = f"matrix_log_{current_time}.csv"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, config.ENV_NAME + "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = os.path.join(log_dir, log_file)
    log_path_matrix = os.path.join(log_dir, log_file_matrix)

    # env, env_params = BraxGymnaxWrapper(config.ENV_NAME"]), None
    # env = LogWrapper(env)
    # env = ClipAction(env)
    # env = VecEnv(env)
    # if config.NORMALIZE_ENV"]:
    #     env = NormalizeVecObservation(env)
    #     env = NormalizeVecReward(env, config.GAMMA)

    #TODO: (tian) env_factory with changed reward in config
    dynamics_model = TricycleModel(gk_geometry=GoKartGeometry(), model_params=TricycleParams(),
                                   paj_params=PajieckaParams(), dt=0.1)

    env = GokartRacingEnvironment(
            dynamics_model=dynamics_model,
            config=dataclasses.replace(
                    _config.EnvironmentConfig(),
                    max_num_objects=1,
                    init_steps=1  # => state.timestep = 0
            ),
    )
    dynamics_model = InvertibleBicycleModel()
    env = PlanningAgentEnvironment(
        dynamics_model=dynamics_model,
        config=dataclasses.replace(
                _config.EnvironmentConfig(),
                max_num_objects=32,
        ),
        sim_agent_actors=[agents.create_expert_actor(
                dynamics_model=StateDynamics(),
                is_controlled_func=lambda state: not state.object_metadata.is_sdc,
        )],
        sim_agent_params=[None],
    )
    env = WaymaxLogWrapper(env)
    # if config["NORMALIZE_ENV"]:
    #     env = NormalizeVecObservation(env)
    #     env = NormalizeVecReward(env, config["GAMMA"])

    def linear_schedule(count):
        frac = (1.0 - (count // (config.NUM_MINIBATCHES * config.UPDATE_EPOCHS)) / config.NUM_UPDATES)
        return config.LR * frac

    def train(rng):
        # INIT NETWORK
        network = ActorCritic(action_dim=3, activation=config.ACTIVATION)
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((config.NUM_OBS,))
        network_params = network.init(_rng, init_x)  # init_x used to determine input shape
        if config.ANNEAL_LR:
            tx = optax.chain(
                    optax.clip_by_global_norm(config.MAX_GRAD_NORM),
                    optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                    optax.clip_by_global_norm(config.MAX_GRAD_NORM),
                    optax.adam(config.LR, eps=1e-5),
            )

        # encapsulates the apply function, model parameters, and optimizer state
        train_state = TrainState.create(
                apply_fn=network.apply,  # used during training and evaluation to compute the output of the model
                params=network_params,
                tx=tx,
        )

        # INIT ENV
        env_state = create_batch_init_state(batch_size=config.NUM_ENVS)

        # rng, _rng = jax.random.split(rng)
        # reset_rng = jax.random.split(_rng, config.NUM_ENVS)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(env_state)

        # env_state, obsv = env.reset(env_state)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng, num_updates = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                value: Float[Array, "NUM_ENV"]
                last_obs: Float[Array, "NUM_ENV NUM_OBS"]
                pi, value = network.apply(train_state.params, last_obs)
                # raw_action = pi.sample(seed=_rng)
                raw_action: Float[Array, "NUM_ENV 3"]
                log_prob: Float[Array, "NUM_ENV"]
                raw_action, log_prob = pi.sample_and_log_prob(seed=_rng)
                action = jnp.clip(raw_action, -1.0, 1.0)  # shape [NUM_ENVS, 3]
                # jax.debug.print("action: {}", action)
                waymax_action = convert_to_waymaxaction(action)
                # log_prob = pi.log_prob(raw_action) # shape [NUM_ENVS]

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config.NUM_ENVS)
                # obsv, env_state, reward, done, info = env.step(
                #     env_state, waymax_action.action
                # )
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0))(
                        env_state, waymax_action.action
                )  # obsv [NUM_ENVS, NUM_OBS], reward [NUM_ENVS], done [NUM_ENVS]
                # transition:
                transition = Transition(
                        done, action, value, reward, log_prob, last_obs, info
                )
                # env_state, obsv = jax.vmap(env.post_step, in_axes=(0,0,0))(env_state, obsv, done) # reset the env if done
                runner_state = (train_state, env_state, obsv, rng, num_updates)
                return runner_state, transition

            # traj_batch is collection of Transition
            # traj_batch.action: Float[Array, "NUM_STEPS NUM_ENV 3"]
            runner_state, traj_batch = jax.lax.scan(
                    _env_step, runner_state, None, config.NUM_STEPS
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng, num_updates = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value  # gae: [NUM_ENVS], next_value: [NUM_ENVS]
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config.GAMMA * next_value * (1 - done) - value
                    gae = (
                            delta
                            + config.GAMMA * config.GAE_LAMBDA * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                        _get_advantages,
                        (jnp.zeros_like(last_val), last_val),
                        traj_batch,
                        reverse=True,
                        unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        # traj_batch.obs: Float[Array, "MINIBATCH_SIZE NUM_OBS"]
                        pi, value = network.apply(params, traj_batch.obs)  # shape [MINIBATCH_SIZE]
                        log_prob = pi.log_prob(traj_batch.action)  # retuen NaN !!!! check log prob function!

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                                value - traj_batch.value
                        ).clip(-config.CLIP_EPS, config.CLIP_EPS)
                        value_losses = jnp.square(value - targets)  # shape [MINIBATCH_SIZE]
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                                0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        last_log_prob = traj_batch.log_prob
                        # jax.debug.print("ratio: {}", ratio)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae  # shape [MINIBATCH_SIZE]
                        loss_actor2 = (
                                jnp.clip(
                                        ratio,
                                        1.0 - config.CLIP_EPS,
                                        1.0 + config.CLIP_EPS,
                                )
                                * gae
                        )
                        # jax.debug.print("Loss1 value: {}", loss_actor1)
                        # jax.debug.print("Loss2 value: {}", loss_actor2) # divided by 0??
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                                loss_actor
                                + config.VF_COEF * value_loss
                                - config.ENT_COEF * entropy
                        )
                        # jax.debug.print("Total loss {}", total_loss)
                        return total_loss, (
                            value_loss, loss_actor, entropy, traj_batch.action, last_log_prob, log_prob, ratio, gae,
                            value,
                            targets, traj_batch.obs, traj_batch.action)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (total_loss, aux_outputs), grads = grad_fn(
                            train_state.params, traj_batch, advantages, targets
                    )  # aux_outputs: (value_loss, loss_actor, entropy)
                    # jax.debug.print("total_loss: {}", total_loss)
                    # jax.debug.print("grads: {}", grads)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, (total_loss, aux_outputs)

                train_state, traj_batch, advantages, targets, rng = update_state  # e.g. traj_batch.obs: [NUM_STEPS, NUM_ENVS, NUM_OBS]
                rng, _rng = jax.random.split(rng)
                batch_size = config.MINIBATCH_SIZE * config.NUM_MINIBATCHES
                assert (
                        batch_size == config.NUM_STEPS * config.NUM_ENVS
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                        lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )  # shape [NUM_STEPS * NUM_ENVS = batch_size, ...]
                shuffled_batch = jax.tree_util.tree_map(
                        lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                        lambda x: jnp.reshape(
                                x, [config.NUM_MINIBATCHES, -1] + list(x.shape[1:])
                        ),
                        shuffled_batch,
                )  # shape[NUM_MINIBATCHES, MINIBATCH_SIZE, ...]
                train_state, (total_loss, aux_outputs) = jax.lax.scan(
                        _update_minbatch, train_state, minibatches
                )  # iteratively run _update_minbatch over minibatches along axis 0 (NUM_MINIBATCHES)
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, (total_loss, aux_outputs)

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, (loss_info, aux_outputs) = jax.lax.scan(
                    _update_epoch, update_state, None, config.UPDATE_EPOCHS
            )
            loss_info: Float[Array, "UPDATE_EPOCHS NUM_MINIBATCHES"]
            train_state = update_state[0]
            #
            obs_debug: Float[Array, "UPDATE_EPOCHS NUM_MINIBATCHES MINIBATCH_SIZE NUM_OBS"]
            value_loss, loss_actor, entropy, actions, last_log_prob, log_prob, ratio, gae, value_debug, targets_debug, obs_debug, action_debug = aux_outputs
            num_updates += 1

            # DEBUG
            index = jnp.argmin(obs_debug[..., -1])
            i, j, k = jnp.unravel_index(index, obs_debug[..., -1].shape)
            pos_yaw = obs_debug[i, j, k, :3]

            metric = traj_batch.info
            metric["train/params"] = train_state.params
            metric["iteration"] = num_updates
            metric["loss/total_loss"] = loss_info.mean()
            metric["loss/value_loss"] = value_loss.mean()
            metric["loss/loss_actor"] = loss_actor.mean()
            metric["loss/entropy"] = entropy.mean()
            metric["loss/max_action"] = jnp.max(actions)
            metric["loss/min_action"] = jnp.min(actions)
            metric["loss/last_log_prob"] = last_log_prob.mean()
            metric["loss/log_prob"] = log_prob.mean()
            metric["loss/ratio"] = ratio.mean()
            metric["loss/gae"] = gae.mean()
            metric["loss/value_debug"] = value_debug.mean()
            metric["loss/targets_debug"] = targets_debug.mean()
            metric["debug/diff_dist_proj"] = jnp.min(obs_debug[..., -2])
            metric["debug/proj_distances"] = jnp.min(obs_debug[..., -1])
            metric["debug/pos_yaw"] = pos_yaw
            metric["matrix/obs"] = obs_debug
            metric["matrix/action"] = action_debug
            rng = update_state[-1]
            if config.DEBUG:

                def callback(info):
                    global df_log
                    global df_matrix
                    # return_values = info["returned_episode_returns"][
                    #     info["returned_episode"]
                    # ]
                    # timesteps = (
                    #     info["timestep"][info["returned_episode"]] * config.NUM_ENVS
                    # )
                    # for t in range(len(timesteps)):
                    #     print(
                    #         f"global step={timesteps[t]}, episodic return={return_values[t]}"
                    #     )
                    # info["returned_episode_returns"] # shape [NUM_STEPS, NUM_ENVS] 
                    data_log = {
                        # "iteration": info["iteration"],
                        "reward/episode_return": info["returned_episode_returns"].mean(-1).reshape(-1)[-1],
                        "loss/total_loss"      : info["loss/total_loss"],
                        "loss/value_loss"      : info["loss/value_loss"],
                        "loss/loss_actor"      : info["loss/loss_actor"],
                        "loss/entropy"         : info["loss/entropy"],
                        "loss/max_action"      : info["loss/max_action"],
                        "loss/min_action"      : info["loss/min_action"],
                        "loss/last_log_prob"   : info["loss/last_log_prob"],
                        "loss/log_prob"        : info["loss/log_prob"],
                        "loss/ratio"           : info["loss/ratio"],
                        "loss/gae"             : info["loss/gae"],
                        "loss/value_debug"     : info["loss/value_debug"],
                        "loss/targets_debug"   : info["loss/targets_debug"],
                        "debug/diff_dist_proj" : info["debug/diff_dist_proj"],
                        "debug/proj_distances" : info["debug/proj_distances"],
                        "debug/pos_yaw"        : info["debug/pos_yaw"]
                    }
                #     data_matrix = {
                #         "matrix/obs": info["matrix/obs"],
                #         "matrix/action": info["matrix/action"]
                #     }
                    wandb.log(data_log, step=info["iteration"])
                    num_updates = int(info["iteration"])
                    # params = jax.device_get(train_state.params)
                    if num_updates % config.EVAL_FREQ == 0:
                        params = info["train/params"]
                        # params = jax.device_get(train_state.params)
                        imgs, _ = evaluate_policy(params, config.NUM_EVAL_STEPS, viz_cfg)
                        imgs_np = np.stack(imgs, axis=0)
                        wandb.log({
                            f"Iteration {num_updates}": wandb.Video(
                                    np.moveaxis(imgs_np, -1, 1),
                                    fps=10,
                                    format="gif",
                            )
                        })

                    new_row = pd.DataFrame([data_log])
                    df_log = pd.concat([df_log, new_row], ignore_index=True)
                    df_log.to_csv(log_path, index=False)

                #     new_matrix = pd.DataFrame([data_matrix])
                #     df_matrix = pd.concat([df_matrix, new_matrix], ignore_index=True)
                #     df_matrix.to_csv(log_path_matrix, index=False)

                # jax.debug.callback(lambda info: callback(info, df_log), metric)
                jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, rng, num_updates)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        num_updates = 0
        runner_state = (train_state, env_state, obsv, _rng, num_updates)
        runner_state, metric = jax.lax.scan(
                _update_step, runner_state, None, config.NUM_UPDATES
        )  # runner_state:final state after all iterations, metric: A collection of outputs from each iteration
        return {"runner_state": runner_state, "metrics": metric}

    return train


def evaluate_policy(params, num_eval_steps, viz_cfg):
    network = ActorCritic(action_dim=3, activation="tanh")

    # Re-initialize the environment
    dynamics_model = TricycleModel(
            gk_geometry=GoKartGeometry(),
            model_params=TricycleParams(),
            paj_params=PajieckaParams(),
            dt=0.1
    )
    eval_env = GokartRacingEnvironment(
            dynamics_model=dynamics_model,
            config=dataclasses.replace(
                    _config.EnvironmentConfig(),
                    max_num_objects=1,
                    init_steps=1
            ),
    )
    eval_env = WaymaxLogWrapper(eval_env)
    eval_state = create_init_state()
    imgs = []

    obs, eval_state = eval_env.reset(eval_state)

    total_reward = 0.0
    reward_list = [0]
    action_list = []
    for _ in range(num_eval_steps):
        pi, _ = network.apply(
                params,
                obs,
        )
        action = pi.mean()
        waymax_action = datatypes.Action(data=action, valid=jnp.array([True]))
        imgs.append(visualization.plot_simulator_state(eval_state.env_state, use_log_traj=False, viz_config=viz_cfg))
        obs, eval_state, reward, done, info = eval_env.step(eval_state, waymax_action)
        action_list.append(action)
        reward_list.append(reward.item())
        total_reward += reward

        if done:
            # print(f"episode reward: {total_reward}")
            # print(f"reward list: {reward_list}")
            # print(f"action list: {action_list}")
            break

    return imgs, total_reward


def convert_to_waymaxaction(action: Float[Array, "b A"]) -> actor_core.WaymaxActorOutput:
    # input: action of shape (batch_size, 3) 
    # here only one object is controlled
    action = datatypes.Action(
            data=action, valid=jnp.ones((action.shape[0], 1), dtype=jnp.bool_))
    return actor_core.WaymaxActorOutput(
            action=action,
            is_controlled=jnp.ones((action.shape[0], 1), dtype=jnp.bool_),
            actor_state=None)
