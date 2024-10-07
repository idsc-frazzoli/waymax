import dataclasses

from waymax.dynamics.tricycle_model import TricycleModel
from waymax.env import PlanningAgentEnvironment, GokartRacingEnvironment
from rl.ppo.config import PPOconfig
from waymax.utils.gokart_config import GoKartGeometry, PajieckaParams, TricycleParams
from waymax import config as _config


def get_environment(config: PPOconfig) -> PlanningAgentEnvironment:
    if config.ENV_NAME == "gokart":
        # env, env_params = BraxGymnaxWrapper(config.ENV_NAME"]), None
        # env = LogWrapper(env)
        # env = ClipAction(env)
        # env = VecEnv(env)
        # if config.NORMALIZE_ENV"]:
        #     env = NormalizeVecObservation(env)
        #     env = NormalizeVecReward(env, config.GAMMA)
        dynamics_model = TricycleModel(gk_geometry=GoKartGeometry(), model_params=TricycleParams(),
                                       paj_params=PajieckaParams(), dt=0.1, normalize_actions=True,)

        env = GokartRacingEnvironment(
                dynamics_model=dynamics_model,
                config=dataclasses.replace(
                        _config.EnvironmentConfig(),
                        max_num_objects=1,
                        init_steps=1  # => state.timestep = 0
                ),
        )
        # if config["NORMALIZE_ENV"]:
        #     env = NormalizeVecObservation(env)
        #     env = NormalizeVecReward(env, config["GAMMA"])
    elif config.ENV_NAME == "waymax":
        # todo
        raise NotImplementedError("Waymax environment not implemented yet")
    else:
        raise ValueError(f"Unsupported environment {config.ENV_NAME}")
    return env
