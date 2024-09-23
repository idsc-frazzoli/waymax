import dataclasses

from dynamics.tricycle_model import TricycleModel
from env import PlanningAgentEnvironment, GokartRacingEnvironment
from rl.ppo.config import PPOconfig
from utils.gokart_config import GoKartGeometry, PajieckaParams, TricycleParams
from waymax import config as _config


def get_environment(config: PPOconfig) -> PlanningAgentEnvironment:
    if config.ENV_NAME == "gokart":
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
    elif config.ENV_NAME == "waymax":

        # todo
    else:
        raise ValueError(f"Unsupported environment {config.ENV_NAME}")
    return env
