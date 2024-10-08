import dataclasses

import jax.numpy as jnp

from waymax.dynamics.tricycle_model import TricycleModel
from waymax.env import PlanningAgentEnvironment, GokartRacingEnvironment
from waymax.env.waymax_environment import WaymaxDrivingEnvironment
from rl.ppo.config import PPOconfig
from waymax.utils.gokart_config import GoKartGeometry, PajieckaParams, TricycleParams
from waymax import config as _config
from waymax.dynamics.bicycle_model import InvertibleBicycleModel
from waymax.config import LinearCombinationRewardConfig
from waymax import agents
from waymax.dynamics import StateDynamics
from waymax.datatypes import SimulatorState
from waymax.utils.gokart_utils import create_batch_init_state
from waymax import dataloader
from waymax.utils.waymax_utils import replicate_init_state_to_form_batch


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
        dynamics_model = InvertibleBicycleModel()

        env = WaymaxDrivingEnvironment(
            dynamics_model=dynamics_model,
            config=dataclasses.replace(
                _config.EnvironmentConfig(),
                #TODO: (tian) add below setting into config
                max_num_objects=16,
                rewards=LinearCombinationRewardConfig(
                    rewards={'overlap': -1.0, 'offroad': -1.0, 'log_divergence': -1.0}
                ),
            ),
            sim_agent_actors=[agents.create_expert_actor(
                dynamics_model=StateDynamics(),
                is_controlled_func=lambda state: jnp.logical_not(state.object_metadata.is_sdc),
            )],
            sim_agent_params=[None],
        )
    else:
        raise ValueError(f"Unsupported environment {config.ENV_NAME}")
    return env

def init_environment(config: PPOconfig) -> SimulatorState:
    if config.ENV_NAME == "gokart":
        env_state = create_batch_init_state(batch_size=config.NUM_ENVS)
    elif config.ENV_NAME == "waymax":
        sce_config = dataclasses.replace(
                _config.WOD_1_1_0_VALIDATION, 
                path='d:/github repos for MT/waymax/dataset/validation/validation_tfexample.tfrecord-00050-of-00150', 
                max_num_objects=16,
                repeat=1 # set to 1 to test the length of the iterator, while set to None by default
            )
        data_iter = dataloader.simulator_state_generator(config=sce_config)
        for i in range(30):
            scenario = next(data_iter)
        env_state = replicate_init_state_to_form_batch(scenario, config.NUM_ENVS)
    else:
        raise ValueError(f"Unsupported environment {config.ENV_NAME}")
    return env_state
