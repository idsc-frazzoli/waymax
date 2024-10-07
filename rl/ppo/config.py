from chex import dataclass


@dataclass
class PPOconfig:
    LR: float = 3e-4
    "learning rate"
    NUM_ENVS: int = 100
    "number of parallel environments"
    NUM_OBS: int = 2034
    "dimension of observations"
    NUM_STEPS: int = 5
    "Num steps * num envs = steps per update"
    TOTAL_TIMESTEPS: float = 1e6  # 5e7
    "total env steps = num envs * num steps * num updates"
    UPDATE_EPOCHS: int = 5  # 2
    "number of epochs per update"
    NUM_MINIBATCHES: int = 25  # 32
    "number of minibatches = num envs * num steps / minibatch size"
    EVAL_FREQ: int = 400
    "evaluate every EVAL_FREQ updates"
    NUM_EVAL_STEPS: int = 80
    "number of steps to evaluate"
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.95
    CLIP_EPS: float = 0.2
    ENT_COEF: float = 0.0
    VF_COEF: float = 0.5
    MAX_GRAD_NORM: float = 0.5
    ACTIVATION: str = "tanh"
    ENV_NAME: str = "waymax"
    ANNEAL_LR: bool = False
    NORMALIZE_ENV: bool = True
    DEBUG: bool = True
    SEED: int = 30


@dataclass
class VizConfig:
    front_x: float = 20.0
    back_x: float = 20.0
    front_y: float = 20.0
    back_y: float = 20.0
    px_per_meter: float = 15.0

"""
    config = {
        "LR"             : 3e-4,
        "NUM_ENVS"       : 100,  # number of parallel environments
        "NUM_OBS"        : 15,
        "NUM_STEPS"      : 4,  # num steps * num envs = steps per update
        "TOTAL_TIMESTEPS": 8e4,  # 5e7
        "UPDATE_EPOCHS"  : 2,  # 2
        "NUM_MINIBATCHES": 20,  # 32
        "EVAL_FREQ"      : 100,
        "NUM_EVAL_STEPS" : 100,
        "GAMMA"          : 0.99,
        "GAE_LAMBDA"     : 0.95,
        "CLIP_EPS"       : 0.2,
        "ENT_COEF"       : 0.0,
        "VF_COEF"        : 0.5,
        "MAX_GRAD_NORM"  : 0.5,
        "ACTIVATION"     : "tanh",
        "ENV_NAME"       : "hopper",
        "ANNEAL_LR"      : False,
        "NORMALIZE_ENV"  : True,
        "DEBUG"          : True,
    }
    viz_cfg = {
        "front_x"     : 20.0,
        "back_x"      : 20.0,
        "front_y"     : 20.0,
        "back_y"      : 20.0,
        "px_per_meter": 15.0
    }
    """