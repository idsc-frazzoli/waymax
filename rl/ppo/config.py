from dataclasses import field

from chex import dataclass


@dataclass
class PPOconfig:
    LR: float = 3e-4
    "learning rate"
    MAX_EPISODE_LENGTH: int = 300
    "number of max steps per episode"
    NUM_ENVS: int = 100
    "number of parallel environments"
    NUM_OBS: int = 11
    "dimension of observations"
    NUM_STEPS: int = 5
    "Num steps * num envs = steps per update (i.e. length of trajectory)"
    TOTAL_TIMESTEPS: int = 2e5  # 5e7
    "total env steps = num envs * num steps * num updates"
    UPDATE_EPOCHS: int = 5  # 2
    "number of epochs per update"
    NUM_MINIBATCHES: int = 25  # 32
    "number of minibatches = num envs * num steps / minibatch size"
    EVAL_FREQ: int = 200
    "evaluate every EVAL_FREQ updates"
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.95
    "generalized advantage estimation lambda"
    CLIP_EPS: float = 0.2
    "clip epsilon in ppo loss"
    ENT_COEF: float = 0.0
    "entropy coefficient in loss function"
    VF_COEF: float = 0.5
    "value function coefficient in loss function"
    MAX_GRAD_NORM: float = 0.5
    "max gradient norm"
    ACTIVATION: str = "tanh"
    ENV_NAME: str = "gokart"
    ANNEAL_LR: bool = False
    NORMALIZE_ENV: bool = True
    DEBUG: bool = True
    SEED: int = 30
    NUM_UPDATES: int = field(init=False)
    MINIBATCH_SIZE: int = field(init=False)
    BATCH_SIZE: int = field(init=False)

    def __post_init__(self):
        self.NUM_UPDATES = self.TOTAL_TIMESTEPS // self.NUM_STEPS // self.NUM_ENVS
        self.MINIBATCH_SIZE = self.NUM_ENVS * self.NUM_STEPS // self.NUM_MINIBATCHES
        self.BATCH_SIZE = self.MINIBATCH_SIZE * self.NUM_MINIBATCHES
        assert self.BATCH_SIZE == self.NUM_STEPS * self.NUM_ENVS, \
            "batch size must be equal to number of steps * number of envs"


@dataclass
class VizConfig:
    front_x: float = 20.0
    back_x: float = 20.0
    front_y: float = 20.0
    back_y: float = 20.0
    px_per_meter: float = 15.0
