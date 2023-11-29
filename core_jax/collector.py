


import jax


from flax import struct
from dataclasses import dataclass
from core.env import Env
from core_jax.envs.env import EnvState

from core_jax.evaluators.evaluator import Evaluator, EvaluatorState
from core_jax.utils.replay_memory import EndRewardReplayBufferState


@dataclass
class CollectorConfig:
    pass

@struct.dataclass
class CollectorState:
    key: jax.random.PRNGKey
    evaluator_state: EvaluatorState
    env_state: EnvState
    buff_state: EndRewardReplayBufferState


class Collector:
    def __init__(self, 
        config: CollectorConfig,
        env: Env,
        evaluator: Evaluator       
    ): 
        self.config = config
        self.env = env
        self.evaluator = evaluator