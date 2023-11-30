

from typing import Any
from flax import struct
from flax.training import train_state
from dataclasses import dataclass
from flax import linen as nn
import jax
import jax.numpy as jnp
from core_jax.envs.env import Env, EnvState

from core_jax.evaluators.evaluator import Evaluator, EvaluatorState
from core_jax.utils.replay_memory import EndRewardReplayBufferState


@
class TrainState(train_state.TrainState):
    batch_stats: Any


@dataclass
class TrainerConfig:
    pass


@struct.dataclass
class AgentState(CollectorState):
    # key: jax.random.PRNGKey
    # evaluator_state: EvaluatorState
    # env_state: EnvState
    # buff_state: EndRewardReplayBufferState
    train_state: TrainState

class Trainer(Collector):
    def __init__(self,
        config: TrainerConfig,
        env: Env,
        evaluator: Evaluator,
        model: nn.Module
    ):
        self.config = config
        self.env = env
        self.evaluator = evaluator
        self.model = model

    def init(self, key: jax.random.PRNGKey) -> AgentState:
        



