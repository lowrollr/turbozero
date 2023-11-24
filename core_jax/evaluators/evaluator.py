
from typing import Tuple
from chex import PRNGKey
from flax import struct
from dataclasses import dataclass
import jax.numpy as jnp


@dataclass
class EvaluatorConfig:
    name: str
    epsilon: float = 1e-8


@struct.dataclass
class EvaluatorState:
    pass

@struct.dataclass
class EnvState:
    pass


class Evaluator:
    def __init__(self,
        config: EvaluatorConfig,
        *args,
        **kwargs
    ):
        self.config = config
        self.args = args
        self.kwargs = kwargs

        self.state = EvaluatorState()
    
    def evaluate(self, env_state: EnvState) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # returns probability distribution over actions, and the value estimation for the current state
        raise NotImplementedError()
    
    def choose_action(self, evaluator_state: EvaluatorState, legal_actions_mask: jnp.ndarray):
        raise NotImplementedError()
    
    def update_evaluator_state(
            self, 
            evaluator_state: EvaluatorState, 
            actions: jnp.ndarray, 
            terminated: jnp.ndarray
        ) -> EvaluatorState:

        
        return evaluator_state
    