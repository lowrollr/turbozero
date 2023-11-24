
from typing import Tuple
from flax import struct
from dataclasses import dataclass
import jax
import jax.numpy as jnp

from core_jax.envs.env import Env


@dataclass
class EvaluatorConfig:
    epsilon: float = 1e-8


@struct.dataclass
class EvaluatorState:
    key: jax.random.PRNGKey

class Evaluator:
    def __init__(self,
        config: EvaluatorConfig,
        *args,
        **kwargs
    ):
        self.config = config
        self.args = args
        self.kwargs = kwargs

    def reset(self, key: jax.random.PRNGKey) -> EvaluatorState:
        return EvaluatorState(key=key)
    
    def evaluate(self, 
        evaluator_state: EvaluatorState, 
        env: Env, 
        env_state: struct.PyTreeNode, 
        observation: struct.PyTreeNode, 
        *args
    ) -> Tuple[EvaluatorState, jnp.ndarray, jnp.ndarray]:
        # returns policy logits, and value estimation for the current state
        key, _ = jax.random.split(evaluator_state.key)
        return (
            evaluator_state,
            jax.random.normal(key, (*observation.action_mask.shape,)),
            jnp.zeros((1,))
        )
    
    def choose_action(self, 
        evaluator_state: EvaluatorState, 
        env: Env,
        env_state: struct.PyTreeNode,
        observation: struct.PyTreeNode,
        policy_logits: jnp.ndarray    
    ):
        return env.get_random_legal_action(
            env_state,
            observation
        )
    
    def step_evaluator(
            self, 
            evaluator_state: EvaluatorState, 
            actions: jnp.ndarray, 
            terminated: jnp.ndarray
        ) -> EvaluatorState:

        return evaluator_state
    