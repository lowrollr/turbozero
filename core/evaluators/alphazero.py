
from dataclasses import dataclass
from typing import Tuple
from flax import struct, linen as nn
import jax
import jax.numpy as jnp
from core.envs.env import Env, EnvState
from core.evaluators.mcts import MCTS, MCTSConfig, MCTSState
from core.evaluators.nn_evaluator import NNEvaluator

@dataclass
class AlphaZeroConfig(MCTSConfig):
    budget: int
    temperature: float

@struct.dataclass
class AlphaZeroState(MCTSState):
    pass

class AlphaZero(MCTS, NNEvaluator):
    def __init__(self, 
        env: Env,
        config: AlphaZeroConfig,
        model: nn.Module,
        **kwargs
    ):
        super().__init__(env=env, config=config, model=model, **kwargs)
        
        self.config: AlphaZeroConfig
    
    
    def evaluate_leaf(self, 
        state: AlphaZeroState, 
        observation: struct.PyTreeNode,
        model_params: struct.PyTreeNode,
        **kwargs
    ) -> Tuple[AlphaZeroState, jnp.ndarray, jnp.ndarray]:
        policy, evaluation = self.predict(observation, model_params)
        return state, policy.squeeze(axis=0), evaluation.squeeze(axis=0)
    
    def choose_action(self, 
        state: AlphaZeroState,
        env_state: EnvState
    ) -> Tuple[AlphaZeroState, jnp.ndarray]:
        if self.config.temperature > 0:
            rand_key, new_key = jax.random.split(state.key)
            policy = self.get_policy(state)
            action = jax.random.choice(rand_key, len(policy), p=policy)
            return state.replace(key=new_key), action
        else:
            action = jnp.argmax(state.n_vals[1])
            return state, action

    def evaluate(self, 
        state: AlphaZeroState, 
        env_state: EnvState,
        model_params: struct.PyTreeNode,
        **kwargs
    ) -> AlphaZeroState:
        return super().evaluate(state, env_state, num_iters=self.config.budget, model_params=model_params, **kwargs)

    def get_policy(self, state: AlphaZeroState) -> jnp.ndarray:
        action_visits = state.n_vals[1]
        action_visits = action_visits ** (1/self.config.temperature)
        return action_visits / action_visits.sum()