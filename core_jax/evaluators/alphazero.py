





from dataclasses import dataclass
from typing import Tuple
from flax import struct, linen as nn
import jax
import jax.numpy as jnp
from core_jax.envs.env import Env, EnvState
from core_jax.evaluators.mcts import MCTS, MCTSConfig, MCTSState

@dataclass
class AlphaZeroConfig(MCTSConfig):
    mcts_iters: int
    temperature: float


class AlphaZero(MCTS):
    def __init__(self, 
        config: AlphaZeroConfig,
        observation_shape: Tuple[int, ...], 
        policy_shape: Tuple[int, ...],
        num_players: int,
        model: nn.Module
    ):
        super().__init__(config, policy_shape, num_players)
        self.model = model
        self.config: AlphaZeroConfig
        self.observation_shape = observation_shape
    
    def evaluate_leaf(self, 
        env: Env, 
        state: MCTSState, 
        observation: struct.PyTreeNode,
        model_params: struct.PyTreeNode,
        batch_stats: struct.PyTreeNode
    ) -> Tuple[MCTSState, jnp.ndarray, jnp.ndarray]:
        policy, evaluation = self.model.apply(
            {'params': model_params, 'batch_stats': batch_stats},
            observation[None, ...], 
            training=False
        )
        return state, policy.squeeze(axis=0), evaluation.squeeze(axis=0)
    
    def choose_action(self, 
        state: MCTSState,
        env: Env,
        env_state: EnvState
    ) -> Tuple[MCTSState, jnp.ndarray]:
        if self.config.temperature > 0:
            rand_key, new_key = jax.random.split(state.key)
            policy = self.get_policy(state)
            action = jax.random.choice(rand_key, len(policy), p=policy)
            return state.replace(key=new_key), action
        else:
            action = jnp.argmax(state.n_vals[1])
            return state, action
        
    def evaluate(self, 
        state: MCTSState, 
        env: Env, 
        env_state: EnvState,
        model_params: struct.PyTreeNode,
        batch_stats: struct.PyTreeNode,
    ) -> MCTSState:
        return super().evaluate(
            state, env, env_state, 
            num_iters=self.config.mcts_iters, 
            model_params=model_params,
            batch_stats=batch_stats
        )

    def get_policy(self, state: MCTSState) -> jnp.ndarray:
        action_visits = state.n_vals[1]
        action_visits = action_visits ** (1/self.config.temperature)
        return action_visits / action_visits.sum()