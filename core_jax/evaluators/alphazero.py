





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

    def init_params(self, key: jax.random.PRNGKey) -> struct.PyTreeNode:
        return self.model.init(key, jnp.ones((1, *self.observation_shape), dtype=jnp.float32))
    
    def update_params(self, state: MCTSState, model_params: struct.PyTreeNode) -> MCTSState:
        return state.replace(model_params=model_params)

    def evaluate_leaf(self, 
        env: Env, 
        state: MCTSState, 
        observation: struct.PyTreeNode,
        model_params: struct.PyTreeNode
    ) -> Tuple[MCTSState, jnp.ndarray, jnp.ndarray]:
        policy, evaluation = self.model.apply(model_params, observation[None, ...], training=False)
        return state, policy.squeeze(axis=0), evaluation.squeeze(axis=0)
    
    def choose_action(self, 
        state: MCTSState,
        env: Env,
        env_state: EnvState
    ) -> Tuple[MCTSState, jnp.ndarray]:
        if self.config.temperature > 0:
            rand_key, new_key = jax.random.split(state.key)
            action = jax.random.categorical(rand_key, jnp.power(state.p_vals[1], 1/self.config.temperature), shape=())
            return state.replace(key=new_key), action
        else:
            action = jnp.argmax(state.p_vals[1])
            return state, action
        
    def evaluate(self, 
        state: MCTSState, 
        env: Env, 
        env_state: EnvState,
        model_params: struct.PyTreeNode
    ) -> MCTSState:
        return super().evaluate(state, env, env_state, num_iters=self.config.mcts_iters, model_params=model_params)

        