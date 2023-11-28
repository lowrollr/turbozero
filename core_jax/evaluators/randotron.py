



from typing import Tuple
import jax
import jax.numpy as jnp
from flax import struct
from core_jax.envs.env import Env, EnvState
from core_jax.evaluators.evaluator import EvaluatorState
from core_jax.evaluators.mcts import MCTS, MCTSState


class Randotron(MCTS):
    def evaluate_leaf(self,
        env: Env,
        state: MCTSState,
        observation: struct.PyTreeNode,
    ) -> MCTSState:
        random_key, new_key = jax.random.split(state.key)

        return state.replace(
            key=new_key,
        ), jax.random.normal(random_key, (*env.action_space_dims,)).flatten(), jnp.zeros((1,))
        
    def evaluate(self, state: MCTSState, env: Env, env_state: EnvState) -> MCTSState:
        return super().evaluate(state, env, env_state, num_iters=100)
    
    def choose_action(self, 
        state: EvaluatorState, 
        env: Env, 
        env_state: EnvState
    ) -> Tuple[EvaluatorState, jnp.ndarray]:
        return state, jnp.argmax(jnp.where(
            env_state.legal_action_mask.reshape(-1),
            self.get_raw_policy(state),
            -jnp.inf
        ))
        