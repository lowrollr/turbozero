



from typing import Tuple
import jax
import jax.numpy as jnp
from flax import struct
from core.envs.env import Env, EnvState
from core.evaluators.evaluator import EvaluatorState
from core.evaluators.mcts import MCTS, MCTSState


class Randotron(MCTS):
    def evaluate_leaf(self,
        state: MCTSState,
        observation: struct.PyTreeNode,
    ) -> MCTSState:
        random_key, new_key = jax.random.split(state.key)

        return state.replace(
            key=new_key,
        ), jax.random.normal(random_key, (*self.env.get_action_shape(),)).flatten(), jnp.zeros((1,))
        
    def evaluate(self, state: MCTSState, env_state: EnvState) -> MCTSState:
        return super().evaluate(state, env_state, num_iters=100)
    
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
        