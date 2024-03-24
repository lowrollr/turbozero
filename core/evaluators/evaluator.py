
from typing import Dict
import chex
from chex import dataclass
import jax
import jax.numpy as jnp

@dataclass(frozen=True)
class EvalOutput:
    eval_state: chex.ArrayTree
    action: int
    policy_weights: chex.Array

class Evaluator:
    def __init__(self, discount: float, *args, **kwargs):
        self.discount = discount

    def init(self, *args, **kwargs) -> chex.ArrayTree:
        raise NotImplementedError()
    
    def init_batched(self, batch_size: int, *args, **kwargs) -> chex.ArrayTree: 
        tree = self.init(*args, **kwargs) 
        return jax.tree_map(lambda x: jnp.broadcast_to(x, (batch_size,) + x.shape), tree)

    def reset(self, state: chex.ArrayTree) -> chex.ArrayTree:
        raise NotImplementedError()
    
    def evaluate(self, key: chex.PRNGKey, eval_state: chex.ArrayTree, env_state: chex.ArrayTree, **kwargs) -> EvalOutput:
        raise NotImplementedError()

    def step(self, state: chex.ArrayTree, action: chex.Array) -> chex.ArrayTree:
        return state
    
    def get_value(self, state: chex.ArrayTree) -> chex.Array:
        raise NotImplementedError()
    
    def get_config(self) -> Dict:
        return {}
