from typing import Dict
import chex
from core.evaluators.mcts.state import MCTSTree
from core.trees.tree import get_child_data
import jax.numpy as jnp
import jax

def normalize_q_values(
    q_values: chex.Array, 
    child_n_values: chex.Array, 
    parent_q_value: float,
    epsilon: float
) -> chex.Array:
    min_value = jnp.minimum(parent_q_value, jnp.min(q_values, axis=-1))
    max_value = jnp.maximum(parent_q_value, jnp.max(q_values, axis=-1))
    completed_by_min = jnp.where(child_n_values > 0, q_values, min_value)
    normalized = (completed_by_min - min_value) / (
        jnp.maximum(max_value - min_value, epsilon))
    return normalized


class MCTSActionSelector:
    def __init__(self, epsilon: float = 1e-8): 
        self.epsilon = epsilon

    def __call__(self, tree: MCTSTree, index: int, discount: float) -> int:
        raise NotImplementedError()
    
    def get_config(self) -> Dict:
        return {
            "epsilon": self.epsilon
        }
    
class PUCTSelector(MCTSActionSelector):
    def __init__(self, 
        c: float = 1.0,
        epsilon: float = 1e-8, 
        q_transform = normalize_q_values
    ):
        super().__init__(epsilon=epsilon)
        self.c = c
        self.q_transform = q_transform

    def get_config(self) -> Dict:
        return {
            "c": self.c,
            'q_transform': self.q_transform.__name__,
            **super().get_config()
        }

    def __call__(self, tree: MCTSTree, index: int, discount: float) -> int:
        node = tree.at(index)
        q_values = get_child_data(tree, tree.data.q, index)
        discounted_q_values = q_values * discount
        n_values = get_child_data(tree, tree.data.n, index)
        q_values = self.q_transform(discounted_q_values, n_values, node.q, self.epsilon)
        u_values = self.c * node.p * jnp.sqrt(node.n) / (n_values + 1)
        puct_values = q_values + u_values
        return puct_values.argmax()
    
class MuZeroPUCTSelector(MCTSActionSelector):
    def __init__(self, 
        c1: float = 1.25, 
        c2: float = 19652, 
        epsilon: float = 1e-8,
        q_transform = normalize_q_values
    ):
        super().__init__(epsilon=epsilon)
        self.c1 = c1
        self.c2 = c2
        self.q_transform = q_transform
    
    def get_config(self) -> Dict:
        return {
            "c1": self.c1,
            "c2": self.c2,
            "q_transform": self.q_transform.__name__,
            **super().get_config()
        }

    def __call__(self, tree: MCTSTree, index: int, discount: float) -> int:
        node = tree.at(index)
        q_values = get_child_data(tree, tree.data.q, index)
        discounted_q_values = q_values * discount
        n_values = get_child_data(tree, tree.data.n, index)
        q_values = self.q_transform(discounted_q_values, q_values, n_values, node.q, self.epsilon)
        base_term = node.p * jnp.sqrt(node.n) / (n_values + 1)
        log_term = jnp.log((node.n + self.c2 + 1) / self.c2) + self.c1
        u_values = base_term * log_term
        puct_values = q_values + u_values
        return puct_values.argmax()