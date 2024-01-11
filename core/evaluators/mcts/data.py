

import chex
from chex import dataclass
import jax.numpy as jnp

from core.trees.tree import Tree

@dataclass(frozen=True)
class MCTSNode:
    n: jnp.number
    p: jnp.number
    w: jnp.number
    terminal: jnp.number
    embedding: chex.ArrayTree

    @property
    def q(self) -> jnp.number:
        return self.w / self.n
    
MCTSTree = Tree[MCTSNode] 

@dataclass(frozen=True)
class TraversalState:
    parent: int
    action: int

@dataclass(frozen=True)
class BackpropState:
    node_idx: int
    value: float
    tree: MCTSTree