

import chex
from chex import dataclass
import jax.numpy as jnp
import graphviz
import jax

from core.trees.tree import Tree

@dataclass(frozen=True)
class MCTSNode:
    n: jnp.number
    p: jnp.number
    w: jnp.number
    terminal: jnp.number
    embedding: chex.ArrayTree
    
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

def tree_to_graph(tree, batch_id=0):
    graph = graphviz.Digraph()

    def get_child_visits_no_batch(tree, index):
        mapping = tree.edge_map[batch_id, index]
        child_data = tree.data.n[batch_id, mapping]
        return jnp.where(
            (mapping == Tree.NULL_INDEX).reshape((-1,) + (1,) * (child_data.ndim - 1)),
            0,
            child_data,
        )

    for n_i in range(tree.parents.shape[1]):
        node = jax.tree_util.tree_map(lambda x: x[batch_id, n_i], tree.data)
        if node.n.item() > 0:
            graph.node(str(n_i), str({
                "i": str(n_i),
                "n": str(node.n.item()),
                "w": f"{node.w.item():.2f}",
                "t": str(node.terminal.item())
            }))

            child_visits = get_child_visits_no_batch(tree, n_i)
            mapping = tree.edge_map[batch_id, n_i]
            for a_i in range(tree.edge_map.shape[2]):
                v_a = child_visits[a_i].item()
                if v_a > 0:
                    graph.edge(str(n_i), str(mapping[a_i]), f'{node.p[a_i]:.4f}')
        else:
            break
    
    return graph