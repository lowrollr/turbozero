

from typing import Optional
import chex
from chex import dataclass
import jax.numpy as jnp
import graphviz
import jax
from core.evaluators.evaluator import EvalOutput

from core.trees.tree import Tree

@dataclass(frozen=True)
class MCTSNode:
    n: jnp.number
    p: chex.Array
    q: jnp.number
    terminated: jnp.number
    embedding: chex.ArrayTree

    @property
    def w(self) -> jnp.number:
        return self.q * self.n


def visit_node(
    node: MCTSNode,
    value: float,
    p: Optional[chex.Array] = None,
    terminated: Optional[bool] = None,
    embedding: Optional[chex.ArrayTree] = None
) -> MCTSNode:
    
    q_value = ((node.q * node.n) + value) / (node.n + 1)
    if p is None:
        p = node.p
    if terminated is None:
        terminated = node.terminated
    if embedding is None:
        embedding = node.embedding
    return node.replace(
        n=node.n + 1,
        q=q_value,
        p=p,
        terminated=terminated,
        embedding=embedding
    )

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

@dataclass(frozen=True)
class MCTSOutput(EvalOutput):
    eval_state: MCTSTree
    root_value: float
    policy_weights: chex.Array

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
                "q": f"{node.q.item():.2f}",
                "t": str(node.terminated.item())
            }))

            child_visits = get_child_visits_no_batch(tree, n_i)
            mapping = tree.edge_map[batch_id, n_i]
            for a_i in range(tree.edge_map.shape[2]):
                v_a = child_visits[a_i].item()
                if v_a > 0:
                    graph.edge(str(n_i), str(mapping[a_i]), f'{a_i}:{node.p[a_i]:.4f}')
        else:
            break
    
    return graph