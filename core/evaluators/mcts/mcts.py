
from typing import Any, Callable, Optional, Tuple
import jax
import chex
from chex import dataclass
import jax.numpy as jnp
from core.trees.tree import Tree, add_node, update_node

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

class MCTS:
    def __init__(self,
        step_fn: Callable[[chex.ArrayTree, Any], Tuple[chex.ArrayTree, float, bool]],
        eval_fn: Callable[[chex.ArrayTree], Tuple[chex.ArrayTree, float]],
        action_selection_fn: Callable[[MCTSTree, int], int],
        discount: Optional[float] = -1.0
    ):
        self.step_fn = step_fn
        self.eval_fn = eval_fn
        self.action_selection_fn = action_selection_fn
        self.discount = discount

    def search(self, tree: MCTSTree, num_iterations: int) -> MCTSTree:
        return jax.lax.fori_loop(0, num_iterations, self.iterate, tree)
    
    def iterate(self, tree: MCTSTree) -> MCTSTree:
        # traverse from root -> leaf
        parent, action = self.traverse(tree)
        # evaluate and expand leaf
        new_embedding, reward, terminated = self.get_next_state(tree, parent, action)
        policy, value = self.eval_fn(new_embedding)
        value = jnp.where(terminated, reward, value)
        new_node = MCTSNode(n=1, p=policy, w=value, terminal=terminated, embedding=new_embedding)
        tree = add_node(tree, parent, action, new_node)
        # backpropagate
        return self.backpropagate(tree, parent, value)
    
    def get_next_state(self, tree: MCTSTree, parent: int, action: int) -> Tuple[chex.ArrayTree, float, bool]:
        return self.step_fn(tree.data[parent], action)
    
    def choose_root_action(self, tree: MCTSTree) -> int:
        return self.action_selection_fn(tree, tree.ROOT_INDEX)

    def traverse(self, tree: MCTSTree) -> Tuple[int, int]:
        def cond_fn(state: TraversalState) -> bool:
            return tree.edge_map[state.parent, state.action] != Tree.NULL_INDEX
        
        def body_fn(state: TraversalState) -> TraversalState:
            node_idx = tree.edge_map[state.parent, state.action]
            action = self.action_selection_fn(tree, node_idx)
            return TraversalState(parent=node_idx, action=action)
        
        root_action = self.choose_root_action(tree)
        return jax.lax.while_loop(
            cond_fn, body_fn, 
            TraversalState(parent=tree.ROOT_INDEX, action=root_action)
        )
    
    def backpropagate(self, tree: MCTSTree, parent: int, value: float) -> MCTSTree:
        def body_fn(state: BackpropState) -> Tuple[int, MCTSTree]:
            node_idx, value, tree = state
            value *= self.discount
            node = tree.data[node_idx]
            new_node = node.replace(
                n=node.n + 1,
                w=node.w + value,
            )
            tree = update_node(tree, node_idx, new_node)
            return BackpropState(node_idx=tree.parents[node_idx], value=value, tree=tree)
        
        state = jax.lax.while_loop(
            lambda s: s.node_idx != s.tree.ROOT_INDEX, body_fn, 
            BackpropState(node_idx=parent, value=value, tree=tree)
        )
        return state.tree
