
from typing import Any, Callable, Optional, Tuple
import jax
import chex
from chex import dataclass
import jax.numpy as jnp
from core.evaluators.mcts.action_selection import MCTSActionSelector
from core.evaluators.mcts.data import BackpropState, MCTSNode, MCTSTree, TraversalState
from core.trees.tree import Tree, add_node, set_root, update_node

class MCTS:
    def __init__(self,
        step_fn: Callable[[chex.ArrayTree, Any], Tuple[chex.ArrayTree, float, bool]],
        eval_fn: Callable[[chex.ArrayTree], Tuple[chex.ArrayTree, float]],
        action_selection_fn: MCTSActionSelector,
        discount: Optional[float] = -1.0
    ):
        self.step_fn = step_fn
        self.eval_fn = eval_fn
        self.action_selection_fn = action_selection_fn
        self.discount = discount

    def search(self, tree: MCTSTree, root_embedding: chex.ArrayTree, num_iterations: int) -> MCTSTree:   
        tree = self.update_root(tree, root_embedding)
        return jax.lax.fori_loop(0, num_iterations, lambda _, t: self.iterate(t), tree)
    
    def update_root(self, tree: MCTSTree, root_embedding: chex.ArrayTree) -> MCTSTree:
        root_node = tree.at(tree.ROOT_INDEX)
        root_policy, root_value = self.evaluate_root(root_embedding)
        visited = root_node.n > 0
        root_node = root_node.replace(
            p=root_policy,
            w=jnp.where(visited, root_node.w, root_value),
            n=jnp.where(visited, root_node.n, 1),
            embedding=root_embedding
        )
        return set_root(tree, root_node)
    
    def evaluate_root(self, root_embedding: chex.ArrayTree) -> Tuple[chex.ArrayTree, float]:
        return self.eval_fn(root_embedding)
    
    def iterate(self, tree: MCTSTree) -> MCTSTree:
        # traverse from root -> leaf
        traversal_state = self.traverse(tree)
        parent, action = traversal_state.parent, traversal_state.action
        # evaluate and expand leaf
        new_embedding, reward, terminated = self.get_next_state(tree, parent, action)
        policy, value = self.eval_fn(new_embedding)
        value = jnp.where(terminated, reward, value)
        new_node = MCTSNode(n=1, p=policy, w=value, terminal=terminated, embedding=new_embedding)
        tree = add_node(tree, parent, action, new_node)
        # backpropagate
        return self.backpropagate(tree, parent, value)
    
    def get_next_state(self, tree: MCTSTree, parent: int, action: int) -> Tuple[chex.ArrayTree, float, bool]:
        return self.step_fn(tree.at(parent).embedding, action)
    
    def choose_root_action(self, tree: MCTSTree) -> int:
        return self.action_selection_fn(tree, tree.ROOT_INDEX, self.discount)

    def traverse(self, tree: MCTSTree) -> TraversalState:
        def cond_fn(state: TraversalState) -> bool:
            return tree.edge_map[state.parent, state.action] != Tree.NULL_INDEX
        
        def body_fn(state: TraversalState) -> TraversalState:
            node_idx = tree.edge_map[state.parent, state.action]
            action = self.action_selection_fn(tree, node_idx, self.discount)
            return TraversalState(parent=node_idx, action=action)
        
        root_action = self.choose_root_action(tree)
        return jax.lax.while_loop(
            cond_fn, body_fn, 
            TraversalState(parent=tree.ROOT_INDEX, action=root_action)
        )
    
    def backpropagate(self, tree: MCTSTree, parent: int, value: float) -> MCTSTree:
        def body_fn(state: BackpropState) -> Tuple[int, MCTSTree]:
            node_idx, value, tree = state.node_idx, state.value, state.tree
            value *= self.discount
            node = tree.at(node_idx)
            new_node = node.replace(
                n=node.n + 1,
                w=node.w + value,
            )
            tree = update_node(tree, node_idx, new_node)
            return BackpropState(node_idx=tree.parents[node_idx], value=value, tree=tree)
        
        state = jax.lax.while_loop(
            lambda s: s.node_idx != s.tree.NULL_INDEX, body_fn, 
            BackpropState(node_idx=parent, value=value, tree=tree)
        )
        return state.tree
