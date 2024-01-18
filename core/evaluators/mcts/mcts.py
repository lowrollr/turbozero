
from functools import partial
from typing import Tuple
import jax
import chex
import jax.numpy as jnp
from core.evaluators.evaluator import Evaluator
from core.evaluators.mcts.action_selection import MCTSActionSelector
from core.evaluators.mcts.data import BackpropState, MCTSNode, MCTSTree, TraversalState, MCTSOutput
from core.trees.tree import _init, add_node, get_child_data, get_rng, get_subtree, init_batched_tree, reset_tree, set_root, update_node
from core.types import EnvStepFn, EvalFn, StepMetadata

class MCTS(Evaluator):
    def __init__(self,
        action_selection_fn: MCTSActionSelector,
        branching_factor: int,
        max_nodes: int,
        discount: float = -1.0,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.branching_factor = branching_factor
        self.max_nodes = max_nodes
        self.action_selection_fn = action_selection_fn
        self.discount = discount
        self.temperature = temperature

    def evaluate(self, 
        eval_state: MCTSTree, 
        env_state: chex.ArrayTree,
        root_metadata: StepMetadata,
        params: chex.ArrayTree,
        num_iterations: int,
        env_step_fn: EnvStepFn,
        eval_fn: EvalFn,
        **kwargs
    ) -> MCTSOutput:
        eval_state = self.update_root(eval_state, env_state, root_metadata, params, eval_fn)
        iterate = partial(self.iterate, 
            params=params,
            env_step_fn=env_step_fn,
            eval_fn=eval_fn
        )
        eval_state = jax.lax.fori_loop(0, num_iterations, lambda _, t: iterate(t), eval_state)
        eval_state, action, policy_weights = self.sample_root_action(eval_state)
        root_node = eval_state.at(eval_state.ROOT_INDEX)
        return MCTSOutput(
            eval_state=eval_state,
            action=action,
            root_value=root_node.w / root_node.n,
            policy_weights=policy_weights
        )


    def update_root(self, tree: MCTSTree, root_embedding: chex.ArrayTree, 
                    params: chex.ArrayTree, eval_fn: EvalFn, **kwargs) -> MCTSTree:
        root_policy_logits, root_value = eval_fn(root_embedding, params)
        root_policy = jax.nn.softmax(root_policy_logits)
        root_node = tree.at(tree.ROOT_INDEX)
        visited = root_node.n > 0
        root_node = root_node.replace(
            p=root_policy,
            w=jnp.where(visited, root_node.w, root_value),
            n=jnp.where(visited, root_node.n, 1),
            embedding=root_embedding
        )
        return set_root(tree, root_node)
    
    def iterate(self, tree: MCTSTree, params: chex.ArrayTree, env_step_fn: EnvStepFn, eval_fn: EvalFn) -> MCTSTree:
        # traverse from root -> leaf
        traversal_state = self.traverse(tree)
        parent, action = traversal_state.parent, traversal_state.action
        # evaluate and expand leaf
        embedding = tree.at(parent).embedding
        new_embedding, metadata = env_step_fn(embedding, action)
        player_reward = metadata.rewards[metadata.cur_player_id]
        policy_logits, value = eval_fn(new_embedding, params)
        policy_logits = jnp.where(metadata.action_mask, policy_logits, -jnp.inf)
        policy = jax.nn.softmax(policy_logits)
        value = jnp.where(metadata.terminated, player_reward, value)
        node_exists = tree.is_edge(parent, action)
        node_id = tree.edge_map[parent, action]
        node = tree.at(node_id)
        tree = jax.lax.cond(
            node_exists,
            lambda _: update_node(tree, node_id,
                node.replace(
                    n = node.n + 1,
                    w = node.w + value,
                    p = policy,
                    terminated = metadata.terminated,
                    embedding = new_embedding        
                )),
            lambda _: add_node(tree, parent, action, 
                MCTSNode(n=1, p=policy, w=value, terminated=metadata.terminated, embedding=new_embedding)),
            None
        )
        # backpropagate
        return self.backpropagate(tree, parent, value)
    
    def choose_root_action(self, tree: MCTSTree) -> int:
        return self.action_selection_fn(tree, tree.ROOT_INDEX, self.discount)

    def traverse(self, tree: MCTSTree) -> TraversalState:
        def cond_fn(state: TraversalState) -> bool:
            return jnp.logical_and(
                tree.is_edge(state.parent, state.action),
                ~(tree.at(tree.edge_map[state.parent, state.action]).terminated)
                # TODO: maximum depth
            )
        
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

    def sample_root_action(self, tree: MCTSTree) -> Tuple[MCTSTree, int, chex.Array]:
        action_visits = get_child_data(tree, tree.data.n, tree.ROOT_INDEX)
        policy_weights = action_visits / action_visits.sum()
        rand_key, tree = get_rng(tree)
        if self.temperature == 0:
            return tree, jnp.argmax(policy_weights), policy_weights
        
        policy_weights = policy_weights ** (1/self.temperature)
        policy_weights /= policy_weights.sum()
        action = jax.random.choice(rand_key, policy_weights.shape[-1], p=policy_weights)
        return tree, action, policy_weights

    def reset(self, state: MCTSTree) -> MCTSTree:
        return reset_tree(state)

    def step(self, state: MCTSTree, action: int) -> MCTSTree:
        return get_subtree(state, action)
    
    def init(self, key: jax.random.PRNGKey, template_embedding: chex.ArrayTree) -> MCTSTree:
        return _init(key, self.max_nodes, self.branching_factor, MCTSNode(
            n=jnp.array(0, dtype=jnp.int32),
            p=jnp.zeros(self.branching_factor, dtype=jnp.float32),
            w=jnp.array(0, dtype=jnp.float32),
            terminated=jnp.array(False, dtype=jnp.bool_),
            embedding=template_embedding
        ))
