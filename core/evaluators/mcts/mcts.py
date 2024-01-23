
from functools import partial
from typing import Dict, Optional, Tuple
import jax
import chex
import jax.numpy as jnp
from core.evaluators.evaluator import Evaluator
from core.evaluators.mcts.action_selection import MCTSActionSelector
from core.evaluators.mcts.state import BackpropState, MCTSNode, MCTSTree, TraversalState, MCTSOutput
from core.trees.tree import _init, add_node, get_child_data, get_rng, get_subtree, reset_tree, set_root, update_node
from core.types import EnvStepFn, EvalFn, StepMetadata

class MCTS(Evaluator):
    def __init__(self,
        action_selection_fn: MCTSActionSelector,
        branching_factor: int,
        max_nodes: int,
        num_iterations: int,
        discount: float = -1.0,
        temperature: float = 1.0,
        tiebreak_noise: float = 1e-8
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.branching_factor = branching_factor
        self.max_nodes = max_nodes
        self.action_selection_fn = action_selection_fn
        self.discount = discount
        self.temperature = temperature
        self.tiebreak_noise = tiebreak_noise
    
    def get_config(self) -> Dict:
        return {
            "num_iterations": self.num_iterations,
            "branching_factor": self.branching_factor,
            "max_nodes": self.max_nodes,
            "discount": self.discount,
            "temperature": self.temperature,
            "action_selection_config": self.action_selection_fn.get_config()
        }

    def evaluate(self, 
        eval_state: MCTSTree, 
        env_state: chex.ArrayTree,
        root_metadata: StepMetadata,
        params: chex.ArrayTree,
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
        eval_state = jax.lax.fori_loop(0, self.num_iterations, lambda _, t: iterate(t), eval_state)
        eval_state, action, policy_weights = self.sample_root_action(eval_state)
        root_node = eval_state.at(eval_state.ROOT_INDEX)
        return MCTSOutput(
            eval_state=eval_state,
            action=action,
            root_value=root_node.q,
            policy_weights=policy_weights
        )


    def update_root(self, tree: MCTSTree, root_embedding: chex.ArrayTree, 
                    params: chex.ArrayTree, eval_fn: EvalFn, **kwargs) -> MCTSTree:
        key, tree = get_rng(tree)
        root_policy_logits, root_value = eval_fn(root_embedding, params, key)
        root_policy = jax.nn.softmax(root_policy_logits)
        root_node = tree.at(tree.ROOT_INDEX)
        root_node = self.update_root_node(root_node, root_policy, root_value, root_embedding)
        return set_root(tree, root_node)
    
    def iterate(self, tree: MCTSTree, params: chex.ArrayTree, env_step_fn: EnvStepFn, eval_fn: EvalFn) -> MCTSTree:
        # traverse from root -> leaf
        traversal_state = self.traverse(tree)
        parent, action = traversal_state.parent, traversal_state.action
        # evaluate and expand leaf
        embedding = tree.at(parent).embedding
        new_embedding, metadata = env_step_fn(embedding, action)
        player_reward = metadata.rewards[metadata.cur_player_id]
        key, tree = get_rng(tree)
        policy_logits, value = eval_fn(new_embedding, params, key)
        policy_logits = jnp.where(metadata.action_mask, policy_logits, jnp.finfo(policy_logits).min)
        policy = jax.nn.softmax(policy_logits)
        value = jnp.where(metadata.terminated, player_reward, value)
        node_exists = tree.is_edge(parent, action)
        node_idx = tree.edge_map[parent, action]

        tree = jax.lax.cond(
            node_exists,
            lambda _: update_node(tree, node_idx, self.visit_node(node=tree.at(node_idx), value=value, p=policy, 
                                 terminated=metadata.terminated, embedding=new_embedding)), 
            lambda _: add_node(tree, parent, action, 
                self.new_node(policy=policy, value=value, terminated=metadata.terminated, embedding=new_embedding)),
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
            new_node = self.visit_node(node, value)
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
        
        if self.temperature == 0:
            rand_key, tree = get_rng(tree)
            noise = jax.random.uniform(rand_key, shape=policy_weights.shape, maxval=self.tiebreak_noise)
            policy_weights += noise
            return tree, jnp.argmax(policy_weights), policy_weights
        
        policy_weights = policy_weights ** (1/self.temperature)
        policy_weights /= policy_weights.sum()
        rand_key, tree = get_rng(tree)
        action = jax.random.choice(rand_key, policy_weights.shape[-1], p=policy_weights)
        return tree, action, policy_weights
    
    @staticmethod
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
    
    @staticmethod
    def new_node(policy: chex.Array, value: float, embedding: chex.ArrayTree, terminated: bool) -> MCTSNode:
        return MCTSNode(
            n=jnp.array(1, dtype=jnp.int32),
            p=policy,
            q=jnp.array(value, dtype=jnp.float32),
            terminated=jnp.array(terminated, dtype=jnp.bool_),
            embedding=embedding
        )
    
    @staticmethod
    def update_root_node(root_node: MCTSNode, root_policy: chex.Array, root_value: float, root_embedding: chex.ArrayTree) -> MCTSNode:
        visited = root_node.n > 0
        return root_node.replace(
            p=root_policy,
            q=jnp.where(visited, root_node.q, root_value),
            n=jnp.where(visited, root_node.n, 1),
            embedding=root_embedding
        )

    def reset(self, state: MCTSTree) -> MCTSTree:
        return reset_tree(state)

    def step(self, state: MCTSTree, action: int) -> MCTSTree:
        return get_subtree(state, action)
    
    def init(self, key: jax.random.PRNGKey, template_embedding: chex.ArrayTree) -> MCTSTree:
        return _init(key, self.max_nodes, self.branching_factor, self.new_node(
            policy=jnp.zeros((self.branching_factor,)),
            value=0.0,
            embedding=template_embedding,
            terminated=False
        ))
