
from functools import partial
from typing import Dict, Optional, Tuple
import jax
import chex
import jax.numpy as jnp
from core.evaluators.evaluator import Evaluator
from core.evaluators.mcts.action_selection import MCTSActionSelector
from core.evaluators.mcts.state import BackpropState, MCTSNode, MCTSTree, TraversalState, MCTSOutput
from core.trees.tree import init_tree
from core.types import EnvStepFn, EvalFn, StepMetadata

class MCTS(Evaluator):
    """Batched implementation of Monte Carlo Tree Search (MCTS).
    
    Not stateful. This class operates on 'MCTSTree' state objects.
    
    Compatible with `jax.vmap`, `jax.pmap`, `jax.jit`, etc."""
    def __init__(self,
        eval_fn: EvalFn,
        action_selector: MCTSActionSelector,
        branching_factor: int,
        max_nodes: int,
        num_iterations: int,
        discount: float = -1.0,
        temperature: float = 1.0,
        tiebreak_noise: float = 1e-8,
        persist_tree: bool = True
    ):
        """
        Args:
        - `eval_fn`: leaf node evaluation function (env_state -> (policy_logits, value))
        - `action_selector`: action selection function (eval_state -> action)
        - `branching_factor`: max number of actions (== children per node)
        - `max_nodes`: allocated size of MCTS tree, any additional nodes will not be created, 
                but values from out-of-bounds leaf nodes will still backpropagate
        - `num_iterations`: number of MCTS iterations to perform per evaluate call
        - `discount`: discount factor for MCTS (default: -1.0)
            - use a negative discount in two-player games (e.g. -1.0)
            - use a positive discount in single-player games (e.g. 1.0)
        - `temperature`: temperature for root action selection (default: 1.0)
        - `tiebreak_noise`: magnitude of noise to add to policy weights for breaking ties (default: 1e-8)
        - `persist_tree`: whether to persist search tree state between calls to `evaluate` (default: True)
        """
        super().__init__(discount=discount)
        self.eval_fn = eval_fn
        self.num_iterations = num_iterations
        self.branching_factor = branching_factor
        self.max_nodes = max_nodes
        self.action_selector = action_selector
        self.temperature = temperature
        self.tiebreak_noise = tiebreak_noise
        self.persist_tree = persist_tree


    def get_config(self) -> Dict:
        """returns a config object for checkpoints"""
        return {
            "eval_fn": self.eval_fn.__name__,
            "num_iterations": self.num_iterations,
            "branching_factor": self.branching_factor,
            "max_nodes": self.max_nodes,
            "action_selection_config": self.action_selector.get_config(),
            "discount": self.discount,
            "temperature": self.temperature,
            "tiebreak_noise": self.tiebreak_noise,
            "persist_tree": self.persist_tree
        }


    def evaluate(self, #pylint: disable=arguments-differ
        key: chex.PRNGKey,
        eval_state: MCTSTree, 
        env_state: chex.ArrayTree,
        root_metadata: StepMetadata,
        params: chex.ArrayTree,
        env_step_fn: EnvStepFn,
        **kwargs
    ) -> MCTSOutput:
        """Performs `self.num_iterations` MCTS iterations on an `MCTSTree`.
        Samples an action to take from the root node after search is completed.
        
        Args:
        - `eval_state`: `MCTSTree` to evaluate, could be empty or partially complete
        - `env_state`: current environment state
        - `root_metadata`: metadata for the root node of the tree
        - `params`: parameters to pass to the the leaf evaluation function
        - `env_step_fn`: env step fn: (env_state, action) -> (new_env_state, metadata)

        Returns:
        - (MCTSOutput): contains new tree state, selected action, root value, and policy weights
        """
        # store current state metadata in the root node
        key, root_key = jax.random.split(key)
        eval_state = self.update_root(root_key, eval_state, env_state, params, root_metadata=root_metadata)
        # perform 'num_iterations' iterations of MCTS
        iterate = partial(self.iterate, params=params, env_step_fn=env_step_fn)

        iterate_keys = jax.random.split(key, self.num_iterations)
        eval_state, _ = jax.lax.scan(lambda state, k: (iterate(k, state), None), eval_state, iterate_keys)
        # sample action based on root visit counts
        eval_state, action, policy_weights = self.sample_root_action(key, eval_state)
        return MCTSOutput(
            eval_state=eval_state,
            action=action,
            policy_weights=policy_weights
        )
    

    def get_value(self, state: MCTSTree) -> chex.Array:
        """Returns value estimate of the environment state stored in the root node of the tree.

        Args:
        - `state`: MCTSTree to evaluate

        Returns:
        - (chex.Array): value estimate of the environment state stored in the root node of the tree
        """
        return state.data_at(state.ROOT_INDEX).q
    

    def update_root(self, key: chex.PRNGKey, tree: MCTSTree, root_embedding: chex.ArrayTree, 
                    params: chex.ArrayTree, **kwargs) -> MCTSTree:
        """Populates the root node of an MCTSTree."""
        root_policy_logits, root_value = self.eval_fn(root_embedding, params, key)
        root_policy = jax.nn.softmax(root_policy_logits)
        root_node = tree.data_at(tree.ROOT_INDEX)
        root_node = self.update_root_node(root_node, root_policy, root_value, root_embedding)
        return tree.set_root(root_node)
    
    
    def iterate(self, key: chex.PRNGKey, tree: MCTSTree, params: chex.ArrayTree, env_step_fn: EnvStepFn) -> MCTSTree:
        """ Performs one iteration of MCTS.
        1. Traverse to leaf node.
        2. Evaluate Leaf Node
        3. Expand Leaf Node (add to tree)
        4. Backpropagate

        Args:
        - `tree`: MCTSTree to evaluate
        - `params`: parameters to pass to the the leaf evaluation function
        - `env_step_fn`: env step fn: (env_state, action) -> (new_env_state, metadata)

        Returns:
        - (MCTSTree): updated MCTSTree
        """
        # traverse from root -> leaf
        traversal_state = self.traverse(tree)
        parent, action = traversal_state.parent, traversal_state.action
        # get env state (embedding) for leaf node
        embedding = tree.data_at(parent).embedding
        new_embedding, metadata = env_step_fn(embedding, action)
        player_reward = metadata.rewards[metadata.cur_player_id]
        # evaluate leaf node
        policy_logits, value = self.eval_fn(new_embedding, params, key)
        policy_logits = jnp.where(metadata.action_mask, policy_logits, jnp.finfo(policy_logits).min)
        policy = jax.nn.softmax(policy_logits)
        value = jnp.where(metadata.terminated, player_reward, value)
        # add leaf node to tree
        node_exists = tree.is_edge(parent, action)
        node_idx = tree.edge_map[parent, action]

        node_data = jax.lax.cond(
            node_exists,
            lambda: self.visit_node(node=tree.data_at(node_idx), value=value, p=policy, terminated=metadata.terminated, embedding=new_embedding),
            lambda: self.new_node(policy=policy, value=value, embedding=new_embedding, terminated=metadata.terminated)
        )

        tree = jax.lax.cond(
            node_exists,
            lambda: tree.update_node(index=node_idx, data = node_data),
            lambda: tree.add_node(parent_index=parent, edge_index=action, data=node_data)
        )
        # backpropagate
        return self.backpropagate(tree, parent, value)

    
    def choose_root_action(self, tree: MCTSTree) -> int:
        """ Choose an action to take from the root node of the search tree using `self.action_selector`

        Args:
        - `tree`: MCTSTree to evaluate

        Returns:
        - (int): action to take from the root node
        """
        return self.action_selector(tree, tree.ROOT_INDEX, self.discount)

    def traverse(self, tree: MCTSTree) -> TraversalState:
        def cond_fn(state: TraversalState) -> bool:
            return jnp.logical_and(
                tree.is_edge(state.parent, state.action),
                ~(tree.data_at(tree.edge_map[state.parent, state.action]).terminated)
                # TODO: maximum depth
            )
        
        def body_fn(state: TraversalState) -> TraversalState:
            node_idx = tree.edge_map[state.parent, state.action]
            action = self.action_selector(tree, node_idx, self.discount)
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
            node = tree.data_at(node_idx)
            new_node = self.visit_node(node, value)
            tree = tree.update_node(node_idx, new_node)
            return BackpropState(node_idx=tree.parents[node_idx], value=value, tree=tree)
        
        state = jax.lax.while_loop(
            lambda s: s.node_idx != s.tree.NULL_INDEX, body_fn, 
            BackpropState(node_idx=parent, value=value, tree=tree)
        )
        return state.tree

    def sample_root_action(self, key: chex.PRNGKey, tree: MCTSTree) -> Tuple[MCTSTree, int, chex.Array]:
        action_visits = tree.get_child_data('n', tree.ROOT_INDEX)
        total_visits = action_visits.sum(axis=-1)
        policy_weights = action_visits / jnp.maximum(total_visits, 1)
        policy_weights = jnp.where(total_visits > 0, policy_weights, 1 / self.branching_factor)

        if self.temperature == 0:
            noise = jax.random.uniform(key, shape=policy_weights.shape, maxval=self.tiebreak_noise)
            noisy_policy_weights = policy_weights + noise
            return tree, jnp.argmax(noisy_policy_weights), policy_weights
        
        policy_weights_t = policy_weights ** (1/self.temperature)
        policy_weights_t /= policy_weights_t.sum()
        action = jax.random.choice(key, policy_weights_t.shape[-1], p=policy_weights_t)
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
        return state.reset()


    def step(self, state: MCTSTree, action: int) -> MCTSTree:
        if self.persist_tree:
            return state.get_subtree(action)
        return state.reset()


    def init(self, template_embedding: chex.ArrayTree, *args, **kwargs) -> MCTSTree: #pylint: disable=arguments-differ
        return init_tree(self.max_nodes, self.branching_factor, self.new_node(
            policy=jnp.zeros((self.branching_factor,)),
            value=0.0,
            embedding=template_embedding,
            terminated=False
        ))
