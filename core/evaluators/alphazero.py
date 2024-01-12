

from typing import Any, Callable, Optional, Tuple

import chex
import jax 
import jax.numpy as jnp
from core.evaluators.mcts.action_selection import MCTSActionSelector
from core.evaluators.mcts.data import MCTSNode, MCTSTree
from core.evaluators.mcts.mcts import MCTS
from core.trees.tree import get_child_data, get_rng, set_root

class AlphaZero(MCTS):
    def __init__(self,
        step_fn: Callable[[chex.ArrayTree, Any], Tuple[chex.ArrayTree, float, bool]],
        eval_fn: Callable[[chex.ArrayTree], Tuple[chex.ArrayTree, float]],
        action_selection_fn: MCTSActionSelector,
        action_mask_fn: Callable[[chex.ArrayTree], chex.Array] = lambda _: jnp.array([True]),
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        discount: float = -1.0,
        temperature: float = 1.0
    ):
        super().__init__(step_fn, eval_fn, action_selection_fn, action_mask_fn, discount, temperature)
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    def update_root(self, tree: MCTSTree, root_embedding: chex.ArrayTree) -> MCTSTree:
        root_policy_logits, root_value = self.evaluate_root(root_embedding)
        root_policy = jax.nn.softmax(root_policy_logits)

        dir_key, new_key = jax.random.split(tree.key)
        tree = tree.replace(key=new_key)

        dirichlet_noise = jax.random.dirichlet(
            dir_key,
            alpha=jnp.full(
                [tree.branching_factor], 
                fill_value=self.dirichlet_alpha
            )
        )
        noisy_policy = (
            ((1-self.dirichlet_epsilon) * root_policy) +
            (self.dirichlet_epsilon * dirichlet_noise)
        )
        new_logits = jnp.log(jnp.maximum(noisy_policy, jnp.finfo(noisy_policy).tiny))
        renorm_policy = jax.nn.softmax(new_logits)
        policy_mask = self.action_mask_fn(root_embedding)
        policy = jnp.where(policy_mask, renorm_policy, -jnp.inf)

        root_node = tree.at(tree.ROOT_INDEX)
        visited = root_node.n > 0

        root_node = root_node.replace(
            p=policy,
            w=jnp.where(visited, root_node.w, root_value),
            n=jnp.where(visited, root_node.n, 1),
            embedding=root_embedding
        )
        return set_root(tree, root_node)
    
