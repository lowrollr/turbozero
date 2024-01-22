
from typing import Dict
import chex
import jax 
import jax.numpy as jnp
from core.evaluators.mcts.action_selection import MCTSActionSelector
from core.evaluators.mcts.data import MCTSTree
from core.evaluators.mcts.mcts import MCTS
from core.trees.tree import set_root
from core.types import EvalFn, StepMetadata

class AlphaZero(MCTS):
    def __init__(self,
        action_selection_fn: MCTSActionSelector,
        branching_factor: int,
        max_nodes: int,
        num_iterations: int,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        discount: float = -1.0,
        temperature: float = 1.0
    ):
        super().__init__(action_selection_fn, branching_factor, max_nodes, num_iterations, discount, temperature)
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    def get_config(self) -> Dict:
        return {
            "dirichlet_alpha": self.dirichlet_alpha,
            "dirichlet_epsilon": self.dirichlet_epsilon,
            **super().get_config()
        }

    def update_root(self, tree: MCTSTree, root_embedding: chex.ArrayTree, root_metadata: StepMetadata, params: chex.ArrayTree, eval_fn: EvalFn) -> MCTSTree:
        root_policy_logits, root_value = eval_fn(root_embedding, params)
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
        
        policy = jnp.where(root_metadata.action_mask, new_logits, jnp.finfo(noisy_policy).min)
        renorm_policy = jax.nn.softmax(policy)

        root_node = tree.at(tree.ROOT_INDEX)
        visited = root_node.n > 0

        root_node = root_node.replace(
            p=renorm_policy,
            w=jnp.where(visited, root_node.w, root_value),
            n=jnp.where(visited, root_node.n, 1),
            embedding=root_embedding
        )
        return set_root(tree, root_node)
    
