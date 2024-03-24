
from typing import Dict
import chex
import jax 
import jax.numpy as jnp
from core.evaluators.mcts.state import MCTSTree
from core.evaluators.mcts.mcts import MCTS
from core.types import StepMetadata


class _AlphaZero:
    def __init__(self,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    def get_config(self) -> Dict:
        return {
            "dirichlet_alpha": self.dirichlet_alpha,
            "dirichlet_epsilon": self.dirichlet_epsilon,
            **super().get_config() #pylint: disable=no-member
        }

    def update_root(self, key: chex.PRNGKey, tree: MCTSTree, root_embedding: chex.ArrayTree, params: chex.ArrayTree, root_metadata: StepMetadata) -> MCTSTree:
        root_key, dir_key = jax.random.split(key, 2)
        root_policy_logits, root_value = self.eval_fn(root_embedding, params, root_key) #pylint: disable=no-member
        root_policy = jax.nn.softmax(root_policy_logits)

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

        root_node = tree.data_at(tree.ROOT_INDEX)
        root_node = self.update_root_node(root_node, renorm_policy, root_value, root_embedding) #pylint: disable=no-member
        return tree.set_root(root_node)
    

class AlphaZero(MCTS):
    # allows AlphaZero to be instantiated with a MCTS variant
    def __new__(cls, base_type: type = MCTS):
        assert issubclass(base_type, MCTS)
        cls_type = type("AlphaZero", (_AlphaZero, base_type), {})
        cls_type.__name__ = f'AlphaZero({base_type.__name__})'
        return cls_type