


from typing import Dict, Tuple
from chex import dataclass
import chex
from core.evaluators.mcts.mcts import MCTS
from core.evaluators.mcts.state import BackpropState, MCTSNode, MCTSTree
from core.trees.tree import get_rng, update_node
from core.evaluators.mcts.action_selection import normalize_q_values
import jax.numpy as jnp
import jax

@dataclass(frozen=True)
class WeightedMCTSNode(MCTSNode):
    r: float # raw value returned at leaf evaluation

class WeightedMCTS(MCTS):
    def __init__(self,
        q_temperature: float = 1.0,
        epsilon: float = 1e-8,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.q_temperature = q_temperature
        self.epsilon = epsilon
    
    def get_config(self) -> Dict:
        return {
            "q_temperature": self.q_temperature,
            "epsilon": self.epsilon,
            **super().get_config()
        }

    @staticmethod
    def new_node(policy: chex.Array, value: float, embedding: chex.ArrayTree, terminated: bool) -> WeightedMCTSNode:
        return WeightedMCTSNode(
            n=jnp.array(1, dtype=jnp.int32),
            p=policy,
            q=jnp.array(value, dtype=jnp.float32),
            r=jnp.array(value, dtype=jnp.float32),
            terminated=jnp.array(terminated, dtype=jnp.bool_),
            embedding=embedding
        )
    
    @staticmethod
    def update_root_node(root_node: MCTSNode, root_policy: chex.Array, root_value: float, root_embedding: chex.ArrayTree) -> WeightedMCTSNode:
        visited = root_node.n > 0
        return root_node.replace(
            p=root_policy,
            q=jnp.where(visited, root_node.q, root_value),
            r=jnp.where(visited, root_node.r, root_value),
            n=jnp.where(visited, root_node.n, 1),
            embedding=root_embedding
        )

    def backpropagate(self, tree: MCTSTree, parent: int, value: float) -> MCTSTree:
        def body_fn(state: BackpropState) -> Tuple[int, MCTSTree]:
            node_idx, tree = state.node_idx, state.tree
            # get node data
            node = tree.data_at(node_idx)
            # get q values, visit counts of children 
            child_q_values = tree.get_child_data('q', node_idx) * self.discount
            child_n_values = tree.get_child_data('n', node_idx)

            # normalize q-values to [0, 1]
            normalized_q_values = normalize_q_values(child_q_values, child_n_values, node.q, self.epsilon)
            
            if self.q_temperature > 0:
                # if temperature > 0, apply temperature to q-values
                q_values = normalized_q_values ** (1/self.q_temperature)
                # mask out unvisited action a[i] so softmax(a)[i] = 0.0
                q_values_masked = jnp.where(
                    child_n_values > 0, normalized_q_values, jnp.finfo(normalized_q_values).min
                )
            else:
                # if temperature == 0, select max q-value
                rand_key, tree = get_rng(tree)
                # apply random noise to break ties amongst nodes w/ same number of visits
                noise = jax.random.uniform(rand_key, shape=normalized_q_values.shape, maxval=self.tiebreak_noise)
                noisy_q_values = normalized_q_values + noise

                # mask out all values except for max value index
                # so softmax output at a[max_index] = 1 and 0 everywhere else
                max_vector = jnp.full_like(noisy_q_values, jnp.finfo(noisy_q_values).min)
                index_of_max = jnp.argmax(noisy_q_values)
                max_vector = max_vector.at[index_of_max].set(1)
                q_values = normalized_q_values
                q_values_masked = max_vector

            # compute weights
            child_weights = jax.nn.softmax(q_values_masked, axis=-1)
            # computer weighted sum of q-values
            weighted_value = jnp.sum(child_weights * q_values)
            # update node with weighted value
            node = node.replace(q=weighted_value)
            # adjust node value to ((weighted_value * node_visits) + raw_value) / (node_visits + 1)
            # and increment visit count
            node = self.visit_node(node, node.r)
            # update search tree
            tree = update_node(tree, node_idx, node)
            # backprop to parent node
            return BackpropState(node_idx=tree.parents[node_idx], value=value, tree=tree)
        
        state = jax.lax.while_loop(
            lambda s: s.node_idx != s.tree.NULL_INDEX, body_fn, 
            BackpropState(node_idx=parent, value=value, tree=tree)
        )
        return state.tree
