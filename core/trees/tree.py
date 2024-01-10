

from typing import Tuple, TypeVar, Generic, ClassVar
from chex import dataclass
import chex
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class Tree:
    key: jax.random.PRNGKey
    
    # N -> max nodes
    # F -> branching Factor
    next_free_idx: chex.Array # ()
    parents: chex.Array # (N)
    edge_map: chex.Array # (N, F)
    data: chex.ArrayTree # structured data with leaves of shape (N, ...)

    NULL_INDEX: ClassVar[int] = -1
    NULL_VALUE: ClassVar[int] = 0


def _init(key: jax.random.PRNGKey, max_nodes: int, branching_factor: int, dummy_data: chex.ArrayTree) -> Tree:
    return Tree(
        key=key,
        next_free_idx=0,
        parents=jnp.full((max_nodes,), fill_value=Tree.NULL_INDEX, dtype=jnp.int32),
        edge_map=jnp.full((max_nodes, branching_factor), fill_value=Tree.NULL_INDEX, dtype=jnp.int32),
        data=jax.tree_util.tree_map(
            lambda x: jnp.zeros((max_nodes, *x.shape), dtype=x.dtype),
            dummy_data
        )
    )


def init_batched_tree(
    key: jax.random.PRNGKey,
    batch_size: int, 
    max_nodes: int, 
    branching_factor: int, 
    dummy_data: chex.ArrayTree
) -> Tree:
    keys = jax.random.split(key, batch_size)
    return jax.vmap(
        _init, 
        in_axes=(0, None, None, jax.tree_util.tree_map(
            lambda _: None, dummy_data))
    )(keys, max_nodes, branching_factor, dummy_data)


def get_child_data(
    tree: Tree, 
    x: chex.Array, 
    index: int, 
    null_value=Tree.NULL_VALUE
) -> chex.Array:
    mapping = tree.edge_map[index]
    child_data = x[mapping]
    return jnp.where(
        (mapping == Tree.NULL_INDEX).reshape((-1,) + (1,) * (child_data.ndim - 1)),
        null_value,
        child_data,
    )


def add_node(
    tree: Tree, 
    parent_index: int, 
    edge_index: int, 
    data: chex.ArrayTree
) -> Tree:
    return tree.replace(
        next_free_idx=tree.next_free_idx + 1,
        parents=tree.parents.at[tree.next_free_idx].set(parent_index),
        edge_map=tree.edge_map.at[parent_index, edge_index].set(tree.next_free_idx),
        data=jax.tree_util.tree_map(
            lambda x, y: x.at[tree.next_free_idx].set(y),
            tree.data,
            data
        )
    )


def get_rng(tree: Tree) -> Tuple[jax.random.PRNGKey, Tree]:
    rng, new_rng = jax.random.split(tree.key, 2)
    return rng, tree.replace(key=new_rng)
