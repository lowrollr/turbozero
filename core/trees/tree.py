

from typing import Tuple, TypeVar, Generic, ClassVar
from chex import dataclass
import chex
import jax
import jax.numpy as jnp

NodeType = TypeVar('NodeType')

@dataclass(frozen=True)
class Tree(Generic[NodeType]):
    key: jax.random.PRNGKey
    
    # N -> max nodes
    # F -> branching Factor
    next_free_idx: chex.Array # ()
    parents: chex.Array # (N)
    edge_map: chex.Array # (N, F)
    data: chex.ArrayTree # structured data with leaves of shape (N, ...)

    NULL_INDEX: ClassVar[int] = -1
    NULL_VALUE: ClassVar[int] = 0
    ROOT_INDEX: ClassVar[int] = 0

    @property
    def capacity(self) -> int:
        return self.parents.shape[0]
    
    def at(self, index: int) -> NodeType:
        return jax.tree_util.tree_map(
            lambda x: x[index],
            self.data
        )

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
    tree: Tree[NodeType], 
    parent_index: int, 
    edge_index: int, 
    data: NodeType
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

def set_root(
    tree: Tree, 
    data: NodeType
) -> Tree:
    return tree.replace(
        next_free_idx=jnp.maximum(tree.next_free_idx, 1),
        data=jax.tree_util.tree_map(
            lambda x, y: x.at[tree.ROOT_INDEX].set(y),
            tree.data,
            data
        )
    )

def update_node(
    tree: Tree[NodeType],
    index: int,
    data: NodeType
) -> Tree:
    return tree.replace(
        data=jax.tree_util.tree_map(
            lambda x, y: x.at[index].set(y),
            tree.data,
            data
        )
    )


def get_rng(tree: Tree) -> Tuple[jax.random.PRNGKey, Tree]:
    rng, new_rng = jax.random.split(tree.key, 2)
    return rng, tree.replace(key=new_rng)

def _get_translation(
    tree: Tree,
    child_index: chex.Array
) -> Tuple[chex.Array, chex.Array, chex.Array]:
  subtrees = jnp.arange(tree.capacity)

  def propagate(_, subtrees):
    parents_subtrees = jnp.where(
        tree.parents != tree.NULL_INDEX,
        subtrees[tree.parents],
        0
    )
    return jnp.where(
        jnp.greater(parents_subtrees, 0),
        parents_subtrees,
        subtrees
    )

  subtrees = jax.lax.fori_loop(0, tree.capacity-1, propagate, subtrees)
  slots_aranged = jnp.arange(tree.capacity)
  subtree_idx = tree.edge_map[tree.ROOT_INDEX, child_index]
  nodes_to_retain = subtrees == subtree_idx
  old_subtree_idxs = nodes_to_retain * slots_aranged
  cumsum = jnp.cumsum(nodes_to_retain)
  new_next_node_index = cumsum[-1]

  translation = jnp.where(
      nodes_to_retain,
      nodes_to_retain * (cumsum-1),
      tree.NULL_INDEX
  )
  erase_idxs = slots_aranged >= new_next_node_index

  return old_subtree_idxs, translation, erase_idxs

def get_subtree(
    tree: Tree, 
    subtree_index: int
) -> Tree:
    old_subtree_idxs, translation, erase_idxs = _get_translation(tree, subtree_index)

    new_next_node_index = translation.max(axis=-1) + 1

    def translate(x, null_value=tree.NULL_VALUE):
        return jnp.where(
            erase_idxs.reshape((-1,) + (1,) * (x.ndim - 1)),
            null_value,
            # cases where translation == -1 will set last index
            # but since we are at least removing the root node
            # (and making one of its children the new root)
            # the last index will always be freed
            # and overwritten with zeros
            x.at[translation].set(x[old_subtree_idxs]),
        )

    def translate_idx(x, null_value=tree.NULL_INDEX):
        return jnp.where(
            erase_idxs.reshape((-1,) + (1,) * (x.ndim - 1)),
            null_value,
            # in this case we need to explicitly check for index
            # mappings to UNVISITED, since otherwise thsese will
            # map to the value of the last index of the translation
            x.at[translation].set(jnp.where(
                x == null_value,
                null_value,
                translation[x])))

    def translate_pytree(x, null_value=tree.NULL_VALUE):
        return jax.tree_map(
            lambda t: translate(t, null_value=null_value), x)
    
    return tree.replace(
        next_free_idx=new_next_node_index,
        parents=translate_idx(tree.parents),
        edge_map=translate_idx(tree.edge_map),
        data=translate_pytree(tree.data)
    )