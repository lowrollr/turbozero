
from functools import partial
from typing import Tuple
from chex import dataclass
import chex
import jax
import jax.numpy as jnp

@dataclass(frozen=True)
class BaseExperience:
    reward: chex.Array
    policy_weights: chex.Array
    policy_mask: chex.Array
    observation_nn: chex.Array
    cur_player_id: chex.Array

@dataclass(frozen=True)
class ReplayBufferState:
    key: jax.random.PRNGKey
    next_idx: int
    episode_start_idx: int
    buffer: BaseExperience
    populated: chex.Array
    has_reward: chex.Array


class EpisodeReplayBuffer:
    def __init__(self,
        capacity: int,
    ):
        self.capacity = capacity

    def get_config(self):
        return {
            'capacity': self.capacity,
        }

    def add_experience(self,
        state: ReplayBufferState,
        experience: BaseExperience
    ) -> ReplayBufferState:
        return state.replace(
            buffer = jax.tree_util.tree_map(
                lambda x, y: x.at[state.next_idx].set(y),
                state.buffer,
                experience
            ),
            next_idx = (state.next_idx + 1) % self.capacity,
            populated = state.populated.at[state.next_idx].set(True),
            has_reward = state.has_reward.at[state.next_idx].set(False)
        )
    
    def assign_rewards(self,
        state: ReplayBufferState,
        reward: chex.Array
    ) -> ReplayBufferState:
        return state.replace(
            episode_start_idx = state.next_idx,
            has_reward = jnp.full_like(state.has_reward, True),
            buffer = state.buffer.replace(
                reward = jnp.where(
                    ~state.has_reward[..., None],
                    reward[None, ...],
                    state.buffer.reward
                )
            )
        )
    
    def truncate(self,
        state: ReplayBufferState,
    ) -> ReplayBufferState:
        # un-assigned trajectory indices have populated set to False
        # so their buffer contents will be overwritten (eventually)
        # and cannot be sampled
        # so there's no need to overwrite them with zeros here
        return state.replace(
            next_idx = state.episode_start_idx,
            has_reward = jnp.full_like(state.has_reward, True),
            populated = jnp.where(
                ~state.has_reward,
                False,
                state.populated 
            )
        )
    
    def init_batched_buffer(self,
        key: jax.random.PRNGKey,
        batch_size: int,
        template_experience: chex.ArrayTree,
    ) -> ReplayBufferState:
        keys = jax.random.split(key, batch_size)
        return jax.vmap(
            _init, 
            in_axes=(0, None, jax.tree_util.tree_map(
                lambda _: None, template_experience))
        )(keys, self.capacity, template_experience)
    
    # assumes input is batched!! (dont vmap)
    def sample(self,
        state: ReplayBufferState,
        key: jax.random.PRNGKey,
        sample_size: int
    ) -> chex.ArrayTree:

        masked_weights = jnp.logical_and(
            state.populated, 
            state.has_reward
        ).reshape(-1)

        num_partitions = state.populated.shape[0]
        num_batches = state.populated.shape[1]

        indices = jax.random.choice(
            key,
            self.capacity * num_partitions * num_batches,
            shape=(sample_size,),
            replace=False,
            p = masked_weights / masked_weights.sum()
        )

        partition_indices = indices // (self.capacity * num_batches)
        batch_indices = (indices // self.capacity) % num_partitions
        item_indices = indices % self.capacity

        sampled_buffer_items = jax.tree_util.tree_map(
            lambda x: x[partition_indices, batch_indices, item_indices],
            state.buffer
        )

        return sampled_buffer_items

def _init(key: jax.random.PRNGKey, capacity: int, template_experience: BaseExperience) -> ReplayBufferState:
    return ReplayBufferState(
        key = key,
        next_idx = 0,
        episode_start_idx = 0,
        buffer = jax.tree_util.tree_map(
            lambda x: jnp.zeros((capacity, *x.shape), dtype=x.dtype),
            template_experience
        ),
        populated = jnp.full((capacity,), fill_value=False, dtype=jnp.bool_),
        has_reward = jnp.full((capacity,), fill_value=True, dtype=jnp.bool_),
    )



