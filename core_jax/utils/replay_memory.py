from flax import struct
import jax.numpy as jnp
import jax
from functools import partial

@struct.dataclass
class EndRewardReplayBufferState:
    next_index: jnp.ndarray # next index to write experience to
    next_reward_index: jnp.ndarray # next index to write reward to
    buffer: struct.PyTreeNode # buffer of experiences
    reward_buffer: jnp.ndarray # buffer of rewards
    needs_reward: jnp.ndarray # does experience need reward
    populated: jnp.ndarray # is experience populated

class EndRewardReplayBuffer:
    def __init__(self,
        template_experience: struct.PyTreeNode,
        batch_size: int,
        max_len_per_batch: int,
        sample_batch_size: int,
    ):
        self.sample_batch_size = sample_batch_size
        self.max_len_per_batch = max_len_per_batch
        self.batch_size = batch_size

        experience = jax.tree_map(
            lambda x: jnp.broadcast_to(
                x, (batch_size, max_len_per_batch, *x.shape)
            ),
            template_experience,
        )

        self.state = EndRewardReplayBufferState(
            next_index=jnp.zeros((batch_size,), dtype=jnp.int32),
            next_reward_index=jnp.zeros((batch_size,), dtype=jnp.int32),
            reward_buffer=jnp.zeros((batch_size, max_len_per_batch, 1), dtype=jnp.float32),
            buffer=experience,
            needs_reward=jnp.zeros((batch_size, max_len_per_batch, 1), dtype=jnp.bool_),
            populated=jnp.zeros((batch_size, max_len_per_batch, 1), dtype=jnp.bool_),
        )

    def add_experience(self, experience: struct.PyTreeNode) -> None:
        self.state = add_experience(self.state, experience, self.batch_size, self.max_len_per_batch)    

    def assign_rewards(self, rewards: jnp.ndarray, select_batch: jnp.ndarray) -> None:
        self.state = assign_rewards(self.state, rewards, select_batch, self.max_len_per_batch)

    def sample(self, rng: jax.random.PRNGKey) -> struct.PyTreeNode:
        return sample(self.state, rng, self.batch_size, self.max_len_per_batch, self.sample_batch_size)




@partial(jax.jit, static_argnums=(2,3))
def add_experience(
    buffer_state: EndRewardReplayBufferState,
    experience: struct.PyTreeNode,
    batch_size: int,
    max_len_per_batch: int,
) -> EndRewardReplayBufferState:
    
    def add_item(items, new_item):
        return items.at[jnp.arange(batch_size), buffer_state.next_index].set(new_item)

    return buffer_state.replace(
        buffer = jax.tree_map(add_item, buffer_state.buffer, experience),
        next_index = (buffer_state.next_index + 1) % max_len_per_batch,
        needs_reward = buffer_state.needs_reward.at[:, buffer_state.next_index, 0].set(True),
        populated = buffer_state.populated.at[:, buffer_state.next_index, 0].set(True)
    )

@partial(jax.jit, static_argnums=(3,))
def assign_rewards(
    buffer_state: EndRewardReplayBufferState,
    rewards: jnp.ndarray,
    select_batch: jnp.ndarray,
    max_len_per_batch: int
) -> EndRewardReplayBufferState:
    rolled = jax.vmap(jnp.roll, in_axes=(0, 0))(rewards, buffer_state.next_reward_index)
    tiled = jnp.tile(rolled, (1, max_len_per_batch // rewards.shape[1]))
    
    return buffer_state.replace(
        reward_buffer = jnp.where(
            select_batch[..., None, None] & buffer_state.needs_reward,
            tiled[..., None],
            buffer_state.reward_buffer
        ),
        next_reward_index = jnp.where(
            select_batch,
            buffer_state.next_index,
            buffer_state.next_reward_index
        ),
        needs_reward = jnp.where(
            select_batch[..., None, None],
            False,
            buffer_state.needs_reward
        )
    )

@partial(jax.jit, static_argnums=(2,3,4))
def sample(
    buffer_state: EndRewardReplayBufferState,
    rng: jax.random.PRNGKey,
    batch_size: int,
    max_len_per_batch: int,
    sample_batch_size: int
) -> struct.PyTreeNode:
    probs = ((~buffer_state.needs_reward).reshape(-1) * buffer_state.populated.reshape(-1)).astype(jnp.float32)
    indices = jax.random.choice(
        rng,
        max_len_per_batch * batch_size,
        shape=(sample_batch_size,),
        replace=False,
        p = probs / probs.sum()
    )
    batch_indices = indices // max_len_per_batch
    item_indices = indices % max_len_per_batch

    return jax.tree_util.tree_map(
        lambda x: x[batch_indices, item_indices],
        buffer_state.buffer
    ), buffer_state.reward_buffer[batch_indices, item_indices]