
from functools import partial
import jax
import jax.numpy as jnp
from flax import struct
from core_jax.utils.replay_memory import EndRewardReplayBuffer, EndRewardReplayBufferState, init as super_init


@struct.dataclass
class RankedRewardReplayBufferState(EndRewardReplayBufferState):
    next_index: jnp.ndarray # next index to write experience to
    next_reward_index: jnp.ndarray # next index to write reward to
    buffer: struct.PyTreeNode # buffer of experiences
    reward_buffer: jnp.ndarray # buffer of ranked rewards
    raw_reward_buffer: jnp.ndarray # buffer of raw rewards
    needs_reward: jnp.ndarray # does experience need reward
    populated: jnp.ndarray # is experience populated
    key: jax.random.PRNGKey

class RankedRewardReplayBuffer(EndRewardReplayBuffer):
    def __init__(self,
        batch_size: int,
        max_len_per_batch: int,
        sample_batch_size: int,
        percentile: int = 75
    ):
        super().__init__(batch_size, max_len_per_batch, sample_batch_size)
        self.percentile = percentile

    def init(self, template_experience: struct.PyTreeNode) -> EndRewardReplayBufferState:
        return init(template_experience, self.batch_size, self.max_len_per_batch)

    def assign_rewards(self, state: EndRewardReplayBufferState, rewards: jnp.ndarray, select_batch: jnp.ndarray) -> EndRewardReplayBufferState:
        return assign_rewards(state, rewards, select_batch.astype(jnp.bool_), self.max_len_per_batch, self.percentile)



@partial(jax.jit, static_argnums=(1,2))
def init(
    template_experience: struct.PyTreeNode,
    batch_size: int,
    max_len_per_batch: int,
) -> RankedRewardReplayBufferState:
    buffer_state = super_init(template_experience, batch_size, max_len_per_batch)
    return RankedRewardReplayBufferState(
        **buffer_state.__dict__, 
        raw_reward_buffer=jnp.zeros((batch_size, max_len_per_batch, 1))
    )
    


@partial(jax.jit, static_argnums=(3,4))
def assign_rewards(
    buffer_state: RankedRewardReplayBufferState,
    rewards: jnp.ndarray,
    select_batch: jnp.ndarray,
    max_len_per_batch: int,
    percentile: int
) -> RankedRewardReplayBufferState:
    rand_key, new_key = jax.random.split(buffer_state.key)
    rand_bools = jax.random.uniform(rand_key, rewards.shape) < 0.5
    percentile_value = jax.vmap(jnp.percentile, in_axes=(0,None,None))(
        buffer_state.raw_reward_buffer, 
        percentile, 
        1
    ).mean()
    
    def rank_rewards(reward, boolean):
        return jnp.where(
            reward < percentile_value, 
            -1,
            jnp.where(
                reward > percentile_value,
                1,
                jnp.where(
                    boolean,
                    1,
                    -1
                )
            )
        )
    
    
    ranked_rewards = jax.vmap(rank_rewards)(rewards, rand_bools)
    
    rewards = jax.vmap(jnp.roll, in_axes=(0, 0))(rewards, buffer_state.next_reward_index)
    rewards = jnp.tile(rewards, (1, max_len_per_batch // rewards.shape[-1]))

    ranked_rewards = jax.vmap(jnp.roll, in_axes=(0, 0))(ranked_rewards, buffer_state.next_reward_index)
    ranked_rewards = jnp.tile(ranked_rewards, (1, max_len_per_batch // ranked_rewards.shape[-1]))
    
    
    return buffer_state.replace(
        key=new_key,
        reward_buffer = jnp.where(
            select_batch[..., None, None] & buffer_state.needs_reward,
            ranked_rewards[..., None],
            buffer_state.reward_buffer
        ),
        raw_reward_buffer = jnp.where(
            select_batch[..., None, None] & buffer_state.needs_reward,
            rewards[..., None],
            buffer_state.raw_reward_buffer
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