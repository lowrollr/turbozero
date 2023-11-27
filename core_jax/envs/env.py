from typing import Tuple
import jax
import jax.numpy as jnp
from flax import struct

from core_jax.utils.action_utils import unflatten_action

@struct.dataclass
class EnvState:
    key: jax.random.PRNGKey
    legal_action_mask: jnp.ndarray
    cur_player_id: jnp.ndarray
    reward: jnp.ndarray
    _state: struct.PyTreeNode
    _observation: struct.PyTreeNode

class Env:
    def __init__(self, 
        env: any,
        action_space_dims: Tuple[int,...],
        num_players: int, 
        *args, **kwargs
    ):
        self.args = args
        self.kwargs = kwargs
        self.action_space_dims = action_space_dims
        self.num_players = num_players
        self._env = env

    def step(self, state: EnvState, action: jnp.ndarray,) -> Tuple[EnvState, jnp.ndarray]:
        raise NotImplementedError()

    def reset(self, key: jax.random.PRNGKey) -> Tuple[EnvState, jnp.ndarray]:
        raise NotImplementedError()

    def reset_if_terminated(self, state: EnvState, terminated: jnp.ndarray) -> Tuple[EnvState, jnp.ndarray]:
        reset_key, new_key = jax.random.split(state.key)
        reset_state, terminated = jax.lax.cond(
            terminated,
            lambda: self.reset(reset_key),
            lambda: (state, terminated)
        )

        return reset_state.replace(key=new_key), terminated

    def get_random_legal_action(self, state: EnvState) -> Tuple[EnvState, jnp.ndarray]:
        rand_key, new_key = jax.random.split(state.key)
        action = jax.random.choice(
            rand_key,
            jnp.arange(state.legal_action_mask.shape[0]),
            p=state.legal_action_mask
        )

        return state.replace(key=new_key), unflatten_action(action, self.action_space_dims)