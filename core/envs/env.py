from dataclasses import dataclass
from typing import Callable, Tuple
import jax
import jax.numpy as jnp
from flax import struct

from core.utils.action_utils import unflatten_action


@struct.dataclass
class EnvConfig:
    env_pkg: str
    env_name: str
    base_config: dict

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
        config: EnvConfig,
        *args, **kwargs
    ):
        self.config = config
        self.args = args
        self.kwargs = kwargs
        self._env = env
        self.action_space_dims = self.get_action_shape()
        self.num_actions = jnp.prod(jnp.array(self.action_space_dims)).item()

    def step(self, state: EnvState, action: jnp.ndarray,) -> Tuple[EnvState, jnp.ndarray]:
        raise NotImplementedError()

    def reset(self, key: jax.random.PRNGKey) -> Tuple[EnvState, jnp.ndarray]:
        raise NotImplementedError()
    
    def get_action_shape(self) -> Tuple[int]:
        raise NotImplementedError()
    
    def get_observation_shape(self) -> Tuple[int]:
        raise NotImplementedError()
    
    def num_players(self) -> int:
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