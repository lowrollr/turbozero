from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple
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
    terminated: jnp.ndarray
    _state: struct.PyTreeNode
    _observation: struct.PyTreeNode

class Env:
    def __init__(self, 
        env: any, 
        config: EnvConfig,
        custom_reward_fn: Optional[Callable] = None,
        *args, **kwargs
    ):
        self.config = config
        self.args = args
        self.kwargs = kwargs
        self._env = env
        self.action_space_dims = self.get_action_shape()
        self.num_actions = jnp.prod(jnp.array(self.action_space_dims)).item()
        self.custom_reward_fn = custom_reward_fn
    
    # step wrapped environment
    def _init(self, key: jax.random.PRNGKey) -> Any:
        raise NotImplementedError()

    def _step(self, _state: Any, action: Any) -> Any:
        raise NotImplementedError()

    def step(self, state: EnvState, action: jnp.ndarray,) -> Tuple[EnvState, jnp.ndarray]:
        env_state = self._step(state._state, action)
        return self._update(state, env_state)
    
    def reset(self, key: jax.random.PRNGKey) -> Tuple[EnvState, jnp.ndarray]:
        raise NotImplementedError()
    
    def get_action_shape(self) -> Tuple[int]:
        raise NotImplementedError()
    
    def get_observation_shape(self) -> Tuple[int]:
        raise NotImplementedError()
    
    @staticmethod
    def _get_terminated(_state: Any) -> jnp.ndarray:
        raise NotImplementedError()
    
    def _get_reward(self, _state: Any) -> jnp.ndarray:
        if self.custom_reward_fn:
            return self.custom_reward_fn(_state)
        else:
            return self._get_state_reward(_state)
    
    @staticmethod
    def _get_state_reward(_state: Any) -> jnp.ndarray:
        raise NotImplementedError()
    
    @staticmethod
    def _get_legal_action_mask(_state: Any) -> jnp.ndarray:
        raise NotImplementedError()
    
    @staticmethod
    def _get_cur_player_id(_state: Any) -> jnp.ndarray:
        raise NotImplementedError()
    
    @staticmethod
    def _get_observation(_state: Any) -> jnp.ndarray:
        raise NotImplementedError()
    
    def _update(self, state: EnvState, _state: Any) -> EnvState:
        return state.replace(
            legal_action_mask = self._get_legal_action_mask(_state),
            reward = self._get_reward(_state),
            cur_player_id = self._get_cur_player_id(_state),
            terminated = self._get_terminated(_state),
            _state = _state,
            _observation = self._get_observation(_state)
        )
    
    def reset(self, key: jax.random.PRNGKey) -> EnvState:
        state_key, base_key = jax.random.split(key)
        _state = self._init(base_key)
        return EnvState(
            key = state_key,
            legal_action_mask = self._get_legal_action_mask(_state),
            reward = self._get_reward(_state),
            cur_player_id = self._get_cur_player_id(_state),
            terminated = self._get_terminated(_state),
            _state = _state,
            _observation = self._get_observation(_state)
        )
    
    def num_players(self) -> int:
        raise NotImplementedError()

    def reset_if_terminated(self, state: EnvState) -> EnvState:
        return jax.lax.cond(
            state.terminated,
            lambda: self.reset(state.key),
            lambda: state
        )

    def get_random_legal_action(self, state: EnvState) -> Tuple[EnvState, jnp.ndarray]:
        rand_key, new_key = jax.random.split(state.key)
        action = jax.random.choice(
            rand_key,
            jnp.arange(state.legal_action_mask.shape[0]),
            p=state.legal_action_mask
        )

        return state.replace(key=new_key), unflatten_action(action, self.action_space_dims)