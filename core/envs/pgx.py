

from typing import Any, Tuple
import pgx
import jax.numpy as jnp
import jax
from core.envs.env import Env, EnvConfig, EnvState


class PgxEnv(Env):
    def __init__(self, env: pgx.core.Env, config: EnvConfig, **kwargs):
        super().__init__(
            env = env, 
            config = config,
            **kwargs
        )
        self._env: pgx.core.Env

    def get_action_shape(self) -> Tuple[int]:
        return (self._env.num_actions,)
    
    def get_observation_shape(self) -> Tuple[int]:
        return self._env.observation_shape
    
    def num_players(self) -> int:
        return self._env.num_players
    
    @staticmethod
    def _get_state_reward(_state: Any) -> jnp.ndarray:
        return _state.rewards
    
    @staticmethod
    def _get_terminated(_state: Any) -> jnp.ndarray:
        return _state.terminated
    
    @staticmethod
    def _get_legal_action_mask(_state: Any) -> jnp.ndarray:
        return _state.legal_action_mask
    
    @staticmethod
    def _get_cur_player_id(_state: Any) -> jnp.ndarray:
        return _state.current_player
    
    @staticmethod
    def _get_observation(_state: Any) -> jnp.ndarray:
        return _state.observation
    
    def _init(self, key: jax.random.PRNGKey) -> Any:
        return self._env.init(key)
    
    def _step(self, _state: Any, action: jnp.ndarray) -> Any:
        return self._env.step(_state, action)

def make_pgx_env(env_name, **kwargs) -> PgxEnv:
    env = pgx.make(env_name, **kwargs)
    return PgxEnv(env, config=EnvConfig(
        env_pkg='pgx',
        env_name=env_name,
        base_config=kwargs
    ))