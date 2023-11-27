

from typing import Tuple
import pgx
import jax.numpy as jnp
import jax
from core_jax.envs.env import Env, EnvState


class PgxEnv(Env):
    def __init__(self, env):
        super().__init__(
            env = env,
            num_players=env.num_players,
            action_space_dims=(env.num_actions,)
        )
        self._env: PgxEnv

    def step(self, state: EnvState, action: jnp.ndarray) -> Tuple[EnvState, jnp.ndarray]:
        # returns state, observation, reward, terminated
        env_state = self._env.step(state._state, action)
        return state.replace(
            legal_action_mask=env_state.legal_action_mask,
            reward=env_state.rewards,
            cur_player_id=env_state.current_player,
            _state=env_state,
            _observation=env_state.observation,
        ), env_state.terminated

    def reset(self, key: jax.random.PRNGKey) -> Tuple[EnvState, jnp.ndarray]:
        cls_key, base_key = jax.random.split(key)
        env_state = self._env.init(base_key)
        return EnvState(
            key=cls_key,
            legal_action_mask=env_state.legal_action_mask,
            cur_player_id=env_state.current_player,
            reward=env_state.rewards,
            _state=env_state,
            _observation=env_state.observation,
        ), env_state.terminated
        

def make_pgx_env(env_name, *args, **kwargs) -> PgxEnv:
    env = pgx.make(env_name, *args, **kwargs)
    return PgxEnv(env)