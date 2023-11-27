

from typing import Tuple
import pgx
import jax.numpy as jnp
import jax
from pgx.envs.core import State as PgxState, Env as PgxEnv
from core_jax.envs.env import Env


class PgxEnv(Env):
    def __init__(self, env):
        super().__init__(env = env)
        self._env: PgxEnv
        self.num_values = self._env.action_spec().num_values


    def step(self, )
        

    def reset(self, key: jax.random.PRNGKey) -> Tuple[PgxState, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        state, timestep = self._env.reset(key)
        return state, timestep.observation, timestep.reward.reshape(-1), timestep.last()