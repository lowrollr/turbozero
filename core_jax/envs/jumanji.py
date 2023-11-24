


from typing import Tuple
import jax
import jax.numpy as jnp
from core_jax.envs.env import Env, EnvState 
import jumanji
from jumanji.env import Environment as JEnv, State as JState
from jumanji.types import Observation, TimeStep
from flax import struct

class JumanjiEnv(Env):
    def __init__(self, jumanji_env: JEnv):
        super().__init__(env = jumanji_env)
        self._env: JEnv
        self.num_values = self._env.action_spec().num_values
    
    def step(self, state: JState, actions: jnp.ndarray,) -> Tuple[JState, Observation, jnp.ndarray, jnp.ndarray]:
        # returns state, observation, reward, terminated
        state, timestep = self._env.step(state, actions)
        return state, timestep.observation, timestep.reward, timestep.last()
    
    def reset(self, key: jax.random.PRNGKey) -> Tuple[JState, Observation, jnp.ndarray, jnp.ndarray]:
        state, timestep = self._env.reset(key)
        return state, timestep.observation, timestep.reward, timestep.last()
    
    def reset_if_terminated(self, state: JState, observation: struct.PyTreeNode, reward: jnp.ndarray, terminated: jnp.ndarray, key: jax.random.PRNGKey) -> Tuple[JState, Observation, jnp.ndarray, jnp.ndarray]:
        return jax.lax.cond(
            terminated,
            lambda: self.reset(key),
            lambda: (state, observation, reward, terminated)
        )
    
    def get_legal_actions(self, state: JState, observation: struct.PyTreeNode) -> jnp.ndarray:
        return observation.action_mask
    
    def get_random_legal_action(self, state: JState, observation: struct.PyTreeNode) -> jnp.ndarray:
        key, _ = jax.random.split(state.key)
        action_mask = observation.action_mask.reshape(-1)
        
        action = jax.random.choice(
            key,
            jnp.arange(action_mask.shape[0]),
            p=action_mask
        )

        return jax.lax.scan(
            lambda value, size: (value // size, value % size),
            action,
            self.num_values
        )[1][::-1].reshape(-1)


def make_jumanji_env(env_name, *args, **kwargs) -> JumanjiEnv:
    jenv = jumanji.make(env_name, *args, **kwargs)
    return JumanjiEnv(jenv)
