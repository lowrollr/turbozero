


from typing import Tuple
import jax
import jax.numpy as jnp
from core_jax.envs.env import Env, EnvState
import jumanji
from jumanji.env import Environment as JEnv
from flax import struct

from core_jax.utils.action_utils import unflatten_action

class JumanjiEnv(Env):
    def __init__(self, jumanji_env: JEnv):
        dims = jumanji_env.action_spec().num_values
        self.unflatten = unflatten_action 
        if type(dims) == int:
            dims = (dims,)
            self.unflatten = lambda x, _: x

        super().__init__(
            env = jumanji_env,
            action_space_dims = dims,
            num_players = 1
        )
        self._env: JEnv
        
    
    def step(self, state: EnvState, action: jnp.ndarray,) -> Tuple[EnvState, jnp.ndarray]:
        # returns state, observation, reward, terminated
        env_state, timestep = self._env.step(state._state, self.unflatten(action, self.action_space_dims))
        return state.replace(
            legal_action_mask=timestep.observation.action_mask,
            reward=timestep.reward.reshape(-1),
            _state=env_state,
            _observation=timestep.observation,
        ), timestep.last()
    
    def reset(self, key: jax.random.PRNGKey) -> Tuple[EnvState, jnp.ndarray]:
        cls_key, base_key = jax.random.split(key)
        env_state, timestep = self._env.reset(base_key)
        return EnvState(
            key=cls_key,
            legal_action_mask=timestep.observation.action_mask,
            reward=timestep.reward.reshape(-1),
            cur_player_id = jnp.array([0]),
            _state=env_state,
            _observation=timestep.observation,
        ), timestep.last()


    def reset_if_terminated(self, state: EnvState, terminated: jnp.ndarray) -> Tuple[EnvState, jnp.ndarray]:
        reset_key, new_key = jax.random.split(state.key)
        reset_state, terminated = jax.lax.cond(
            terminated,
            lambda: self.reset(reset_key),
            lambda: (state, terminated)
        )

        return reset_state.replace(key=new_key), terminated
    
    


def make_jumanji_env(env_name, *args, **kwargs) -> JumanjiEnv:
    jenv = jumanji.make(env_name, *args, **kwargs)
    return JumanjiEnv(jenv)
