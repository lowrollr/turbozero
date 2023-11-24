from typing import Tuple
import jax
import jax.numpy as jnp
from flax import struct

@struct.dataclass
class EnvState:
    action_mask: jnp.ndarray
    key: jnp.ndarray
    
class Env:
    def __init__(self, env, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self._env = env

    def step(self, state: struct.PyTreeNode, actions: jnp.ndarray) -> Tuple[struct.PyTreeNode, struct.PyTreeNode, jnp.ndarray, jnp.ndarray]:
        # returns state, observation, reward, terminated
        raise NotImplementedError()

    def reset(self, key: jax.random.PRNGKey) -> Tuple[struct.PyTreeNode, struct.PyTreeNode, jnp.ndarray, jnp.ndarray]:
        raise NotImplementedError()

    def reset_if_terminated(self, state: struct.PyTreeNode, observation: struct.PyTreeNode, reward: jnp.ndarray, terminated: jnp.ndarray, key: jax.random.PRNGKey) -> Tuple[struct.PyTreeNode, struct.PyTreeNode, jnp.ndarray, jnp.ndarray]:        
        raise NotImplementedError()
    

    def get_legal_actions(self, state: struct.PyTreeNode, observation: struct.PyTreeNode) -> jnp.ndarray:
        raise NotImplementedError()
    
    def get_random_legal_action(self, state: struct.PyTreeNode, observation: struct.PyTreeNode) -> jnp.ndarray:
        raise NotImplementedError()