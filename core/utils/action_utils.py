import jax
import jax.numpy as jnp
from typing import Tuple

def unflatten_action(action: jnp.ndarray, action_space_dims: Tuple[int,...]) -> jnp.ndarray:
        
    return jax.lax.scan(
        lambda value, size: (value // size, value % size),
        action,
        action_space_dims
    )[1][::-1].reshape(-1)