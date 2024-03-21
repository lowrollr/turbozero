
from typing import Callable, Tuple
import chex
import flax
import jax


def make_nn_eval_fn(
    nn: flax.linen.Module,
    state_to_nn_input_fn: Callable[[chex.ArrayTree], chex.Array]
) -> Callable[[chex.ArrayTree, chex.ArrayTree, chex.PRNGKey], Tuple[chex.Array, chex.Array]]:
    
    def eval_fn(state, params, *args):
        policy_logits, value = nn.apply(params, state_to_nn_input_fn(state)[None,...], train=False)
        return jax.nn.softmax(policy_logits, axis=-1).squeeze(0), \
                value.squeeze()

    return eval_fn

def make_nn_eval_fn_no_params_callable(
    nn: Callable[[chex.Array], Tuple[chex.Array, chex.Array]],
    state_to_nn_input_fn: Callable[[chex.ArrayTree], chex.Array]
) -> Callable[[chex.ArrayTree, chex.ArrayTree, chex.PRNGKey], Tuple[chex.Array, chex.Array]]:
    def eval_fn(state, *args):
        policy_logits, value = nn(state_to_nn_input_fn(state)[None,...])
        return jax.nn.softmax(policy_logits, axis=-1).squeeze(0), \
                value.squeeze()
                
    return eval_fn