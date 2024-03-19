
from typing import Callable, Tuple
import chex
import flax
import jax


def make_nn_eval_fn(
    nn: flax.linen.Module,
    state_to_nn_input_fn: Callable[[chex.ArrayTree], chex.Array]
) -> Callable[[chex.ArrayTree, chex.ArrayTree, chex.PRNGKey], Tuple[chex.Array, chex.Array]]:
    
    def eval_fn(state, params, rng_key):
        # it's important to package the environement state into a structure that can be consumed by the neural network
        # fortunately, `state.observation` is exactly what we need
        # we will vmap self-play along the batch dimension, so we need to add a dummy batch dimension to the neural network input
        # when defining this function
        # finally, set train=False, we don't want to compute gradients during self-play
        policy_logits, value = nn.apply(params, state_to_nn_input_fn(state)[None,...], train=False)

        # the output should not include the dummy batch dimension
        return jax.nn.softmax(policy_logits, axis=-1).squeeze(0), \
                value.squeeze()

    return eval_fn


