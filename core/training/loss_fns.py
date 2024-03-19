
from typing import Callable, Tuple
import chex
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from core.memory.replay_memory import BaseExperience


def az_default_loss_fn(params: chex.ArrayTree, train_state: TrainState, experience: BaseExperience, 
                       l2_reg_lambda: float = 0.0001) -> Tuple[chex.Array, Tuple[chex.ArrayTree, optax.OptState]]:
    
    variables = {'params': params, 'batch_stats': train_state.batch_stats} \
        if hasattr(train_state, 'batch_stats') else {'params': params}
    
    mutables = ['batch_stats'] if hasattr(train_state, 'batch_stats') else []

    (pred_policy, pred_value), updates = train_state.apply_fn(
        variables, 
        x=experience.observation_nn,
        train=True,
        mutable=mutables
    )

    variables = variables.copy(updates)

    pred_policy = jnp.where(
        experience.policy_mask,
        pred_policy,
        jnp.finfo(jnp.float32).min
    )
    policy_loss = optax.softmax_cross_entropy(pred_policy, experience.policy_weights).mean()
    # select appropriate value from experience.reward
    current_player = experience.cur_player_id
    target_value = experience.reward[jnp.arange(experience.reward.shape[0]), current_player]
    value_loss = optax.l2_loss(pred_value.squeeze(), target_value).mean()

    l2_reg = l2_reg_lambda * jax.tree_util.tree_reduce(
        lambda x, y: x + y,
        jax.tree_map(
            lambda x: (x ** 2).sum(),
            params
        )
    )

    loss = policy_loss + value_loss + l2_reg
    aux_metrics = {
        'policy_loss': policy_loss,
        'value_loss': value_loss
    }
    return loss, (aux_metrics, variables)

