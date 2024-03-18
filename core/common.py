
from functools import partial
from typing import Dict, Tuple
import chex
from chex import dataclass

import jax
import jax.numpy as jnp
from core.evaluators.evaluator import EvalOutput, Evaluator
from core.types import EnvInitFn, EnvStepFn, StepMetadata

def partition(
    data: chex.ArrayTree,
    num_partitions: int
) -> chex.ArrayTree:
    """
    Partition an array into num_partitions
    """
    return jax.tree_map(
        lambda x: x.reshape(num_partitions, x.shape[0] // num_partitions, *x.shape[1:]),
        data
    )

def step_env_and_evaluator(
    key: jax.random.PRNGKey,
    env_state: chex.ArrayTree,
    env_state_metadata: StepMetadata,
    eval_state: chex.ArrayTree,
    params: chex.ArrayTree,
    evaluator: Evaluator,
    env_step_fn: EnvStepFn,
    env_init_fn: EnvInitFn,
) -> Tuple[EvalOutput, chex.ArrayTree,  StepMetadata, bool, chex.Array]:
    output = evaluator.evaluate(
        eval_state=eval_state,
        env_state=env_state,
        root_metadata=env_state_metadata,
        params=params,
        env_step_fn=env_step_fn
    )
    env_state, env_state_metadata = env_step_fn(env_state, output.action)
    terminated = env_state_metadata.terminated
    rewards = env_state_metadata.rewards
    eval_state = jax.lax.cond(
        terminated,
        lambda s: evaluator.reset(s),
        lambda s: evaluator.step(s, output.action),
        output.eval_state
    )
    
    env_state, env_state_metadata = jax.lax.cond(
        terminated,
        lambda _: env_init_fn(key),
        lambda _: (env_state, env_state_metadata),
        None
    )

    output = output.replace(eval_state=eval_state)
    return output, env_state, env_state_metadata, terminated, rewards



@dataclass(frozen=True) 
class SinglePlayerGameState:
    key: jax.random.PRNGKey
    env_state: chex.ArrayTree
    env_state_metadata: StepMetadata
    eval_state: chex.ArrayTree
    outcomes: chex.Array
    completed: bool


def single_player_game(
    key: jax.random.PRNGKey,
    evaluator: Evaluator,
    params: chex.ArrayTree,
    env_step_fn: EnvStepFn,
    env_init_fn: EnvInitFn,
) -> chex.Array:
    # init rng
    env_key, eval_key, key = jax.random.split(key, 3)
    # init env state
    env_state, metadata = env_init_fn(env_key)
    # init evaluator state
    eval_state = evaluator.init(eval_key, template_embedding=env_state)
    # compile step function
    step_fn = partial(step_env_and_evaluator,
        env_step_fn=env_step_fn,
        env_init_fn=env_init_fn,
        evaluator=evaluator,
        params=params
    )

    state = SinglePlayerGameState(
        key = key,
        env_state = env_state,
        env_state_metadata = metadata,
        eval_state = eval_state,
        outcomes = jnp.zeros((metadata.num_players,), dtype=jnp.float32),
        completed = jnp.zeros((), dtype=jnp.bool_)
    )

    state = jax.lax.while_loop(
        lambda s: ~s.completed,
        step_fn,
        state
    )

    return state.outcomes


@dataclass(frozen=True)
class TwoPlayerGameState:
    key: jax.random.PRNGKey
    env_state: chex.ArrayTree
    env_state_metadata: StepMetadata
    active_eval_state: chex.ArrayTree
    other_eval_state: chex.ArrayTree
    outcomes: float
    completed: bool


def two_player_game_step(
    state: TwoPlayerGameState,
    active_evaluator: Evaluator,
    other_evaluator: Evaluator,
    params: chex.ArrayTree,
    env_step_fn: EnvStepFn,
    env_init_fn: EnvInitFn,
) -> TwoPlayerGameState:
    step_key, key = jax.random.split(state.key)
    output, env_state, env_state_metadata, terminated, rewards = step_env_and_evaluator(
        key = step_key,
        env_state = state.env_state,
        env_state_metadata = state.env_state_metadata,
        eval_state = state.active_eval_state,
        params = params,
        evaluator = active_evaluator,
        env_step_fn = env_step_fn,
        env_init_fn = env_init_fn
    )
    active_eval_state = output.eval_state
    other_eval_state = other_evaluator.step(state.other_eval_state, output.action)

    return state.replace(
        key = key,
        env_state = env_state,
        env_state_metadata = env_state_metadata,
        active_eval_state = other_eval_state,
        other_eval_state = active_eval_state,
        outcomes=jnp.where(
            (terminated & ~state.completed)[..., None],
            rewards,
            state.outcomes
        ),
        completed = state.completed | terminated,
    )


def two_player_game(
    key: jax.random.PRNGKey,
    evaluator_1: Evaluator,
    evaluator_2: Evaluator,
    params_1: chex.ArrayTree,
    params_2: chex.ArrayTree,
    env_step_fn: EnvStepFn,
    env_init_fn: EnvInitFn,
) -> chex.Array:
    """
    Run a two player game between two evaluators
    """
    # init rng
    env_key, eval_key, turn_key, key = jax.random.split(key, 4)
    # init env state
    env_state, metadata = env_init_fn(env_key)    
    # init evaluator states
    eval1_key, eval2_key = jax.random.split(eval_key, 2)
    evaluator_1_state = evaluator_1.init(eval1_key, template_embedding=env_state)
    evaluator_2_state = evaluator_2.init(eval2_key, template_embedding=env_state)
    # compile step functions
    step_p1 = partial(two_player_game_step,
        active_evaluator=evaluator_1,
        other_evaluator=evaluator_2,
        params=params_1,
        env_step_fn=env_step_fn,
        env_init_fn=env_init_fn,
    )

    step_p2 = partial(two_player_game_step,
        active_evaluator=evaluator_2,
        other_evaluator=evaluator_1,
        params=params_2,
        env_step_fn=env_step_fn,
        env_init_fn=env_init_fn,
    )
    
    # determine who goes first
    first_player = jax.random.randint(turn_key, (), 0, 2)
    p1_first = first_player == 0

    active_state, other_state = jax.lax.cond(
        p1_first,
        lambda _: (evaluator_1_state, evaluator_2_state),
        lambda _: (evaluator_2_state, evaluator_1_state),
        None
    )
    
    p1_id, p2_id = jax.lax.cond(
        p1_first,
        lambda _: (metadata.cur_player_id, 1 - metadata.cur_player_id),
        lambda _: (1 - metadata.cur_player_id, metadata.cur_player_id),
        None
    )

    state = TwoPlayerGameState(
        key = key,
        env_state = env_state,
        env_state_metadata = metadata,
        active_eval_state = active_state,
        other_eval_state = other_state,
        outcomes = jnp.zeros((2,), dtype=jnp.float32),
        completed = jnp.zeros((), dtype=jnp.bool_)
    )     

    def step_step(state: TwoPlayerGameState) -> TwoPlayerGameState:
        return jax.lax.cond(
            p1_first,
            lambda s: step_p2(step_p1(s)),
            lambda s: step_p1(step_p2(s)),
            state
        )
    
    state = jax.lax.while_loop(
        lambda s: ~s.completed,
        step_step,
        state
    )

    return jnp.array([state.outcomes[p1_id], state.outcomes[p2_id]])