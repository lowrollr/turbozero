
from functools import partial
from typing import Tuple
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
    max_steps: int,
    reset: bool = True
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
    truncated = env_state_metadata.step > max_steps 
    
    rewards = env_state_metadata.rewards
    if reset:
        eval_state = jax.lax.cond(
            terminated | truncated,
            evaluator.reset,
            lambda s: evaluator.step(s, output.action),
            output.eval_state
        )
        
        env_state, env_state_metadata = jax.lax.cond(
            terminated | truncated,
            lambda _: env_init_fn(key),
            lambda _: (env_state, env_state_metadata),
            None
        )

    output = output.replace(eval_state=eval_state)
    return output, env_state, env_state_metadata, terminated, truncated, rewards



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
        params=params,
        reset=False
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
    p1_eval_state: chex.ArrayTree
    p2_eval_state: chex.ArrayTree
    p1_value_estimate: chex.Array
    p2_value_estimate: chex.Array
    outcomes: float
    completed: bool

@dataclass(frozen=True)
class GameFrame:
    env_state: chex.ArrayTree
    p1_value_estimate: chex.Array
    p2_value_estimate: chex.Array
    completed: chex.Array
    outcomes: chex.Array

def two_player_game_step(
    state: TwoPlayerGameState,
    p1_evaluator: Evaluator,
    p2_evaluator: Evaluator,
    params: chex.ArrayTree,
    env_step_fn: EnvStepFn,
    env_init_fn: EnvInitFn,
    use_p1: bool,
    max_steps: int
) -> TwoPlayerGameState:
    if use_p1:
        active_evaluator = p1_evaluator
        other_evaluator = p2_evaluator
        active_eval_state = state.p1_eval_state
        other_eval_state = state.p2_eval_state
    else:
        active_evaluator = p2_evaluator
        other_evaluator = p1_evaluator
        active_eval_state = state.p2_eval_state
        other_eval_state = state.p1_eval_state

    step_key, key = jax.random.split(state.key)
    output, env_state, env_state_metadata, terminated, truncated, rewards = step_env_and_evaluator(
        key = step_key,
        env_state = state.env_state,
        env_state_metadata = state.env_state_metadata,
        eval_state = active_eval_state,
        params = params,
        evaluator = active_evaluator,
        env_step_fn = env_step_fn,
        env_init_fn = env_init_fn,
        max_steps = max_steps,
        reset = False
    )

    active_eval_state = output.eval_state
    other_eval_state = other_evaluator.step(other_eval_state, output.action)

    if use_p1:
        p1_eval_state, p2_eval_state = active_eval_state, other_eval_state
    else:
        p1_eval_state, p2_eval_state = other_eval_state, active_eval_state

    return state.replace(
        key = key,
        env_state = env_state,
        env_state_metadata = env_state_metadata,
        p1_eval_state = p1_eval_state,
        p2_eval_state = p2_eval_state,
        p1_value_estimate = p1_evaluator.get_value(p1_eval_state),
        p2_value_estimate = p2_evaluator.get_value(p2_eval_state),
        outcomes=jnp.where(
            ((terminated | truncated) & ~state.completed)[..., None],
            rewards,
            state.outcomes
        ),
        completed = state.completed | terminated | truncated,
    )


def two_player_game(
    key: jax.random.PRNGKey,
    evaluator_1: Evaluator,
    evaluator_2: Evaluator,
    params_1: chex.ArrayTree,
    params_2: chex.ArrayTree,
    env_step_fn: EnvStepFn,
    env_init_fn: EnvInitFn,
    max_steps: int
) -> Tuple[chex.Array, TwoPlayerGameState, chex.Array, chex.Array]:
    """
    Run a two player game between two evaluators
    """
    # init rng
    env_key, eval_key, turn_key, key = jax.random.split(key, 4)
    # init env state
    env_state, metadata = env_init_fn(env_key)    
    # init evaluator states
    eval1_key, eval2_key = jax.random.split(eval_key, 2)
    p1_eval_state = evaluator_1.init(eval1_key, template_embedding=env_state)
    p2_eval_state = evaluator_2.init(eval2_key, template_embedding=env_state)
    # compile step functions
    game_step = partial(two_player_game_step,
        p1_evaluator=evaluator_1,
        p2_evaluator=evaluator_2,
        env_step_fn=env_step_fn,
        env_init_fn=env_init_fn,
        max_steps=max_steps
    )
    step_p1 = partial(game_step, params=params_1, use_p1=True)
    step_p2 = partial(game_step, params=params_2, use_p1=False)
    
    # determine who goes first
    first_player = jax.random.randint(turn_key, (), 0, 2)
    p1_first = first_player == 0

    
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
        p1_eval_state = p1_eval_state,
        p2_eval_state = p2_eval_state,
        p1_value_estimate = jnp.array(0.0, dtype=jnp.float32),
        p2_value_estimate = jnp.array(0.0, dtype=jnp.float32),
        outcomes = jnp.zeros((2,), dtype=jnp.float32),
        completed = jnp.zeros((), dtype=jnp.bool_)
    )     

    initial_game_frame = GameFrame(
        env_state = state.env_state,
        p1_value_estimate = state.p1_value_estimate,
        p2_value_estimate = state.p2_value_estimate,
        completed = state.completed,
        outcomes = state.outcomes
    )

    def step_step(state: TwoPlayerGameState, _) -> TwoPlayerGameState:
        state = jax.lax.cond(
            state.completed,
            lambda s: s,
            lambda s: jax.lax.cond(
                p1_first,
                step_p1,
                step_p2,
                s
            ),
            state
        )
        frame1 = GameFrame(
            env_state = state.env_state,
            p1_value_estimate = state.p1_value_estimate,
            p2_value_estimate = state.p2_value_estimate,
            completed = state.completed,
            outcomes = state.outcomes
        )
        state = jax.lax.cond(
            state.completed,
            lambda s: s,
            lambda s: jax.lax.cond(
                p1_first,
                step_p2,
                step_p1,
                s
            ),
            state
        )
        frame2 = GameFrame(
            env_state = state.env_state,
            p1_value_estimate = state.p1_value_estimate,
            p2_value_estimate = state.p2_value_estimate,
            completed = state.completed,
            outcomes = state.outcomes
        )
        return state, jax.tree_map(lambda x, y: jnp.stack([x, y]), frame1, frame2)
    
    state, frames = jax.lax.scan(
        step_step,
        state,
        xs=jnp.arange(max_steps//2)
    )
    
    frames = jax.tree_map(lambda x: x.reshape(max_steps, *x.shape[2:]), frames)
    # append initial state to front of frames
    frames = jax.tree_map(lambda i, x: jnp.concatenate([jnp.expand_dims(i, 0), x]), initial_game_frame, frames)

    return jnp.array([state.outcomes[p1_id], state.outcomes[p2_id]]), frames, jnp.array([p1_id, p2_id])