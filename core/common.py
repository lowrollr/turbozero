from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from chex import dataclass

from core.evaluators.evaluator import EvalOutput, Evaluator
from core.types import EnvInitFn, EnvStepFn, StepMetadata


def partition(data: chex.ArrayTree, num_partitions: int) -> chex.ArrayTree:
    """Partition each array in a data structure into num_partitions along the first axis.
    e.g. partitions an array of shape (N, ...) into (num_partitions, N//num_partitions, ...)

    Args:
    - `data`: ArrayTree to partition
    - `num_partitions`: number of partitions

    Returns:
    - (chex.ArrayTree): partitioned ArrayTree
    """
    return jax.tree_map(
        lambda x: x.reshape(num_partitions, x.shape[0] // num_partitions, *x.shape[1:]),
        data,
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
    reset: bool = True,
) -> Tuple[EvalOutput, chex.ArrayTree, StepMetadata, bool, bool, chex.Array]:
    """
    - Evaluates the environment state with the Evaluator and selects an action.
    - Performs a step in the environment with the selected action.
    - Updates the internal state of the Evaluator.
    - Optionally resets the environment and evaluator state if the episode is terminated or truncated.

    Args:
    - `key`: rng
    - `env_state`: The environment state to evaluate.
    - `env_state_metadata`: Metadata associated with the environment state.
    - `eval_state`: The internal state of the Evaluator.
    - `params`: nn parameters used by the Evaluator.
    - `evaluator`: The Evaluator.
    - `env_step_fn`: The environment step function.
    - `env_init_fn`: The environment initialization function.
    - `max_steps`: The maximum number of environment steps per episode.
    - `reset`: Whether to reset the environment and evaluator state if the episode is terminated or truncated.

    Returns:
    - (EvalOutput, chex.ArrayTree, StepMetadata, bool, bool, chex.Array)
        - `output`: The output of the evaluation.
        - `env_state`: The updated environment state.
        - `env_state_metadata`: Metadata associated with the updated environment state.
        - `terminated`: Whether the episode is terminated.
        - `truncated`: Whether the episode is truncated.
        - `rewards`: Rewards emitted by the environment.
        - `state_evaluation`: The evaluator's evaluation of the environment state prior to taking the chosen action.
    """
    key, evaluate_key = jax.random.split(key)
    # evaluate the environment state
    output = evaluator.evaluate(
        key=evaluate_key,
        eval_state=eval_state,
        env_state=env_state,
        root_metadata=env_state_metadata,
        params=params,
        env_step_fn=env_step_fn,
    )
    state_evaluation = evaluator.get_value(output.eval_state)
    # take the selected action
    env_state, env_state_metadata = env_step_fn(env_state, output.action)
    # check for termination and truncation
    terminated = env_state_metadata.terminated
    truncated = env_state_metadata.step > max_steps
    # reset the environment and evaluator state if the episode is terminated or truncated
    # else, update the evaluator state
    rewards = env_state_metadata.rewards
    eval_state = jax.lax.cond(
        terminated | truncated,
        evaluator.reset if reset else lambda s: s,
        lambda s: evaluator.step(s, output.action),
        output.eval_state,
    )
    # reset the environment if the episode is terminated or truncated
    env_state, env_state_metadata = jax.lax.cond(
        terminated | truncated,
        lambda _: env_init_fn(key) if reset else (env_state, env_state_metadata),
        lambda _: (env_state, env_state_metadata),
        None,
    )
    output = output.replace(eval_state=eval_state)
    return (
        output,
        env_state,
        env_state_metadata,
        terminated,
        truncated,
        rewards,
        state_evaluation,
    )


@dataclass(frozen=True)
class GameState:
    key: jax.random.PRNGKey
    env_state: chex.ArrayTree
    env_state_metadata: StepMetadata
    eval_states: chex.ArrayTree
    value_estimates: chex.Array
    outcomes: chex.Array
    completed: bool


@dataclass(frozen=True)
class GameFrame:
    env_state: chex.ArrayTree
    value_estimates: chex.Array
    outcomes: chex.Array
    completed: chex.Array


def n_player_game(
    key: jax.random.PRNGKey,
    evaluators: Tuple[Evaluator],
    params: Tuple[chex.ArrayTree],
    env_step_fn: EnvStepFn,
    env_init_fn: EnvInitFn,
    max_steps: int,
) -> Tuple[chex.Array, GameState, chex.Array]:
    assert len(evaluators) == len(params)
    num_evaluators = len(evaluators)
    # init rng
    env_key, turn_key, key = jax.random.split(key, 3)
    # init env state
    env_state, metadata = env_init_fn(env_key)
    assert metadata.rewards.shape == (num_evaluators,)

    # determine who goes first
    first_player = jax.random.randint(turn_key, (), 0, num_evaluators)
    eval_states = tuple(
        evaluator.init(template_embedding=env_state) for evaluator in evaluators
    )
    player_ordering = jnp.array(
        [(metadata.cur_player_id + i) % num_evaluators for i in range(num_evaluators)]
    )

    def take_each_turn(state: GameState, turn_idx: int) -> Tuple[GameState, GameFrame]:
        frames = []
        for idx, evaluator in enumerate(evaluators):
            active_eval_state = state.eval_states[idx]
            step_key, key = jax.random.split(state.key)
            # step active evaluator
            (
                output,
                env_state,
                env_state_metadata,
                terminated,
                truncated,
                rewards,
                active_value_estimate,
            ) = step_env_and_evaluator(
                key=step_key,
                env_state=state.env_state,
                env_state_metadata=state.env_state_metadata,
                eval_state=active_eval_state,
                params=params[idx],
                evaluator=evaluator,
                env_step_fn=env_step_fn,
                env_init_fn=env_init_fn,
                max_steps=max_steps,
                reset=False,
            )

            # step other evaluators
            new_eval_states = []
            new_value_estimates = []
            for i, evaluator in enumerate(evaluators):
                if idx == i:
                    new_eval_states.append(active_eval_state)
                    new_value_estimates.append(active_value_estimate)
                    continue
                new_state = evaluator.step(state.eval_states[i], output.action)
                new_value_estimate = evaluator.get_value(new_state)
                new_eval_states.append(new_state)
                new_value_estimates.append(new_value_estimate)

            new_state = state.replace(
                key=key,
                env_state=env_state,
                env_state_metadata=env_state_metadata,
                eval_states=tuple(new_eval_states),
                value_estimates=jnp.array(new_value_estimates),
                outcomes=jnp.where(
                    ((terminated | truncated) & ~state.completed)[..., None],
                    rewards[player_ordering],
                    state.outcomes,
                ),
                completed=state.completed | terminated | truncated,
            )
            invalid_step = ((turn_idx == 0) & (first_player > idx)) | (
                ((turn_idx * num_evaluators) + idx - first_player) > max_steps
            )
            state = jax.lax.cond(
                invalid_step,
                lambda _: state,
                lambda _: new_state,
                None,
            )
            frames.append(
                GameFrame(
                    env_state=state.env_state,
                    value_estimates=state.value_estimates,
                    outcomes=state.outcomes,
                    completed=state.completed,
                )
            )
        return state, jax.tree_map(
            lambda *xs: jnp.stack(xs),
            *frames,
        )

    # init game state
    state = GameState(
        key=key,
        env_state=env_state,
        env_state_metadata=metadata,
        eval_states=eval_states,
        value_estimates=jnp.array(
            [
                evaluator.get_value(eval_state)
                for eval_state, evaluator in zip(eval_states, evaluators)
            ]
        ),
        outcomes=jnp.zeros((num_evaluators,), dtype=jnp.float32),
        completed=jnp.zeros((), dtype=jnp.bool_),
    )

    # make initial render frame
    initial_game_frame = GameFrame(
        env_state=state.env_state,
        value_estimates=state.value_estimates,
        outcomes=state.outcomes,
        completed=state.completed,
    )

    # play the game
    state, frames = jax.lax.scan(
        take_each_turn, state, jnp.arange(max_steps // num_evaluators)
    )
    frames = jax.tree_map(
        lambda x: x.reshape(max_steps, *x.shape[2:]),
        frames,
    )
    # append initial state to front of frames
    frames = jax.tree_map(
        lambda i, x: jnp.concatenate([jnp.expand_dims(i, 0), x]),
        initial_game_frame,
        frames,
    )

    # return outcome, frames, player ids
    player_ids = jnp.array(
        [(first_player + i) % num_evaluators for i in range(num_evaluators)]
    )
    return (
        jnp.array([state.outcomes[player_ids[i]] for i in range(num_evaluators)]),
        frames,
        jnp.array(player_ids),
    )
