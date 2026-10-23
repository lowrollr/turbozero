"""Packing test episodes for the monitor, which the dashboard draws in the browser.

A training run packs an episode's raw arrays into an `Episode` (with a packer here,
passed to a tester as its `episode_fn`); the server stores them as they are, and the
dashboard plays them with its renderer for the episode's env (static/renderers.js), so
neither the training run nor the server draws anything.

When the test recorded its searches (`two_player_game`'s `record_search`), the episode also
carries what each player's search explored at each position, for the dashboard's search
explorer: every node's statistics and, replayed through the environment here, its board.
"""

from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from core.trees.tree import Tree
from core.types import EnvStepFn
from monitor.client import Episode


def _othello(x: Any) -> tuple[np.ndarray, np.ndarray]:
    # the board is +1 for the player to move and -1 for the other; turn 0 is black's
    to_move = np.asarray(x.turn)
    board = np.where(to_move[:, None] == 0, x.board, -x.board)
    return np.select([board > 0, board < 0], [1, 2], 0).reshape(-1, 8, 8), to_move


def _tic_tac_toe(x: Any) -> tuple[np.ndarray, np.ndarray]:
    # -1 for empty, else the colour (0 is X)
    return (np.asarray(x.board) + 1).reshape(-1, 3, 3), np.asarray(x.color)


def _connect_four(x: Any) -> tuple[np.ndarray, np.ndarray]:
    # rows from the top, -1 for empty, else the colour (0 moves first)
    return (np.asarray(x.board) + 1).reshape(-1, 6, 7), np.asarray(x.color)


# pgx env id -> the boards of a game's states, (frames, rows, cols) of 0 for empty, 1
# for a piece of the colour that moves first in the game's usual opening (black, X)
# and 2 for the other's, and the colour to move in each, 0 or 1
BOARDS: dict[str, Callable[[Any], tuple[np.ndarray, np.ndarray]]] = {
    "othello": _othello,
    "tic_tac_toe": _tic_tac_toe,
    "connect_four": _connect_four,
}


@partial(jax.jit, static_argnums=0)
def _replay(
    env_step_fn: EnvStepFn, roots: Any, paths: jax.Array, depths: jax.Array
) -> Any:
    """The states `depths` moves along `paths` (one action per move) from `roots`."""

    def replay(root, path, depth):
        def move(state, d):
            stepped, _ = env_step_fn(state, path[d])
            return jax.tree.map(
                lambda a, b: jnp.where(d < depth, a, b), stepped, state
            ), None

        return jax.lax.scan(move, root, jnp.arange(path.shape[0]))[0]

    return jax.vmap(replay)(roots, paths, depths)


def _bucket(n: int) -> int:
    """`n` rounded up to a power of two, so replays of similar size share a compilation."""
    return 1 << max(0, n - 1).bit_length()


def _paths(parents: np.ndarray, actions: np.ndarray) -> list[list[int]]:
    """Each node's actions from the root. A node can come before its parent in the tree (when it
    took an evicted node's slot), so each path is built from its parent's once that's done."""
    paths: list[list[int] | None] = [None] * len(parents)
    paths[Tree.ROOT_INDEX] = []

    def path(i: int) -> list[int]:
        if paths[i] is None:
            paths[i] = path(int(parents[i])) + [int(actions[i])]
        return paths[i]  # type: ignore[return-value]

    return [path(i) for i in range(len(parents))]


def _searches(
    env_id: str,
    env_state: Any,
    searches: tuple[Any, Any],
    num_moves: int,
    env_step_fn: EnvStepFn,
) -> dict[str, np.ndarray]:
    """Each player's record of each position played from (see `GameFrame.p1_search`), as the
    dashboard's search explorer reads them: one search per (position, player) with a tree, its
    nodes in a flat list, each with its board replayed through `env_step_fn`.

    Args:
        env_id: the pgx env, which names its board (`BOARDS`)
        env_state: the episode's states, stacked along a leading time axis
        searches: the agent's and the opponent's records, stacked along the same axis
        num_moves: the moves played
        env_step_fn: the environment step function, which the searches used
    """
    found, roots, paths, offsets = [], [], [], [0]
    for t in range(num_moves):
        # the frame after a move holds what each player made of the position it was played from
        for player, records in enumerate(searches):
            if records is None:
                continue
            record = jax.tree.map(lambda x, t=t: np.asarray(x[t + 1]), records)
            k = int(record.num_nodes)
            if k == 0:
                continue
            found.append((t, player, record))
            paths += _paths(record.parents[:k], record.actions[:k])
            roots += [t] * k
            offsets.append(offsets[-1] + k)
    if not found:
        return {}
    num_nodes = offsets[-1]
    depth = max(map(len, paths))
    # padded to sizes shared with other episodes, so most tests don't compile a new replay
    size, depth = _bucket(num_nodes), _bucket(max(depth, 1))
    padded = np.zeros((size, depth), dtype=np.int32)
    depths = np.zeros(size, dtype=np.int32)
    for i, path in enumerate(paths):
        padded[i, : len(path)] = path
        depths[i] = len(path)
    roots = np.pad(np.array(roots), (0, size - num_nodes))
    states = _replay(
        env_step_fn, jax.tree.map(lambda x: x[roots], env_state), padded, depths
    )
    states = jax.device_get(jax.tree.map(lambda x: x[:num_nodes], states))
    board, to_move = BOARDS[env_id](states._x)
    player = np.asarray(states.current_player)

    sizes = np.diff(offsets)
    stats = {
        name: np.concatenate(
            [getattr(record, name)[:k] for (_, _, record), k in zip(found, sizes)]
        )
        for name in ("parents", "actions", "n", "q", "prior", "terminated")
    }
    # each node's value for the player who chose it, its parent's player to move (the root's,
    # for the player to move at the root): q is for the player to move at the node
    parent = stats["parents"] + np.repeat(offsets[:-1], sizes)
    chooser = np.where(stats["parents"] < 0, player, player[np.maximum(parent, 0)])
    value = np.where(chooser == player, stats["q"], -stats["q"])
    return {
        "search_frame": np.array([t for t, _, _ in found]),
        # 0 for the trained agent's, 1 for its opponent's
        "search_player": np.array([p for _, p, _ in found]),
        "search_offset": np.array(offsets),
        # the root's policy, over every action
        "search_policy": np.stack([record.policy for _, _, record in found]),
        # local to the node's search: -1 for the root
        "node_parent": stats["parents"],
        "node_action": stats["actions"],
        "node_visits": stats["n"],
        "node_value": value,
        "node_prior": stats["prior"],
        "node_terminal": stats["terminated"],
        "node_board": board.astype(np.uint8),
        "node_to_move": to_move,
        "node_legal": np.asarray(states.legal_action_mask),
    }


def pgx_two_player_episode() -> Callable[[Any, Any, EnvStepFn], Episode]:
    """An `episode_fn` for testers of two-player pgx board games (those in `BOARDS`):
    packs the first test episode for the dashboard to play, its env named by the game.

    Returns:
        Callable: `episode_fn(frames, p_ids, env_step_fn) -> Episode`, see `BaseTester`
    """

    def episode_fn(frames, p_ids, env_step_fn) -> Episode:
        env_state = frames.env_state
        env_id = str(env_state.env_id)
        if env_id not in BOARDS:
            known = ", ".join(sorted(BOARDS))
            raise ValueError(f"no board for pgx env {env_id!r} (known: {known})")
        board, to_move = BOARDS[env_id](env_state._x)
        completed = np.asarray(frames.completed, dtype=bool)
        # frames after the game ended repeat its last state
        num_frames = (
            int(np.argmax(completed)) + 1 if completed.any() else len(completed)
        )
        agent, opponent = (int(p) for p in p_ids)
        # the trained agent's colour: the one to move first, if it moves first
        first = int(to_move[0])
        moves_first = int(np.asarray(env_state.current_player)[0]) == agent
        outcomes = np.asarray(frames.outcomes)[num_frames - 1]
        searches = _searches(
            env_id,
            env_state,
            (frames.p1_search, frames.p2_search),
            num_frames - 1,
            env_step_fn,
        )
        return Episode(
            env_id,
            board=board[:num_frames],
            to_move=to_move[:num_frames],
            agent=first if moves_first else 1 - first,
            # each one's value estimate of the position, from its own side
            values=np.stack(
                [frames.p1_value_estimate, frames.p2_value_estimate], axis=-1
            )[:num_frames],
            completed=completed.any(),
            # the trained agent's result once the game is over (or cut off): 1 for a
            # win, -1 for a loss, 0 for a draw
            outcome=np.sign(outcomes[agent] - outcomes[opponent]),
            **searches,
        )

    return episode_fn
