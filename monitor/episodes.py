"""Packing test episodes for the monitor, which the dashboard draws in the browser.

A training run packs an episode's raw arrays into an `Episode` (with a packer here,
passed to a tester as its `episode_fn`); the server stores them as they are, and the
dashboard plays them with its renderer for the episode's env (static/renderers.js), so
neither the training run nor the server draws anything.
"""

from collections.abc import Callable
from typing import Any

import numpy as np

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


def pgx_two_player_episode() -> Callable[[Any, Any], Episode]:
    """An `episode_fn` for testers of two-player pgx board games (those in `BOARDS`):
    packs the first test episode for the dashboard to play, its env named by the game.

    Returns:
        Callable: `episode_fn(frames, p_ids) -> Episode`, see `BaseTester`
    """

    def episode_fn(frames, p_ids) -> Episode:
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
        )

    return episode_fn
