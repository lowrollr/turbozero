"""The run monitor: the client logging to a live server, and the server storing and serving runs."""

import functools
import json
import queue
import threading
import urllib.error
import urllib.request

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from core.common import GameFrame
from core.evaluators.mcts.state import SearchRecord
from monitor import Episode, Monitor, client
from monitor.episodes import BOARDS, line_record, pgx_two_player_episode, with_opening

# one compilation of each environment's init and step, whichever test steps it
jit = functools.cache(jax.jit)


def test_run_round_trip(monitor_server):
    monitor = Monitor(monitor_server.url, project="tests", name="round-trip")
    monitor.start(
        config={"batch_size": 4, "run": {"seed": 3, "num_epochs": 2}, "fn": np.mean}
    )
    monitor.log(
        0, {"loss": np.float32(1.5), "policy_loss": 2, "ignored": "not a metric"}
    )
    monitor.log(1, {"loss": 0.5})
    monitor.finish()

    [run] = monitor_server.get("/api/runs")
    assert (run["project"], run["name"], run["status"]) == (
        "tests",
        "round-trip",
        "finished",
    )
    assert (run["step"], run["seed"], run["num_epochs"]) == (1, 3, 2)
    meta = monitor_server.get(f"/api/runs/{monitor.run_id}")
    # functions in the config are recorded by name
    assert meta["config"]["fn"] == "mean"
    rows = monitor_server.get(f"/api/runs/{monitor.run_id}/metrics")["rows"]
    assert [(r["step"], r["loss"]) for r in rows] == [(0, 1.5), (1, 0.5)]
    assert rows[0]["policy_loss"] == 2 and "ignored" not in rows[0]


def test_log_does_not_wait_on_the_server(monitor_server, monkeypatch):
    monitor = Monitor(monitor_server.url)
    monitor.start()
    send = monitor._request
    server_is_back = threading.Event()

    def stalled_request(*args, **kwargs):
        server_is_back.wait(timeout=10)
        return send(*args, **kwargs)

    monkeypatch.setattr(monitor, "_request", stalled_request)

    # returns while the server isn't answering
    monitor.log(0, {"loss": 1.0})
    assert monitor_server.get(f"/api/runs/{monitor.run_id}/metrics")["rows"] == []
    server_is_back.set()
    monitor.flush()

    assert len(monitor_server.get(f"/api/runs/{monitor.run_id}/metrics")["rows"]) == 1


def test_metrics_since(monitor_server):
    monitor = Monitor(monitor_server.url)
    monitor.start()
    for step in range(3):
        monitor.log(step, {"loss": step})
    monitor.flush()

    page = monitor_server.get(f"/api/runs/{monitor.run_id}/metrics?since=2")

    assert [r["step"] for r in page["rows"]] == [2]
    assert page["next"] == 3


def test_non_finite_metrics_are_stored_as_null(monitor_server):
    monitor = Monitor(monitor_server.url)
    monitor.start()

    monitor.log(0, {"loss": float("nan"), "value_loss": np.inf, "policy_loss": 1.0})
    monitor.flush()

    [row] = monitor_server.get(f"/api/runs/{monitor.run_id}/metrics")["rows"]
    assert (
        row["loss"] is None and row["value_loss"] is None and row["policy_loss"] == 1.0
    )


def test_dicts_are_stored_as_data(monitor_server):
    monitor = Monitor(monitor_server.url)
    monitor.start()

    monitor.log(4, {"openings": {"e6": 3, "f5": 1}, "not_data": object()})
    monitor.flush()

    [entry] = monitor_server.get(f"/api/runs/{monitor.run_id}/media")["rows"]
    assert (entry["key"], entry["step"], entry["file"]) == (
        "openings",
        4,
        "openings-4.json",
    )
    assert monitor_server.get(f"/media/{monitor.run_id}/openings-4.json") == {
        "e6": 3,
        "f5": 1,
    }


def test_pictures_are_refused(monitor_server):
    # runs log data; drawing it is the dashboard's job
    monitor = Monitor(monitor_server.url)
    monitor.start()

    request = urllib.request.Request(
        f"{monitor_server.url}/api/runs/{monitor.run_id}/media?key=board&step=0",
        data=b"GIF89a",
        headers={"Content-Type": "image/gif"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(request)
    assert e.value.code == 415


def load_arrays(body: bytes) -> dict[str, np.ndarray]:
    """An episode as the dashboard gets it (static/episodes.js loadArrays)."""
    length = int.from_bytes(body[:4], "little")
    header = json.loads(body[4 : 4 + length])
    return {
        name: np.frombuffer(
            body,
            dtype=spec["dtype"],
            count=int(np.prod(spec["shape"])),
            offset=4 + length + spec["offset"],
        ).reshape(spec["shape"])
        for name, spec in header.items()
    }


def stacked(states):
    """States stacked along a leading time axis."""
    return jax.tree.map(lambda *xs: np.stack(xs), *states)


def tic_tac_toe_frames(ttt, moves, num_frames, key, p1_search=None, p2_search=None):
    """The frames `two_player_game` records of tic-tac-toe played with `moves` from the initial
    state from `key`: the initial state and the state after each move, then the last one again
    up to `num_frames`, as after a game ends. The first evaluator's value estimates are 0,
    0.1, 0.2, ..., the second's their negatives."""
    states = [jit(ttt.env.init)(key)]
    for action in moves:
        states.append(jit(ttt.env.step)(states[-1], action))
    states += [states[-1]] * (num_frames - len(states))
    completed = np.array([bool(state.terminated) for state in states])
    values = np.arange(num_frames) / 10
    return GameFrame(
        env_state=stacked(states),
        p1_value_estimate=jnp.asarray(values),
        p2_value_estimate=jnp.asarray(-values),
        completed=jnp.asarray(completed),
        # each player's reward from the step that ends the game on
        outcomes=jnp.asarray(
            np.where(completed[:, None], np.asarray(states[-1].rewards), 0.0)
        ),
        p1_search=p1_search,
        p2_search=p2_search,
    )


def test_pgx_episode_is_served_for_the_dashboard(monitor_server, ttt):
    # playing the first free square, X wins on the diagonal with the 7th move; the trained
    # agent (the first evaluator) moved second, so played O, and lost
    frames = tic_tac_toe_frames(ttt, range(7), num_frames=10, key=jax.random.PRNGKey(0))
    agent = 1 - int(frames.env_state.current_player[0])
    monitor = Monitor(monitor_server.url)
    monitor.start()

    episode = pgx_two_player_episode()(
        frames, np.array([agent, 1 - agent]), ttt.step_fn
    )
    monitor.log(2, {"game": episode})
    monitor.flush()

    [entry] = monitor_server.get(f"/api/runs/{monitor.run_id}/media")["rows"]
    assert (entry["key"], entry["step"], entry["env"]) == ("game", 2, "tic_tac_toe")
    with urllib.request.urlopen(
        f"{monitor_server.url}/api/runs/{monitor.run_id}/episodes/{entry['episode']}"
    ) as resp:
        arrays = load_arrays(resp.read())
    # the initial board and 7 moves, without the frames after the game ended
    board = arrays["board"]
    assert board.shape == (1 + 7, 3, 3)
    assert (board[0] == 0).all()
    assert board[-1].tolist() == [[1, 2, 1], [2, 1, 2], [1, 0, 0]]
    assert arrays["to_move"].tolist() == [0, 1, 0, 1, 0, 1, 0, 1]
    np.testing.assert_allclose(arrays["values"][:, 0], np.arange(8) / 10)
    np.testing.assert_allclose(arrays["values"][:, 1], -np.arange(8) / 10)
    assert (int(arrays["agent"]), int(arrays["completed"])) == (1, 1)
    assert float(arrays["outcome"]) == -1
    # the game didn't record its searches
    assert "search_offset" not in arrays


def search_record(*nodes, capacity=4):
    """A `SearchRecord` of tic-tac-toe of a hand-built tree: `nodes` are (parent, action, q),
    the root's first, with parent and action -1; the rest of the tree's capacity is unused. The
    node at index i has i + 1 visits, a prior of 1 / (i + 1), and isn't terminal."""
    unused = [(-1, -1, 0.0)] * (capacity - len(nodes))
    parents, actions, q = np.array(list(nodes) + unused).T
    visits = np.arange(1, capacity + 1)
    return SearchRecord(
        parents=jnp.asarray(parents, dtype=jnp.int32),
        actions=jnp.asarray(actions, dtype=jnp.int32),
        n=jnp.asarray(visits, dtype=jnp.int32),
        q=jnp.asarray(q, dtype=jnp.float32),
        prior=jnp.asarray(1 / visits, dtype=jnp.float32),
        terminated=jnp.asarray(np.zeros(capacity, dtype=bool)),
        policy=jnp.asarray(np.full(9, 1 / 9), dtype=jnp.float32),
        num_nodes=jnp.asarray(len(nodes), dtype=jnp.int32),
    )


def test_pgx_episode_carries_both_players_searches(ttt):
    # the trained agent moves first (X) to the centre, then its opponent to the corner. Frame
    # t + 1 holds what each player made of the position at t: the agent searched both
    # positions (keeping its tree of the second), its opponent only the second, so had no
    # tree of the first
    no_tree = search_record()
    agent_searches = [
        no_tree,
        # its first node took an evicted node's slot, before its parent
        search_record((-1, -1, 0.1), (2, 0, 0.3), (0, 4, -0.2)),
        search_record((-1, -1, 0.2), (0, 0, 0.4)),
    ]
    opponent_searches = [
        no_tree,
        no_tree,
        search_record((-1, -1, -0.2), (0, 8, 0.6)),
    ]
    frames = tic_tac_toe_frames(
        ttt,
        [4, 0],
        num_frames=3,
        key=jax.random.PRNGKey(0),
        p1_search=stacked(agent_searches),
        p2_search=stacked(opponent_searches),
    )
    agent = int(frames.env_state.current_player[0])

    episode = pgx_two_player_episode()(
        frames, np.array([agent, 1 - agent]), ttt.step_fn
    )

    arrays = episode.data
    assert arrays["search_frame"].tolist() == [0, 1, 1]
    assert arrays["search_player"].tolist() == [0, 0, 1]
    assert arrays["search_offset"].tolist() == [0, 3, 5, 7]
    assert arrays["node_parent"].tolist() == [-1, 2, 0, -1, 0, -1, 0]
    assert arrays["node_action"].tolist() == [-1, 0, 4, -1, 0, -1, 8]
    # each node's board, replayed from the position searched: 1 for X, 2 for O
    empty, centre = [0] * 9, [0, 0, 0, 0, 1, 0, 0, 0, 0]
    corner, other_corner = [2, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 2]
    expected = [empty, corner, centre, centre, corner, centre, other_corner]
    assert arrays["node_board"].dtype == np.uint8
    assert arrays["node_board"].reshape(7, 9).tolist() == expected
    assert arrays["node_to_move"].tolist() == [0, 0, 1, 1, 0, 1, 0]
    np.testing.assert_array_equal(
        arrays["node_legal"], arrays["node_board"].reshape(7, 9) == 0
    )
    # each node's value for the player who chose its move: the root's own, a child's its
    # parent's, the opposite of its q (players alternate in tic-tac-toe)
    np.testing.assert_allclose(
        arrays["node_value"], [0.1, -0.3, 0.2, 0.2, -0.4, -0.2, -0.6], rtol=1e-6
    )
    # the rest of each node's statistics as they are
    assert arrays["node_visits"].tolist() == [1, 2, 3, 1, 2, 1, 2]
    np.testing.assert_allclose(
        arrays["node_prior"], 1 / np.array([1, 2, 3, 1, 2, 1, 2]), rtol=1e-6
    )
    assert not arrays["node_terminal"].any()
    assert arrays["search_policy"].shape == (3, 9)


def test_pgx_episode_carries_an_engines_lines(ttt):
    # the trained agent moves first (X) to the centre, then an engine that only scores moves
    # (O) to the corner. It scored two: the corner, expecting the game to go 0, 8, 2, worth 0.5
    # to it, and the edge, 1, worth at most -0.25, from a search 3 moves deep. It scored
    # nothing of the agent's position
    lines = functools.partial(line_record, num_actions=9, capacity=6)
    no_tree = search_record()
    frames = tic_tac_toe_frames(
        ttt,
        [4, 0],
        num_frames=3,
        key=jax.random.PRNGKey(0),
        p1_search=stacked([no_tree, search_record((-1, -1, 0.1)), no_tree]),
        p2_search=stacked(
            [
                lines(None),
                lines(None),
                lines([([0, 8, 2], 0.5, 0), ([1], -0.25, -1)], 3),
            ]
        ),
    )
    agent = int(frames.env_state.current_player[0])

    episode = pgx_two_player_episode()(
        frames, np.array([agent, 1 - agent]), ttt.step_fn
    )

    arrays = episode.data
    assert arrays["search_frame"].tolist() == [0, 1]
    assert arrays["search_player"].tolist() == [0, 1]
    # a tree's search has no depth
    assert arrays["search_depth"].tolist() == [-1, 3]
    assert arrays["search_offset"].tolist() == [0, 1, 6]
    # a line per move scored, a node per move of it, each the last one's child
    assert arrays["node_parent"][1:].tolist() == [-1, 0, 1, 2, 0]
    assert arrays["node_action"][1:].tolist() == [-1, 0, 8, 2, 1]
    assert arrays["node_board"][4].reshape(9).tolist() == [2, 0, 2, 0, 1, 0, 0, 0, 1]
    assert arrays["node_board"][5].reshape(9).tolist() == [0, 2, 0, 0, 1, 0, 0, 0, 0]
    # each node's value for the player who chose it: the root's the best line's, a line's
    # own moves its value, the agent's the opposite
    np.testing.assert_allclose(arrays["node_value"][1:], [0.5, 0.5, -0.5, 0.5, -0.25])
    # the edge's value is at most that; the corner's line's, and the agent's tree's, exact
    assert arrays["node_bound"].tolist() == [0, 0, 0, 0, 0, -1]
    # no visits or priors
    assert not arrays["node_visits"][1:].any() and not arrays["node_prior"][1:].any()


def test_pgx_episode_keeps_the_opening_played_for_its_players(ttt):
    # X's first move to the centre was played for the players, who started from the position
    # after it: put back before their game, the opening's position is the episode's first,
    # and the dashboard starts at theirs
    whole = tic_tac_toe_frames(ttt, [4, 0, 8], num_frames=4, key=jax.random.PRNGKey(0))
    played = jax.tree.map(lambda x: x[1:], whole)
    opening = [jax.tree.map(lambda x: x[0], whole.env_state)]
    agent = int(played.env_state.current_player[0])

    episode = pgx_two_player_episode()(
        with_opening(played, opening),
        np.array([agent, 1 - agent]),
        ttt.step_fn,
        start=1,
    )

    expected = pgx_two_player_episode()(
        whole, np.array([agent, 1 - agent]), ttt.step_fn
    )
    np.testing.assert_array_equal(episode.data["board"], expected.data["board"])
    np.testing.assert_array_equal(episode.data["to_move"], expected.data["to_move"])
    assert int(episode.data["agent"]) == int(expected.data["agent"])
    # nobody had a value of the opening's position
    assert episode.data["values"][0].tolist() == [0, 0]
    assert (int(episode.data["start"]), int(expected.data["start"])) == (1, 0)


@pytest.mark.parametrize("env_id", ["othello", "connect_four"])
def test_pgx_boards_are_drawn_from_the_first_movers_side(env_id):
    import pgx

    env = pgx.make(env_id)
    states = [jit(env.init)(jax.random.PRNGKey(0))]
    for _ in range(3):
        action = int(np.argmax(states[-1].legal_action_mask))
        states.append(jit(env.step)(states[-1], action))

    board, to_move = BOARDS[env_id](stacked(states)._x)

    assert to_move.tolist() == [0, 1, 0, 1]
    # pieces of the colour that moves first are 1, whichever pgx player that is and
    # whoever is to move
    if env_id == "othello":
        assert board[0][3].tolist()[3:5] == [2, 1]
        # black's first move, at the first legal square (d3), turns d4 black
        assert (board[1][2][3], board[1][3][3]) == (1, 1)
    else:
        # the first two discs drop into the bottom of the leftmost column
        assert board[2][5][0] == 1 and board[2][4][0] == 2


def test_episode_with_unknown_env_is_still_stored(monitor_server):
    # the dashboard says it has no renderer for it
    monitor = Monitor(monitor_server.url)
    monitor.start()

    monitor.log(0, {"game": Episode("nope", x=[1, 2])})
    monitor.flush()

    [entry] = monitor_server.get(f"/api/runs/{monitor.run_id}/media")["rows"]
    assert entry["env"] == "nope"


def test_episode_names_its_opponent(monitor_server):
    monitor = Monitor(monitor_server.url)
    monitor.start()

    monitor.log(0, {"named": Episode("nope", opponent="edax level 8", x=[1])})
    monitor.log(1, {"unnamed": Episode("nope", x=[1])})
    monitor.flush()

    named, unnamed = monitor_server.get(f"/api/runs/{monitor.run_id}/media")["rows"]
    assert named["opponent"] == "edax level 8"
    assert "opponent" not in unnamed


def test_unreadable_episode_is_rejected(monitor_server, capsys):
    monitor = Monitor(monitor_server.url)
    monitor.start()

    monitor.log(0, {"game": Episode("othello", names=np.array(["a", "b"]))})
    monitor.log(0, {"game": Episode("bad name", x=[1])})
    monitor.flush()

    err = capsys.readouterr().err
    assert "names isn't numbers" in err
    assert "env names are letters" in err
    assert monitor_server.get(f"/api/runs/{monitor.run_id}/media")["rows"] == []


def test_continued_run_is_running_again(monitor_server):
    monitor = Monitor(monitor_server.url)
    monitor.start()
    monitor.finish()
    run_id = monitor.run_id

    # e.g. a second train_loop continuing from the first one's output
    monitor.start()
    monitor.log(1, {"loss": 1.0})
    monitor.flush()

    assert monitor.run_id == run_id
    assert monitor_server.get(f"/api/runs/{run_id}")["status"] == "running"


def test_activity_is_shown_until_the_run_finishes(monitor_server):
    monitor = Monitor(monitor_server.url)
    monitor.start()
    monitor.activity("epoch 0: self-play")
    monitor.flush()
    meta = monitor_server.get(f"/api/runs/{monitor.run_id}")
    assert meta["activity"] == "epoch 0: self-play"
    since = meta["activity_since"]

    # resending the same activity (a heartbeat) keeps when it started
    monitor.activity("epoch 0: self-play")
    monitor.flush()
    meta = monitor_server.get(f"/api/runs/{monitor.run_id}")
    assert meta["activity_since"] == since
    assert meta["updated"] >= since

    monitor.activity("epoch 0: training")
    monitor.flush()
    meta = monitor_server.get(f"/api/runs/{monitor.run_id}")
    assert meta["activity"] == "epoch 0: training"
    assert meta["activity_since"] >= since

    monitor.finish()
    meta = monitor_server.get(f"/api/runs/{monitor.run_id}")
    assert meta["status"] == "finished"
    assert meta["activity"] is None


def drain(q: queue.Queue) -> int:
    """Empties `q`; returns how many items it held."""
    count = 0
    while not q.empty():
        q.get_nowait()
        count += 1
    return count


def test_heartbeat_keeps_a_busy_run_fresh(monitor_server, monkeypatch):
    monkeypatch.setattr(client, "HEARTBEAT_S", 0.01)
    monitor = Monitor(monitor_server.url)
    # an item for each activity sent; set once the heartbeat thread exits
    sent, stopped = queue.Queue(), threading.Event()
    post_activity, heartbeat = monitor._post_activity, monitor._heartbeat

    def posted_activity(*args):
        post_activity(*args)
        sent.put(None)

    def stopping_heartbeat(*args):
        heartbeat(*args)
        stopped.set()

    monkeypatch.setattr(monitor, "_post_activity", posted_activity)
    monkeypatch.setattr(monitor, "_heartbeat", stopping_heartbeat)
    monitor.start()
    monitor.activity("testing: a slow opponent")
    monitor.flush()
    first = monitor_server.get(f"/api/runs/{monitor.run_id}")["updated"]

    # nothing is logged, but the heartbeat still reaches the server. The sender sends one
    # at a time, so of two sent from now on, the second was sent after `first`
    drain(sent)
    for _ in range(2):
        sent.get(timeout=5)
    meta = monitor_server.get(f"/api/runs/{monitor.run_id}")
    assert meta["updated"] > first
    assert meta["activity"] == "testing: a slow opponent"

    # and stops once the run finishes
    monitor.finish()
    drain(sent)
    assert stopped.wait(timeout=5), "the heartbeat didn't stop"
    monitor.flush()
    assert drain(sent) == 0
    assert monitor_server.get(f"/api/runs/{monitor.run_id}")["status"] == "finished"


def test_unreachable_server_never_raises(capsys):
    # nothing listens on port 9 (discard) on the test machines
    monitor = Monitor("http://127.0.0.1:9")

    monitor.start(config={"batch_size": 4})
    monitor.log(0, {"loss": 1.0})
    monitor.finish()

    assert monitor.run_id is None
    assert "can't reach" in capsys.readouterr().err


def test_unknown_run_is_not_found(monitor_server):
    with pytest.raises(urllib.error.HTTPError) as e:
        monitor_server.get("/api/runs/nope/metrics")
    assert e.value.code == 404
    with pytest.raises(urllib.error.HTTPError) as e:
        monitor_server.get("/media/..%2F..%2Fetc/passwd")
    assert e.value.code == 404


def test_dashboard_is_served(monitor_server):
    for path in ("/", "/run/anything", "/static/app.js", "/static/style.css"):
        with urllib.request.urlopen(monitor_server.url + path) as resp:
            assert resp.status == 200
