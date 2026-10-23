"""The run monitor: the client logging to a live server, and the server storing and serving runs."""

import json
import time
import urllib.error
import urllib.request
from functools import partial

import jax
import numpy as np
import pytest

from core.common import two_player_game
from monitor import Episode, Monitor, client
from monitor.episodes import BOARDS, pgx_two_player_episode


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

    def slow_request(*args, **kwargs):
        time.sleep(0.5)
        return send(*args, **kwargs)

    monkeypatch.setattr(monitor, "_request", slow_request)

    start = time.time()
    monitor.log(0, {"loss": 1.0})
    assert time.time() - start < 0.25
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


def test_pgx_episode_is_served_for_the_dashboard(monitor_server, ttt, scripted):
    game = jax.jit(
        partial(
            two_player_game,
            evaluator_1=scripted.first_legal,
            evaluator_2=scripted.first_legal,
            params_1=None,
            params_2=None,
            env_step_fn=ttt.step_fn,
            env_init_fn=ttt.init_fn,
            max_steps=12,
            p1_first=False,
        )
    )
    _, frames, p_ids = jax.device_get(game(jax.random.PRNGKey(0)))
    monitor = Monitor(monitor_server.url)
    monitor.start()

    monitor.log(2, {"game": pgx_two_player_episode()(frames, p_ids)})
    monitor.flush()

    [entry] = monitor_server.get(f"/api/runs/{monitor.run_id}/media")["rows"]
    assert (entry["key"], entry["step"], entry["env"]) == ("game", 2, "tic_tac_toe")
    with urllib.request.urlopen(
        f"{monitor_server.url}/api/runs/{monitor.run_id}/episodes/{entry['episode']}"
    ) as resp:
        arrays = load_arrays(resp.read())
    # playing the first free square, X wins on the diagonal with the 7th move: the
    # initial board and 7 moves, without the frames padding the game out to 12 steps
    board = arrays["board"]
    assert board.shape == (1 + 7, 3, 3)
    assert (board[0] == 0).all()
    assert board[-1].tolist() == [[1, 2, 1], [2, 1, 2], [1, 0, 0]]
    assert arrays["to_move"].tolist() == [0, 1, 0, 1, 0, 1, 0, 1]
    assert arrays["values"].shape == (1 + 7, 2)
    # the trained agent moved second, so played O, and lost
    assert (int(arrays["agent"]), int(arrays["completed"])) == (1, 1)
    assert float(arrays["outcome"]) == -1


@pytest.mark.parametrize("env_id", ["othello", "connect_four"])
def test_pgx_boards_are_drawn_from_the_first_movers_side(env_id):
    import pgx

    env = pgx.make(env_id)
    states = [env.init(jax.random.PRNGKey(0))]
    for _ in range(3):
        action = int(np.argmax(states[-1].legal_action_mask))
        states.append(env.step(states[-1], action))
    stacked = jax.tree.map(lambda *xs: np.stack(xs), *states)

    board, to_move = BOARDS[env_id](stacked._x)

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


def test_heartbeat_keeps_a_busy_run_fresh(monitor_server, monkeypatch):
    monkeypatch.setattr(client, "HEARTBEAT_S", 0.05)
    monitor = Monitor(monitor_server.url)
    monitor.start()
    monitor.activity("testing: a slow opponent")
    monitor.flush()
    first = monitor_server.get(f"/api/runs/{monitor.run_id}")["updated"]

    # nothing is logged, but the heartbeat still reaches the server
    deadline = time.time() + 5
    while monitor_server.get(f"/api/runs/{monitor.run_id}")["updated"] == first:
        assert time.time() < deadline, "no heartbeat arrived"
        time.sleep(0.05)
    meta = monitor_server.get(f"/api/runs/{monitor.run_id}")
    assert meta["activity"] == "testing: a slow opponent"

    # and stops once the run finishes
    monitor.finish()
    finished = monitor_server.get(f"/api/runs/{monitor.run_id}")["updated"]
    time.sleep(0.3)
    assert monitor_server.get(f"/api/runs/{monitor.run_id}")["updated"] == finished


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
