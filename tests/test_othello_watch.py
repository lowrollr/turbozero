"""The Othello example's watch.py: which checkpoints, at which levels, it has vs_engine.py evaluate."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# the example's scripts import their sibling modules
sys.path.insert(0, str(Path(__file__).parents[1] / "examples" / "othello"))
import watch

# stands in for vs_engine.py: appends a record per level played, as vs_engine.py does, and logs
# the levels it played; with STUB_FAIL_AFTER=n it fails after playing n levels
STUB_VS_ENGINE = """
import argparse, json, os, sys
p = argparse.ArgumentParser()
p.add_argument("checkpoint")
for flag in ("--engine", "--blocks", "--channels", "--stop-below", "--results"):
    p.add_argument(flag)
p.add_argument("--sims", type=int)
p.add_argument("--openings", type=int)
p.add_argument("--levels", type=int, nargs="+")
p.add_argument("--skip-levels", type=int, nargs="+", default=[])
a = p.parse_args()
played = [level for level in a.levels if level not in a.skip_levels]
with open(os.environ["STUB_LOG"], "a") as f:
    f.write(json.dumps([os.path.basename(a.checkpoint), played]) + "\\n")
for i, level in enumerate(played):
    if i == int(os.environ.get("STUB_FAIL_AFTER", "99")):
        sys.exit("stub: failed partway")
    with open(a.results, "a") as f:
        record = dict(checkpoint=a.checkpoint, engine=a.engine, level=level, sims=a.sims,
                      games=2 * a.openings, score=0.5, se=0.0)
        f.write(json.dumps(record) + "\\n")
"""


@pytest.fixture
def watcher(tmp_path, monkeypatch):
    """`watcher.run(*args)` runs watch.py on a checkpoint directory, `watcher.ckpt_dir` (without
    --pid, so it evaluates the checkpoints there and exits) with a stub vs_engine.py, and returns
    the stub's calls: (checkpoint, levels played) pairs."""
    stub = tmp_path / "stub_vs_engine.py"
    stub.write_text(STUB_VS_ENGINE)
    monkeypatch.setattr(watch, "VS_ENGINE", stub)
    log = tmp_path / "stub.log"
    monkeypatch.setenv("STUB_LOG", str(log))
    ckpt_dir = tmp_path / "run"
    ckpt_dir.mkdir()

    def run(*args, fail_after=None):
        if fail_after is None:
            monkeypatch.delenv("STUB_FAIL_AFTER", raising=False)
        else:
            monkeypatch.setenv("STUB_FAIL_AFTER", str(fail_after))
        log.unlink(missing_ok=True)
        monkeypatch.setattr(sys, "argv", ["watch.py", str(ckpt_dir), *args])
        watch.main()
        if not log.exists():
            return []
        return [tuple(json.loads(line)) for line in log.read_text().splitlines()]

    return SimpleNamespace(run=run, ckpt_dir=ckpt_dir)


def save(ckpt_dir: Path, *epochs: int):
    for epoch in epochs:
        (ckpt_dir / f"{epoch}.eqx").write_bytes(b"")


def records(ckpt_dir: Path) -> list[tuple[str, int, int]]:
    with open(ckpt_dir / "vs_engine.jsonl") as f:
        lines = [json.loads(line) for line in f]
    return sorted((Path(r["checkpoint"]).name, r["level"], r["games"]) for r in lines)


def test_evaluates_every_kept_checkpoint_and_the_last(watcher):
    save(watcher.ckpt_dir, 0, 25, 50, 60)

    calls = watcher.run("--levels", "4", "6", "--openings", "2")

    # the newest first; 0.eqx is left out
    assert calls == [("60.eqx", [4, 6]), ("50.eqx", [4, 6]), ("25.eqx", [4, 6])]
    # done: nothing more to evaluate
    assert watcher.run("--levels", "4", "6", "--openings", "2") == []


def test_evaluates_levels_a_checkpoint_has_no_result_for(watcher):
    save(watcher.ckpt_dir, 25)
    watcher.run("--levels", "4", "--openings", "2")

    # level 4 is done: only the new levels are played
    calls = watcher.run("--levels", "4", "6", "8", "--openings", "2")

    assert calls == [("25.eqx", [6, 8])]
    assert records(watcher.ckpt_dir) == [
        ("25.eqx", 4, 4),
        ("25.eqx", 6, 4),
        ("25.eqx", 8, 4),
    ]


@pytest.mark.parametrize(
    "args",
    [
        ["--openings", "3"],
        ["--sims", "128"],
        ["--engine", "egaroucid"],
    ],
    ids=["openings", "sims", "engine"],
)
def test_results_at_other_settings_do_not_count(watcher, args):
    save(watcher.ckpt_dir, 25)
    watcher.run("--levels", "4", "--openings", "2")

    assert watcher.run("--levels", "4", "--openings", "2", *args) == [("25.eqx", [4])]


def test_finishes_an_evaluation_that_failed_partway(watcher):
    save(watcher.ckpt_dir, 25)
    with pytest.raises(subprocess.CalledProcessError):
        watcher.run("--levels", "4", "6", "8", "--openings", "2", fail_after=1)
    assert records(watcher.ckpt_dir) == [("25.eqx", 4, 4)]

    calls = watcher.run("--levels", "4", "6", "8", "--openings", "2")

    assert calls == [("25.eqx", [6, 8])]
    assert records(watcher.ckpt_dir) == [
        ("25.eqx", 4, 4),
        ("25.eqx", 6, 4),
        ("25.eqx", 8, 4),
    ]
