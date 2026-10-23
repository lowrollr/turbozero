"""A small HTTP server that training runs push metrics and data to, and that serves
the dashboard for browsing them.

    uv run turbozero-monitor --port 8008 --dir runs

Everything lives on disk under the run directory, one folder per run:

    <dir>/<run id>/meta.json      project, name, config, timestamps, status, activity
    <dir>/<run id>/metrics.jsonl  one {"step": ..., "time": ..., <metrics>} per line
    <dir>/<run id>/media.jsonl    one {"key": ..., "step": ..., "file": ...} per line,
                                  or {..., "env": ..., "episode": ...} for an episode
    <dir>/<run id>/media/         data runs logged as dicts (.json)
    <dir>/<run id>/episodes/      the tests' episodes, as raw arrays (.npz)

Runs send episodes as raw arrays, and the dashboard gets them as they are (see
pack_arrays) and draws them itself (static/renderers.js), so it can play, pause and
step through them, and neither the training run nor the server draws anything.
"""

import argparse
import io
import json
import math
import mimetypes
import os
import re
import secrets
import threading
import time
import zipfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

STATIC_DIR = Path(__file__).parent / "static"

# runs get a memorable name unless the training script gives one
ADJECTIVES = [
    "amber",
    "brisk",
    "calm",
    "deft",
    "eager",
    "fallow",
    "gilded",
    "hardy",
    "idle",
    "jolly",
    "keen",
    "lucid",
    "mellow",
    "nimble",
    "olive",
    "patient",
    "quiet",
    "rustic",
    "sly",
    "tawny",
    "umber",
    "vivid",
    "wary",
    "young",
    "zesty",
]
NOUNS = [
    "badger",
    "crane",
    "dune",
    "ember",
    "finch",
    "grove",
    "heron",
    "ibis",
    "juniper",
    "kestrel",
    "lark",
    "maple",
    "newt",
    "otter",
    "pine",
    "quail",
    "reed",
    "sparrow",
    "thistle",
    "vole",
    "wren",
    "yarrow",
]


def _safe(name: str) -> str:
    """Make a metric/media key safe to use as a file name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "media"


class RunStore:
    """Runs on disk. One lock guards every write; reads go straight to the files."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def _dir(self, run_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
            raise KeyError(run_id)
        path = self.root / run_id
        if not path.is_dir():
            raise KeyError(run_id)
        return path

    def _read_meta(self, run_dir: Path) -> dict[str, Any]:
        return json.loads((run_dir / "meta.json").read_text())

    def _write_meta(self, run_dir: Path, meta: dict[str, Any]) -> None:
        tmp = run_dir / "meta.json.tmp"
        tmp.write_text(json.dumps(meta))
        tmp.replace(run_dir / "meta.json")

    def create(
        self, project: str, name: str | None, config: dict[str, Any]
    ) -> dict[str, Any]:
        now = time.time()
        run_id = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
        run_id += "-" + secrets.token_hex(2)
        if name is None:
            num = sum(1 for m in self.list() if m["project"] == project) + 1
            name = f"{secrets.choice(ADJECTIVES)}-{secrets.choice(NOUNS)}-{num}"
        run = config.get("run", {})
        meta = {
            "id": run_id,
            "project": project,
            "name": name,
            "config": config,
            "created": now,
            "updated": now,
            "status": "running",
            "step": None,
            "seed": run.get("seed"),
            "num_epochs": run.get("num_epochs"),
        }
        with self.lock:
            run_dir = self.root / run_id
            (run_dir / "media").mkdir(parents=True)
            (run_dir / "episodes").mkdir()
            self._write_meta(run_dir, meta)
        return meta

    def log(self, run_id: str, step: int, metrics: dict[str, Any]) -> None:
        run_dir = self._dir(run_id)
        now = time.time()
        # JSON has no NaN or inf (the dashboard couldn't parse them), so a diverged
        # metric is stored as null
        metrics = {
            k: v if isinstance(v, (int, float)) and math.isfinite(v) else None
            for k, v in metrics.items()
        }
        with self.lock:
            with open(run_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps({"step": step, "time": now, **metrics}) + "\n")
            meta = self._read_meta(run_dir)
            meta["updated"] = now
            meta["step"] = step if meta["step"] is None else max(meta["step"], step)
            # a run that logs is running, even one that finished and was continued
            meta["status"] = "running"
            self._write_meta(run_dir, meta)

    def _add_entry(self, run_dir: Path, entry: dict[str, Any]) -> None:
        """Append to the run's media, holding the lock."""
        with open(run_dir / "media.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
        meta = self._read_meta(run_dir)
        meta["updated"] = entry["time"]
        self._write_meta(run_dir, meta)

    def add_media(self, run_id: str, key: str, step: int, data: Any) -> dict[str, Any]:
        """Store data a run logged as a dict, which the dashboard shows as it is."""
        run_dir = self._dir(run_id)
        file = f"{_safe(key)}-{step}.json"
        entry = {"key": key, "step": step, "file": file, "time": time.time()}
        with self.lock:
            (run_dir / "media" / file).write_text(json.dumps(data))
            self._add_entry(run_dir, entry)
        return entry

    def add_episode(
        self, run_id: str, key: str, step: int, env: str, data: bytes
    ) -> dict[str, Any]:
        """Store a test's episode, raw, for the dashboard to draw: an entry in the
        run's media naming its `env`, which picks the dashboard's renderer."""
        run_dir = self._dir(run_id)
        # refuse what the dashboard couldn't read, rather than store it
        pack_arrays(_load_arrays(data))
        file = f"{_safe(key)}-{step}.npz"
        entry = {
            "key": key,
            "step": step,
            "env": env,
            "episode": file,
            "time": time.time(),
        }
        with self.lock:
            (run_dir / "episodes" / file).write_bytes(data)
            self._add_entry(run_dir, entry)
        return entry

    def episode(self, run_id: str, file: str) -> bytes:
        """A test's episode, packed for the dashboard."""
        path = self._dir(run_id) / "episodes" / _safe(file)
        if not path.is_file():
            raise KeyError(file)
        return pack_arrays(_load_arrays(path.read_bytes()))

    def set_activity(self, run_id: str, activity: str | None) -> None:
        """Record what a running run is doing now. Runs also send their current activity
        as a heartbeat, so this marks the run alive even when the activity is unchanged;
        `activity_since` only moves when the activity does."""
        run_dir = self._dir(run_id)
        now = time.time()
        with self.lock:
            meta = self._read_meta(run_dir)
            if meta.get("activity") != activity:
                meta["activity"] = activity
                meta["activity_since"] = now
            meta["updated"] = now
            self._write_meta(run_dir, meta)

    def finish(self, run_id: str, status: str) -> None:
        run_dir = self._dir(run_id)
        with self.lock:
            meta = self._read_meta(run_dir)
            meta["status"] = status
            meta["updated"] = time.time()
            meta["activity"] = None
            meta["activity_since"] = None
            self._write_meta(run_dir, meta)

    def list(self) -> list[dict[str, Any]]:
        runs = []
        for run_dir in self.root.iterdir():
            try:
                meta = self._read_meta(run_dir)
            except (OSError, ValueError):
                continue
            meta.pop("config", None)
            runs.append(meta)
        return sorted(runs, key=lambda m: m["created"], reverse=True)

    def get(self, run_id: str) -> dict[str, Any]:
        return self._read_meta(self._dir(run_id))

    def _read_jsonl(self, path: Path, since: int) -> dict[str, Any]:
        """Lines from `since` on, plus the offset to ask for next time."""
        if not path.exists():
            return {"rows": [], "next": since}
        with open(path) as f:
            lines = f.readlines()
        # a line still being written has no newline yet; leave it for the next poll
        if lines and not lines[-1].endswith("\n"):
            lines = lines[:-1]
        return {
            "rows": [json.loads(line) for line in lines[since:]],
            "next": len(lines),
        }

    def metrics(self, run_id: str, since: int = 0) -> dict[str, Any]:
        return self._read_jsonl(self._dir(run_id) / "metrics.jsonl", since)

    def media(self, run_id: str, since: int = 0) -> dict[str, Any]:
        return self._read_jsonl(self._dir(run_id) / "media.jsonl", since)

    def media_path(self, run_id: str, file: str) -> Path:
        path = self._dir(run_id) / "media" / _safe(file)
        if not path.is_file():
            raise KeyError(file)
        return path


def _load_arrays(data: bytes) -> dict[str, np.ndarray]:
    """The arrays of an .npz file's bytes."""
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as npz:
            return {name: npz[name] for name in npz.files}
    except (zipfile.BadZipFile, OSError, ValueError) as e:
        raise ValueError(f"an episode is an .npz file of arrays ({e})") from e


def _env(query: dict[str, list[str]]) -> str:
    """The env an episode was logged for, which names the dashboard's renderer for it."""
    env = query["env"][0]
    if not re.fullmatch(r"[A-Za-z0-9_]+", env):
        raise ValueError(f"env names are letters, digits and underscores, not {env!r}")
    return env


def pack_arrays(arrays: dict[str, np.ndarray]) -> bytes:
    """Arrays as the dashboard reads them (static/episodes.js loadArrays): the
    length of a JSON header as a little-endian uint32, the header, {name: {"dtype",
    "shape", "offset"}}, then each array's bytes from that offset past the header,
    4-byte aligned. Floats go as float32, other numbers as int32, bools as uint8."""
    header, chunks, offset = {}, [], 0
    for name, a in arrays.items():
        a = np.asarray(a)
        if a.dtype == np.bool_:
            a = a.astype(np.uint8)
        elif np.issubdtype(a.dtype, np.floating):
            a = a.astype("<f4")
        elif np.issubdtype(a.dtype, np.number):
            a = a.astype("<i4")
        else:
            raise TypeError(f"{name} isn't numbers but {a.dtype}")
        data = a.tobytes()
        header[name] = {"dtype": a.dtype.name, "shape": a.shape, "offset": offset}
        chunks.append(data + bytes(-len(data) % 4))
        offset += len(chunks[-1])
    head = json.dumps(header).encode()
    head += b" " * (-len(head) % 4)
    return len(head).to_bytes(4, "little") + head + b"".join(chunks)


class Handler(BaseHTTPRequestHandler):
    store: RunStore

    def log_message(self, format, *args):
        # one line per request is noise next to a training run
        pass

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: Any, status: int = HTTPStatus.OK) -> None:
        self._send(status, json.dumps(obj).encode(), "application/json")

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status)

    def _body(self) -> bytes:
        return self.rfile.read(int(self.headers.get("Content-Length", 0)))

    def _file(self, path: Path) -> None:
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._send(HTTPStatus.OK, path.read_bytes(), content_type)

    def do_GET(self):
        url = urlparse(self.path)
        query = parse_qs(url.query)
        parts = [p for p in url.path.split("/") if p]
        try:
            since = int(query.get("since", ["0"])[0])
            match parts:
                case ["api", "runs"]:
                    self._json(self.store.list())
                case ["api", "runs", run_id]:
                    self._json(self.store.get(run_id))
                case ["api", "runs", run_id, "metrics"]:
                    self._json(self.store.metrics(run_id, since))
                case ["api", "runs", run_id, "media"]:
                    self._json(self.store.media(run_id, since))
                case ["api", "runs", run_id, "episodes", file]:
                    body = self.store.episode(run_id, file)
                    self._send(HTTPStatus.OK, body, "application/octet-stream")
                case ["media", run_id, file]:
                    self._file(self.store.media_path(run_id, file))
                case [] | ["run", _]:
                    # the dashboard routes client-side, so every page is index.html
                    self._file(STATIC_DIR / "index.html")
                case ["static", file] if (STATIC_DIR / _safe(file)).is_file():
                    self._file(STATIC_DIR / _safe(file))
                case _:
                    self._error(HTTPStatus.NOT_FOUND, "not found")
        except KeyError:
            self._error(HTTPStatus.NOT_FOUND, "no such run")
        except ValueError as e:
            self._error(HTTPStatus.BAD_REQUEST, str(e))

    def do_POST(self):
        url = urlparse(self.path)
        query = parse_qs(url.query)
        parts = [p for p in url.path.split("/") if p]
        try:
            match parts:
                case ["api", "runs"]:
                    body = json.loads(self._body())
                    meta = self.store.create(
                        project=body.get("project") or "default",
                        name=body.get("name"),
                        config=body.get("config") or {},
                    )
                    self._json(meta, HTTPStatus.CREATED)
                case ["api", "runs", run_id, "activity"]:
                    body = json.loads(self._body())
                    activity = body.get("activity")
                    if activity is not None and not isinstance(activity, str):
                        raise ValueError("activity must be a string or null")
                    self.store.set_activity(run_id, activity)
                    self._json({"ok": True})
                case ["api", "runs", run_id, "log"]:
                    body = json.loads(self._body())
                    self.store.log(run_id, int(body["step"]), body["metrics"])
                    self._json({"ok": True})
                case ["api", "runs", run_id, "media"]:
                    # runs send data, never pictures: the dashboard draws them
                    content_type = self.headers.get("Content-Type", "")
                    if content_type != "application/json":
                        self._error(
                            HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                            f"runs log data as JSON, not {content_type!r}",
                        )
                        return
                    entry = self.store.add_media(
                        run_id,
                        key=query["key"][0],
                        step=int(query["step"][0]),
                        data=json.loads(self._body()),
                    )
                    self._json(entry, HTTPStatus.CREATED)
                case ["api", "runs", run_id, "episodes"]:
                    entry = self.store.add_episode(
                        run_id,
                        key=query["key"][0],
                        step=int(query["step"][0]),
                        env=_env(query),
                        data=self._body(),
                    )
                    self._json(entry, HTTPStatus.CREATED)
                case ["api", "runs", run_id, "finish"]:
                    body = json.loads(self._body() or b"{}")
                    self.store.finish(run_id, body.get("status", "finished"))
                    self._json({"ok": True})
                case _:
                    self._error(HTTPStatus.NOT_FOUND, "not found")
        except KeyError:
            self._error(HTTPStatus.NOT_FOUND, "no such run")
        except (ValueError, TypeError) as e:
            self._error(HTTPStatus.BAD_REQUEST, str(e))


def make_server(host: str, port: int, run_dir: str) -> ThreadingHTTPServer:
    """The monitor's HTTP server, storing runs in `run_dir` (port 0 picks a free one)."""
    handler = type("Handler", (Handler,), {"store": RunStore(Path(run_dir))})
    return ThreadingHTTPServer((host, port), handler)


def main():
    parser = argparse.ArgumentParser(description="Serve the turbozero run monitor.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--dir", default="runs", help="where runs are stored")
    args = parser.parse_args()
    server = make_server(args.host, args.port, args.dir)
    print(
        f"turbozero monitor on http://{args.host}:{args.port}  "
        f"(runs in {os.path.abspath(args.dir)})"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
