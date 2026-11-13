"""Config loading, seeds and run directories (PLAN.md §7.2, §7.3, §11)."""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import logging
import random
import shlex
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML config file into a dict."""
    with open(path) as fh:
        return yaml.safe_load(fh)


def resolve(path: str | Path) -> Path:
    """Resolve a path relative to the project root (absolute paths are returned unchanged)."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def set_seeds(seed: int) -> None:
    """Seed the global RNGs used in this project (numpy, random)."""
    random.seed(seed)
    np.random.seed(seed)


def git_commit() -> str:
    """Current git commit hash, with a '-dirty' suffix if the tree has uncommitted changes."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True, text=True
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "no-git-commit"


@dataclass
class Run:
    run_id: str
    phase: str
    dir: Path
    config: dict[str, Any]
    config_hash: str
    commit: str
    started: str
    log: logging.Logger

    def write_json(self, name: str, obj: Any) -> Path:
        out = self.dir / name
        with open(out, "w") as fh:
            json.dump(obj, fh, indent=2, default=str)
        return out

    def finish(self, status: str, key_metrics: dict[str, Any] | None = None) -> None:
        """Append one line to runs/registry.csv (PLAN.md §11)."""
        registry = resolve("runs/registry.csv")
        new = not registry.exists()
        with open(registry, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(
                    [
                        "run_id",
                        "phase",
                        "config_hash",
                        "git_commit",
                        "start",
                        "end",
                        "status",
                        "key_metrics",
                    ]
                )
            w.writerow(
                [
                    self.run_id,
                    self.phase,
                    self.config_hash,
                    self.commit,
                    self.started,
                    dt.datetime.now().isoformat(timespec="seconds"),
                    status,
                    json.dumps(key_metrics or {}, default=str),
                ]
            )
        self.log.info("run %s finished: %s", self.run_id, status)


def start_run(phase: str, config_path: str | Path, tag: str = "") -> Run:
    """Create runs/<phase>/<run_id>/ with config_used.yaml, git_commit.txt, command.txt, log.txt."""
    config = load_config(config_path)
    raw = Path(config_path).read_bytes()
    config_hash = hashlib.sha256(raw).hexdigest()[:12]
    now = dt.datetime.now()
    run_id = now.strftime("%Y%m%d-%H%M%S") + (f"-{tag}" if tag else "")
    run_dir = resolve(f"runs/{phase}/{run_id}")
    run_dir.mkdir(parents=True, exist_ok=False)
    with open(run_dir / "config_used.yaml", "w") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)
    commit = git_commit()
    (run_dir / "git_commit.txt").write_text(commit + "\n")
    (run_dir / "command.txt").write_text(
        "uv run python " + " ".join(shlex.quote(a) for a in sys.argv) + "\n"
    )

    log = logging.getLogger(f"hpcdiag.{phase}.{run_id}")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.FileHandler(run_dir / "log.txt"), logging.StreamHandler()):
        h.setFormatter(fmt)
        log.addHandler(h)

    set_seeds(int(config.get("seed", 0)))
    log.info(
        "run %s started; config %s (sha256[:12]=%s); commit %s",
        run_id,
        config_path,
        config_hash,
        commit,
    )
    return Run(
        run_id, phase, run_dir, config, config_hash, commit, now.isoformat(timespec="seconds"), log
    )


def run_guarded(run: Run, body: Callable[[Run], int]) -> int:
    """Call body(run); if it raises, register the run as failed in the registry, then re-raise."""
    try:
        return body(run)
    except BaseException as exc:
        run.log.exception("run %s crashed", run.run_id)
        run.finish(f"failed: {type(exc).__name__}: {exc}"[:300])
        raise
