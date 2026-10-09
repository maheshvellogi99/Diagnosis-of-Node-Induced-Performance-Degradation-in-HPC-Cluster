"""Tests for hpcdiag.utils (run directories, registry, checksums)."""

from __future__ import annotations

import csv
import hashlib

import pytest
import yaml

from hpcdiag.utils import run as run_mod
from hpcdiag.utils.checksums import file_digests, read_checksums, verify


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(run_mod, "PROJECT_ROOT", tmp_path)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"seed": 1}))
    return tmp_path, cfg


def _registry(root):
    with open(root / "runs" / "registry.csv") as fh:
        return list(csv.DictReader(fh))


def test_start_run_writes_required_files(project):
    root, cfg = project
    r = run_mod.start_run("phaseX", cfg, tag="t")
    for name in ("config_used.yaml", "git_commit.txt", "command.txt", "log.txt"):
        assert (r.dir / name).exists()
    r.finish("ok", {"a": 1})
    rows = _registry(root)
    assert [x["status"] for x in rows] == ["ok"]


def test_run_guarded_registers_crash_as_failed(project):
    root, cfg = project
    r = run_mod.start_run("phaseX", cfg, tag="crash")

    def body(_run):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        run_mod.run_guarded(r, body)
    rows = _registry(root)
    assert len(rows) == 1
    assert rows[0]["run_id"] == r.run_id
    assert rows[0]["status"].startswith("failed: RuntimeError: boom")


def test_checksums_roundtrip(tmp_path):
    f = tmp_path / "a" / "x.bin"
    f.parent.mkdir()
    f.write_bytes(b"hello")
    sha, md5, size = file_digests(f)
    assert sha == hashlib.sha256(b"hello").hexdigest()
    assert md5 == hashlib.md5(b"hello").hexdigest() and size == 5
    sums = tmp_path / "CHECKSUMS.txt"
    sums.write_text(f"# comment\n{sha}  {md5}  {size}  a/x.bin\n")
    assert read_checksums(sums) == {"a/x.bin": (sha, md5, 5)}
    assert verify(tmp_path, sums) == []
    f.write_bytes(b"hellO")
    assert verify(tmp_path, sums)[0].startswith("mismatch: a/x.bin")
