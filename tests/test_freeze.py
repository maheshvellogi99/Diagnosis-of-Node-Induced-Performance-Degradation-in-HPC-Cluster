"""D112 freeze guard (src/hpcdiag/freeze.py, scripts/check_frozen.py)."""

from __future__ import annotations

import hashlib
import subprocess

import pytest
import yaml

from hpcdiag import freeze as FZ
from hpcdiag.layer3 import simulate as SIM
from hpcdiag.utils.run import PROJECT_ROOT, load_config

CFG = "configs/phase5.yaml"
SAMPLE = """| File | Role | SHA-256 |
|---|---|---|
| `configs/a.yaml` | Layer 1 | `aaaa` |
| `data/x.parquet` | index | `bbbb` |
"""


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def test_parse_hashes():
    assert FZ.parse_hashes(SAMPLE) == {"configs/a.yaml": "aaaa", "data/x.parquet": "bbbb"}


@pytest.mark.parametrize(
    "key, ok",
    [
        ("probes.k", True),
        ("probes.k.cpu.mbw", True),
        ("probes.sigma", True),
        ("probes.sigma.cpu", True),
        ("probes.cost_s.mem", True),
        ("probes.mem_fixed_shift", True),
        ("probes.healthy_pool", True),
        ("telemetry.counts", True),
        ("telemetry.crossblock_counts", False),
        ("probes.names", False),
        ("evalue.edges", False),
        ("evalue.beta", False),
        ("evalue.bet_s0", False),
        ("evalue.probe_J", False),
        ("policy.s_grid", False),
        ("policy.pi0_h0", False),
        ("policy.tau", False),
        ("policy.exclude_days", False),
        ("probes.k_extra", False),
        ("grid5f", False),
    ],
)
def test_d113_fields(key, ok):
    assert FZ.allowed_change(key) is ok


def test_changed_keys():
    old = {"probes": {"k": {"cpu": {"cpu": 1.0}}, "sigma": 0.05}, "policy": {"tau": 0.95}}
    new = {"probes": {"k": {"cpu": {"cpu": 0.8}}, "sigma": {"cpu": 0.04}, "healthy_pool": "p.csv"},
           "policy": {"tau": 0.95}}  # fmt: skip
    assert FZ.changed_keys(old, new) == {"probes.k.cpu.cpu", "probes.sigma", "probes.healthy_pool"}


def _files(tmp_path, text: str):
    p = tmp_path / "configs" / "phase5.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def test_check_files_allows_only_d113_fields(tmp_path):
    frozen = yaml.safe_dump({"probes": {"sigma": 0.05, "k": {"cpu": {"cpu": 1.0}}},
                             "evalue": {"beta": 0.9}})  # fmt: skip
    hashes = {"configs/phase5.yaml": sha(frozen.encode())}

    def tagged(path):
        return frozen.encode()

    _files(tmp_path, frozen)
    assert FZ.check_files(hashes, tmp_path, tagged) == ([], [])
    _files(tmp_path, frozen.replace("sigma: 0.05", "sigma: 0.04"))
    problems, notes = FZ.check_files(hashes, tmp_path, tagged)
    assert problems == [] and "probes.sigma" in notes[0]
    _files(tmp_path, frozen.replace("beta: 0.9", "beta: 0.5"))
    problems, _ = FZ.check_files(hashes, tmp_path, tagged)
    assert problems and "evalue.beta" in problems[0]


def test_check_files_d113_only_in_phase5_config(tmp_path):
    frozen = yaml.safe_dump({"probes": {"sigma": 0.05}})
    p = tmp_path / "configs" / "phase4_layer2.yaml"
    p.parent.mkdir(parents=True)
    p.write_text(frozen.replace("0.05", "0.04"))
    hashes = {"configs/phase4_layer2.yaml": sha(frozen.encode())}
    problems, _ = FZ.check_files(hashes, tmp_path, lambda path: frozen.encode())
    assert problems


def test_check_files_binary_must_match(tmp_path):
    p = tmp_path / "data" / "x.parquet"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"abc")
    assert FZ.check_files({"data/x.parquet": sha(b"abc")}, tmp_path, None) == ([], [])
    problems, _ = FZ.check_files({"data/x.parquet": sha(b"abd")}, tmp_path, None)
    assert problems
    problems, _ = FZ.check_files({"data/missing.parquet": "00"}, tmp_path, None)
    assert problems


def test_check_files_tag_and_doc_must_agree(tmp_path):
    frozen = yaml.safe_dump({"probes": {"sigma": 0.05}})
    _files(tmp_path, frozen.replace("0.05", "0.04"))
    problems, _ = FZ.check_files(
        {"configs/phase5.yaml": "deadbeef"}, tmp_path, lambda p: frozen.encode()
    )
    assert problems and "FROZEN_CONFIG" in problems[0]


def test_policy_check_passes_for_the_primary():
    assert FZ.check_policy("ig_d109", load_config(CFG)) == []


@pytest.mark.parametrize("name", ["ig", "stage", "d86", "design_rule"])
def test_policy_check_rejects_other_policies(name):
    assert FZ.check_policy(name, load_config(CFG))


def test_policy_check_rejects_unknown_policy():
    assert FZ.check_policy("nope", load_config(CFG))


def test_policy_check_rejects_config_J():
    cfg = load_config(CFG)
    cfg["evalue"]["probe_J"] = 1.0
    assert FZ.check_policy("ig_d109", cfg)


def test_default_probe_J_trap_is_caught():
    """Policy() defaults to probe_J = 1.0 (old formula); a hand-built 'ig_d109' must fail."""
    trap = SIM.Policy("ig_d109_B7200", 7200.0, exclusion="cleared_skipdone")
    assert trap.probe_J == 1.0
    assert FZ.check_policy_object(trap)
    good = SIM.Policy("ig_d109_B7200", 7200.0, exclusion="cleared_skipdone", probe_J=0.01)
    assert FZ.check_policy_object(good) == []
    bad = SIM.Policy("ig_d109_B7200", 7200.0, probes="stage", probe_J=0.01)
    assert FZ.check_policy_object(bad)


def _git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)


def test_tag_checks_in_temp_repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(
        tmp_path,
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@t",
        "commit",
        "-q",
        "--allow-empty",
        "-m",
        "a",
    )
    assert not FZ.tag_exists("v1.0-dev-frozen", tmp_path)
    _git(tmp_path, "tag", "v1.0-dev-frozen")
    assert FZ.tag_exists("v1.0-dev-frozen", tmp_path)
    assert FZ.tag_is_ancestor("v1.0-dev-frozen", tmp_path)


def test_guard_passes_on_this_repo_now():
    if not (PROJECT_ROOT / "data/processed/scenarios/index.parquet").exists():
        pytest.skip("development data not present")
    ok, msgs = FZ.run_guard("ig_d109", CFG)
    assert ok, msgs


def test_guard_fails_for_wrong_policy_on_this_repo():
    if not (PROJECT_ROOT / "data/processed/scenarios/index.parquet").exists():
        pytest.skip("development data not present")
    ok, msgs = FZ.run_guard("ig", CFG)
    assert not ok


def test_cli_exit_codes():
    if not (PROJECT_ROOT / "data/processed/scenarios/index.parquet").exists():
        pytest.skip("development data not present")
    run = lambda pol: subprocess.run(  # noqa: E731
        ["uv", "run", "python", "scripts/check_frozen.py", "--config", CFG, "--policy", pol],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
    )  # fmt: skip
    assert run("ig_d109").returncode == 0
    assert run("ig").returncode != 0
