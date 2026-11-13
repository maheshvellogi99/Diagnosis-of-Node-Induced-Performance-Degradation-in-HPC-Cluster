"""D112 freeze guard: is the code about to run the frozen primary (v1.0-dev-frozen)?

Checks, all of which must pass:
  (a) the tag exists and is an ancestor of HEAD;
  (b) every file listed in docs/FROZEN_CONFIG.md (read from the TAG, not the working tree, and the
      working-tree copy must equal it) has the listed SHA-256, or, for configs/phase5.yaml only,
      every difference from the tagged file is a D113 field;
  (d) src/ differs from the tag only in src/hpcdiag/layer3/, src/hpcdiag/freeze.py and
      src/hpcdiag/phase6/ (D116; working tree, committed or not, and untracked files);
  (c) the policy to be run is ig_d109 with probe_J = 0.01 exactly, built as the runner builds it
      (Policy's default probe_J = 1.0 is the old formula; a hand-built Policy is caught too).
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from collections.abc import Callable
from pathlib import Path

import yaml

from hpcdiag.layer3 import run_phase5 as RP
from hpcdiag.layer3 import simulate as SIM
from hpcdiag.utils.run import PROJECT_ROOT, load_config

TAG = "v1.0-dev-frozen"
FROZEN_DOC = "docs/FROZEN_CONFIG.md"
D113_CONFIG = "configs/phase5.yaml"
# D113: the only inputs Phase 6 may replace by measurements
D113_FIELDS = (
    "probes.k",
    "probes.sigma",
    "probes.cost_s",
    "probes.mem_fixed_shift",
    "probes.healthy_pool",
    "telemetry.crossblock_counts",  # D117: the matrix the primary reads (was telemetry.counts)
)
# D116: src/ may differ from the tag only here (D113 plumbing, the guard, the Phase 6 analysis)
ALLOWED_SRC = ("src/hpcdiag/layer3/", "src/hpcdiag/freeze.py", "src/hpcdiag/phase6/")
FROZEN_POLICY = "ig_d109"
FROZEN_J = 0.01
FROZEN_FIELDS = {"nodes": "rank", "probes": "ig_cost", "exclusion": "cleared_skipdone"}

_ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|.*\|\s*`([0-9a-f]+)`\s*\|\s*$")


def parse_hashes(md: str) -> dict[str, str]:
    """{path: sha256} from the hash table of FROZEN_CONFIG.md."""
    return {m.group(1): m.group(2) for line in md.splitlines() if (m := _ROW.match(line.strip()))}


def allowed_change(key: str) -> bool:
    return any(key == f or key.startswith(f + ".") for f in D113_FIELDS)


def changed_keys(old, new, prefix: str = "") -> set[str]:
    """Dotted keys whose values differ (added, removed or changed), at the deepest dict level."""
    if isinstance(old, dict) and isinstance(new, dict):
        out: set[str] = set()
        for k in set(old) | set(new):
            p = f"{prefix}.{k}" if prefix else str(k)
            if k not in old or k not in new:
                out.add(p)
            else:
                out |= changed_keys(old[k], new[k], p)
        return out
    return set() if old == new else {prefix}


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def check_files(
    hashes: dict[str, str], root: Path, read_tagged: Callable[[str], bytes] | None
) -> tuple[list[str], list[str]]:
    """(problems, notes). read_tagged(path) returns the file's bytes at the tag (configs only)."""
    problems, notes = [], []
    for path, want in hashes.items():
        f = Path(root) / path
        if not f.exists():
            problems.append(f"{path}: missing")
            continue
        cur = f.read_bytes()
        if _sha(cur) == want:
            continue
        if path != D113_CONFIG or read_tagged is None:
            problems.append(
                f"{path}: SHA-256 differs from FROZEN_CONFIG and no D113 change is allowed here"
            )
            continue
        tagged = read_tagged(path)
        if _sha(tagged) != want:
            problems.append(
                f"{path}: the tagged file does not match FROZEN_CONFIG (tag or document altered)"
            )
            continue
        keys = changed_keys(yaml.safe_load(tagged), yaml.safe_load(cur))
        bad = sorted(k for k in keys if not allowed_change(k))
        if bad:
            problems.append(f"{path}: frozen fields changed: {', '.join(bad)}")
        else:
            notes.append(f"{path}: D113 fields changed: {', '.join(sorted(keys))}")
    return problems, notes


def check_policy_object(pol: SIM.Policy) -> list[str]:
    out = []
    if pol.probe_J != FROZEN_J:
        out.append(f"policy {pol.name}: probe_J = {pol.probe_J}, frozen is {FROZEN_J} (D109)")
    for f, v in FROZEN_FIELDS.items():
        if getattr(pol, f) != v:
            out.append(f"policy {pol.name}: {f} = {getattr(pol, f)!r}, frozen is {v!r}")
    return out


def check_policy(name: str, cfg: dict) -> list[str]:
    if name != FROZEN_POLICY:
        return [f"policy {name!r} is not the frozen primary {FROZEN_POLICY!r}"]
    if name not in RP.POLICIES:
        return [f"policy {name!r} is not defined"]
    return check_policy_object(RP.make_policy(name, 0.0, cfg))


def _git(repo, *args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True)


def disallowed_src(paths) -> list[str]:
    """Paths under src/ that are outside the D116 allowed list."""
    return sorted(
        p
        for p in paths
        if not any(p == a or (a.endswith("/") and p.startswith(a)) for a in ALLOWED_SRC)
    )


def src_changes(tag: str, repo=PROJECT_ROOT) -> list[str]:
    """Files under src/ that differ between the tag and the working tree (committed or not),
    plus untracked files under src/ (ignored files such as __pycache__ excluded)."""
    changed = _git(repo, "diff", "--name-only", tag, "--", "src/").stdout.decode().split()
    new = (
        _git(repo, "ls-files", "--others", "--exclude-standard", "--", "src/")
        .stdout.decode()
        .split()
    )
    return sorted(set(changed) | set(new))


def tag_exists(tag: str, repo=PROJECT_ROOT) -> bool:
    return _git(repo, "rev-parse", "-q", "--verify", f"refs/tags/{tag}").returncode == 0


def tag_is_ancestor(tag: str, repo=PROJECT_ROOT) -> bool:
    return _git(repo, "merge-base", "--is-ancestor", tag, "HEAD").returncode == 0


def at_tag(path: str, tag: str = TAG, repo=PROJECT_ROOT) -> bytes:
    r = _git(repo, "show", f"{tag}:{path}")
    if r.returncode:
        raise FileNotFoundError(f"{path} not in {tag}")
    return r.stdout


def run_guard(policy: str, config_path: str, repo=PROJECT_ROOT) -> tuple[bool, list[str]]:
    msgs: list[str] = []
    if not tag_exists(TAG, repo):
        return False, [f"FAIL (a): tag {TAG} does not exist"]
    if not tag_is_ancestor(TAG, repo):
        msgs.append(f"FAIL (a): tag {TAG} is not an ancestor of HEAD")
    doc = at_tag(FROZEN_DOC, TAG, repo)
    if (Path(repo) / FROZEN_DOC).read_bytes() != doc:
        msgs.append(f"FAIL (b): {FROZEN_DOC} in the working tree differs from the tagged copy")
    hashes = parse_hashes(doc.decode())
    if not hashes:
        msgs.append(f"FAIL (b): no hash table found in the tagged {FROZEN_DOC}")
    problems, notes = check_files(hashes, Path(repo), lambda p: at_tag(p, TAG, repo))
    msgs += [f"FAIL (b): {p}" for p in problems] + [f"note (b): {n}" for n in notes]
    msgs += [f"FAIL (c): {p}" for p in check_policy(policy, load_config(config_path))]
    bad_src = disallowed_src(src_changes(TAG, repo))
    msgs += [f"FAIL (d): {p} changed since {TAG}, outside {ALLOWED_SRC} (D116)" for p in bad_src]
    ok = not any(m.startswith("FAIL") for m in msgs)
    msgs.append(
        f"{'PASS' if ok else 'FAIL'}: {len(hashes)} frozen files checked, policy {policy!r}"
    )
    return ok, msgs
