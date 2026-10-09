"""SHA-256 / MD5 records for raw data files (PLAN.md §4 rules, CLAUDE.md rule 5).

CHECKSUMS.txt format (one line per file, whitespace separated, '#' comments):
    <sha256>  <md5>  <size_bytes>  <path relative to data/raw>
"""

from __future__ import annotations

import hashlib
from pathlib import Path

CHUNK = 1 << 20


def file_digests(path: Path) -> tuple[str, str, int]:
    """Return (sha256 hex, md5 hex, size in bytes) of a file, read in 1 MiB chunks."""
    sha, md5, size = hashlib.sha256(), hashlib.md5(), 0
    with open(path, "rb") as fh:
        while chunk := fh.read(CHUNK):
            sha.update(chunk)
            md5.update(chunk)
            size += len(chunk)
    return sha.hexdigest(), md5.hexdigest(), size


def read_checksums(path: Path) -> dict[str, tuple[str, str, int]]:
    """Parse CHECKSUMS.txt into {relative path: (sha256, md5, size)}."""
    out: dict[str, tuple[str, str, int]] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        sha, md5, size, rel = line.split(maxsplit=3)
        out[rel] = (sha, md5, int(size))
    return out


def verify(raw_dir: Path, checksums_file: Path) -> list[str]:
    """Recompute digests of every file listed in CHECKSUMS.txt. Returns a list of problems."""
    problems = []
    for rel, (sha, md5, size) in read_checksums(checksums_file).items():
        p = raw_dir / rel
        if not p.exists():
            problems.append(f"missing: {rel}")
            continue
        got = file_digests(p)
        if got != (sha, md5, size):
            problems.append(f"mismatch: {rel} expected sha256={sha} got {got[0]}")
    return problems
