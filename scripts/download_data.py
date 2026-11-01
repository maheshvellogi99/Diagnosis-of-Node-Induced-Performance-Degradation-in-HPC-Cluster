"""Record (and verify) checksums of the raw datasets; fetch the configured downloads.

Record mode hashes every file already present under data/raw/ (SHA-256 + MD5 + size) and writes
data/raw/CHECKSUMS.txt. Where the publisher gives an MD5 (configs: raw_files.<path>.published_md5
or downloads[].md5) it must match, or the script fails.

--download fetches configs: downloads[] over HTTPS (D93), one file at a time, into a .part file
that is renamed only after its MD5 matches the published value. A mismatch stops the script and
leaves the .part file for inspection. Existing files with the right MD5 are not fetched again.
This is the only project code allowed to use the network (CLAUDE.md rule 5).

Usage:
    uv run python scripts/download_data.py --config configs/phase0_data.yaml             # record
    uv run python scripts/download_data.py --config configs/phase0_data.yaml --verify    # check
    uv run python scripts/download_data.py --config configs/phase0_data.yaml --download  # fetch
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import urllib.request
from pathlib import Path

from hpcdiag.utils.checksums import file_digests, verify
from hpcdiag.utils.run import load_config, resolve

IGNORED = {"CHECKSUMS.txt", ".DS_Store"}


def record(cfg: dict) -> int:
    raw_dir = resolve(cfg["raw_dir"])
    out = resolve(cfg["checksums_file"])
    published = {k: (v or {}).get("published_md5") for k, v in cfg.get("raw_files", {}).items()}
    published.update({d["path"]: d["md5"] for d in cfg.get("downloads", [])})

    files = sorted(p for p in raw_dir.rglob("*") if p.is_file() and p.name not in IGNORED)
    missing = [k for k in published if not (raw_dir / k).exists()]
    if missing:
        print(f"ERROR: configured raw files not found: {missing}", file=sys.stderr)
        return 1

    lines = [
        "# sha256  md5  size_bytes  path (relative to data/raw)",
        f"# written {dt.datetime.now().isoformat(timespec='seconds')} by scripts/download_data.py",
    ]
    status = 0
    for p in files:
        rel = p.relative_to(raw_dir).as_posix()
        sha, md5, size = file_digests(p)
        exp = published.get(rel)
        note = "no published md5"
        if exp:
            note = "published md5 OK" if exp == md5 else f"PUBLISHED MD5 MISMATCH (expected {exp})"
            if exp != md5:
                status = 1
        print(f"{rel}: sha256={sha} md5={md5} size={size} [{note}]")
        lines.append(f"{sha}  {md5}  {size}  {rel}")
    if status:
        print("ERROR: published MD5 mismatch; CHECKSUMS.txt not written.", file=sys.stderr)
        return status
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    return 0


def fetch(url: str, dest: Path, md5: str) -> int:
    """Download url to dest through dest.part; return 0 if the MD5 matches, else 1."""
    if dest.exists():
        have = file_digests(dest)[1]
        if have == md5:
            print(f"{dest.name}: present, md5 OK")
            return 0
        print(f"ERROR: {dest} exists with md5 {have}, expected {md5}", file=sys.stderr)
        return 1
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(part, "wb") as fh:  # noqa: S310
        done = 0
        while chunk := r.read(1 << 20):
            fh.write(chunk)
            done += len(chunk)
            if done % (100 << 20) < (1 << 20):
                print(f"{dest.name}: {done >> 20} MiB", flush=True)
    have = file_digests(part)[1]
    if have != md5:
        print(f"ERROR: md5 mismatch for {dest.name}: got {have}, expected {md5}", file=sys.stderr)
        return 1
    part.rename(dest)
    print(f"{dest.name}: {done} bytes, md5 OK", flush=True)
    return 0


def download(cfg: dict) -> int:
    raw_dir = resolve(cfg["raw_dir"])
    for d in cfg.get("downloads", []):
        if fetch(d["url"], raw_dir / d["path"], d["md5"]):
            return 1
    return record(cfg)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--verify", action="store_true", help="check files against CHECKSUMS.txt")
    ap.add_argument("--download", action="store_true", help="fetch configured downloads (D93)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.download:
        return download(cfg)
    if args.verify:
        problems = verify(resolve(cfg["raw_dir"]), resolve(cfg["checksums_file"]))
        for p in problems:
            print(p, file=sys.stderr)
        print("checksums OK" if not problems else f"{len(problems)} problem(s)")
        return 1 if problems else 0
    return record(cfg)


if __name__ == "__main__":
    sys.exit(main())
