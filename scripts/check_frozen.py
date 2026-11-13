"""D112 freeze guard. Exit 0 only if the frozen primary (v1.0-dev-frozen) is what will run.

    uv run python scripts/check_frozen.py --config configs/phase5.yaml --policy ig_d109

Phase 5F calls this before anything runs. See src/hpcdiag/freeze.py for the checks.
"""

from __future__ import annotations

import argparse
import sys

from hpcdiag.freeze import run_guard


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--policy", required=True)
    a = ap.parse_args()
    ok, msgs = run_guard(a.policy, a.config)
    for m in msgs:
        print(m)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
