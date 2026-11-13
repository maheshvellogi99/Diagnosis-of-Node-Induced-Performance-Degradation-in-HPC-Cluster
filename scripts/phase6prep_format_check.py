"""Which Python files differ from tag v1.0-dev-frozen, and is each difference formatting only?

    uv run python scripts/phase6prep_format_check.py

A file counts as formatting-only when its AST (ast.dump) at the tag equals its AST now. Prints three
lists: AST-identical, AST-different, new. Read-only.
"""

from __future__ import annotations

import ast
import subprocess
import sys

TAG = "v1.0-dev-frozen"


def main() -> int:
    files = subprocess.run(
        ["git", "diff", "--name-only", TAG, "HEAD", "--", "src", "tests", "scripts", "cluster"],
        capture_output=True, text=True, check=True,
    ).stdout.split()  # fmt: skip
    same, diff, new = [], [], []
    for f in (f for f in files if f.endswith(".py")):
        old = subprocess.run(["git", "show", f"{TAG}:{f}"], capture_output=True, text=True)
        if old.returncode:
            new.append(f)
            continue
        with open(f) as fh:
            cur = fh.read()
        (same if ast.dump(ast.parse(old.stdout)) == ast.dump(ast.parse(cur)) else diff).append(f)
    for lab, xs in (("AST-identical (formatting only)", same), ("AST differs", diff), ("new", new)):
        print(f"{lab}: {len(xs)}")
        for f in xs:
            print(f"  {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
