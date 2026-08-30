#!/usr/bin/env python3
"""Hold the ruff finding count at or below a baseline, per rule family.

The tuned ``select`` in ``pyproject.toml`` was chosen to surface findings that
are *bugs* rather than style, and on the tree this script was written against
it reports 96 of them (measured with ruff 0.16.5):

    35  B905    zip-without-explicit-strict
    32  BLE001  blind-except
    18  B007    unused-loop-control-variable
     8  LOG015  root-logger-call
     3  B023    function-uses-loop-variable

Those are real and worth fixing, but demanding all 96 be gone before CI can
exist would mean CI never exists.  Two gates instead:

1. The families that are already clean (``F``, ``E9``, the ``E7`` comparisons,
   ``S110``/``S112``, ``G201``, ``LOG`` other than ``LOG015``) are a hard gate
   in the workflow -- any new finding there fails the build.
2. This script ratchets the rest: the count per rule code may fall, never
   rise.  Introducing a new blind except fails now; fixing debt and
   regenerating the baseline tightens the gate permanently.

Per-code accounting rather than a single total matters: a lump sum lets someone
delete twelve ``zip`` calls and add twelve blind excepts with CI silent.

Usage::

    ruff_ratchet.py --baseline .github/ruff_baseline.json
    ruff_ratchet.py --baseline .github/ruff_baseline.json --write   # regenerate
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

# .github/scripts is included: these three scripts gate the build, so a broken
# one is as bad as a broken test. They are clean, and contribute 0 to the
# baseline below.
TARGETS = ("src", "tests", "tools", ".github/scripts")


def current_counts(ruff: str = "ruff") -> Counter:
    """{rule code: count} from ruff's own JSON output, using pyproject's select."""
    proc = subprocess.run(
        [ruff, "check", "-q", "--output-format", "json", *TARGETS],
        capture_output=True, text=True, check=False,
    )
    if proc.stdout.strip() == "":
        if proc.returncode not in (0, 1):
            raise RuntimeError(f"ruff failed: {proc.stderr.strip()}")
        return Counter()
    try:
        findings = json.loads(proc.stdout)
    except ValueError as exc:
        raise RuntimeError(
            f"could not parse ruff output: {exc}\n{proc.stderr.strip()}"
        ) from exc
    return Counter(f["code"] for f in findings)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", type=Path,
                        default=Path(".github/ruff_baseline.json"))
    parser.add_argument("--ruff", default="ruff", help="ruff executable")
    parser.add_argument("--write", action="store_true",
                        help="overwrite the baseline with the current counts")
    args = parser.parse_args(argv)

    counts = current_counts(args.ruff)
    total = sum(counts.values())

    if args.write:
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(
            json.dumps({"total": total, "by_code": dict(sorted(counts.items()))},
                       indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.baseline}: {total} findings")
        return 0

    if not args.baseline.is_file():
        print(f"error: no baseline at {args.baseline}; regenerate with --write",
              file=sys.stderr)
        return 2
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))["by_code"]

    print(f"ruff findings: {total} now, {sum(baseline.values())} in baseline")
    regressions, improvements = [], []
    for code in sorted(set(counts) | set(baseline)):
        now, was = counts.get(code, 0), baseline.get(code, 0)
        if now > was:
            regressions.append(f"{code}: {was} -> {now}  (+{now - was})")
        elif now < was:
            improvements.append(f"{code}: {was} -> {now}  ({now - was})")

    for line in improvements:
        print(f"  improved  {line}")
    if not regressions:
        if improvements:
            print("\nDebt went down. Regenerate the baseline to lock the gain in:")
            print("  python .github/scripts/ruff_ratchet.py --write")
        return 0

    print("\nruff ratchet FAILED — new findings in families that had a baseline:",
          file=sys.stderr)
    for line in regressions:
        print(f"  {line}", file=sys.stderr)
    print(
        "\nFix them, or (if deliberate) run "
        "`python .github/scripts/ruff_ratchet.py --write` and explain in the PR.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
