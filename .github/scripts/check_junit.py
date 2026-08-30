#!/usr/bin/env python3
"""Fail a CI job when a pytest run passed by not running.

A green pytest exit code says "nothing failed".  It does not say "the tests
ran".  Every failure mode this project has actually hit produces a green exit:

* ``assets/stl`` is a tracked symlink pointing outside the repo, so a fresh
  checkout has a *dangling* link.  The asset-gated modules carry ``skipif``
  guards, so 33 tests became silent skips and the suite stayed green -- the
  defect that motivated this review.
* ``test_shader_compile.py`` skips its entire module when ``glslangValidator``
  is absent.  A CI job that forgets to install it reports success while
  compiling nothing.
* A collection-time exception in a conftest can deselect a whole directory.

So each test job states what it expects to have run, and this script checks the
JUnit XML against that.  Usage::

    check_junit.py report.xml --min-tests 900 --max-skips 20
    check_junit.py report.xml --min-tests 247 --max-skips 0

Exits 0 when every constraint holds, 1 otherwise, 2 on a malformed report.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def summarise(path: Path) -> dict[str, int]:
    """Total counters across every ``<testsuite>`` in a JUnit report."""
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    if not suites:
        raise ValueError(f"{path}: no <testsuite> element")
    keys = ("tests", "failures", "errors", "skipped")
    totals = dict.fromkeys(keys, 0)
    for suite in suites:
        for key in keys:
            totals[key] += int(suite.get(key, 0))
    totals["ran"] = totals["tests"] - totals["skipped"]
    return totals


def skipped_names(path: Path, limit: int = 25) -> list[str]:
    """``module::test -- reason`` for each skipped case, for the failure message."""
    out = []
    for case in ET.parse(path).getroot().iter("testcase"):
        skip = case.find("skipped")
        if skip is None:
            continue
        name = f"{case.get('classname', '?')}::{case.get('name', '?')}"
        out.append(f"{name} -- {(skip.get('message') or '').strip()[:120]}")
        if len(out) >= limit:
            break
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("report", type=Path, help="pytest --junitxml output")
    parser.add_argument(
        "--min-tests", type=int, default=1,
        help="fail if fewer than this many tests were collected",
    )
    parser.add_argument(
        "--max-skips", type=int, default=None,
        help="fail if more than this many tests were skipped",
    )
    args = parser.parse_args(argv)

    if not args.report.is_file():
        print(f"error: no JUnit report at {args.report} — did pytest run?",
              file=sys.stderr)
        return 2
    try:
        totals = summarise(args.report)
    except (ET.ParseError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(
        f"{args.report}: {totals['tests']} collected, {totals['ran']} ran, "
        f"{totals['skipped']} skipped, {totals['failures']} failed, "
        f"{totals['errors']} errored"
    )

    problems = []
    if totals["tests"] < args.min_tests:
        problems.append(
            f"collected {totals['tests']} tests, expected at least "
            f"{args.min_tests}. Tests disappeared rather than failed — check for "
            "a collection error or a marker deselecting more than intended."
        )
    if args.max_skips is not None and totals["skipped"] > args.max_skips:
        problems.append(
            f"{totals['skipped']} tests skipped, budget is {args.max_skips}. "
            "A skip in this job means a dependency the job is supposed to "
            "provide was absent, so the job proved nothing."
        )
    if totals["failures"] or totals["errors"]:
        problems.append(
            f"{totals['failures']} failures and {totals['errors']} errors "
            "(pytest should already have failed the step)"
        )

    if not problems:
        return 0
    print("\nJUnit gate FAILED:", file=sys.stderr)
    for problem in problems:
        print(f"  - {problem}", file=sys.stderr)
    if args.max_skips is not None and totals["skipped"] > args.max_skips:
        print("\nskipped tests:", file=sys.stderr)
        for name in skipped_names(args.report):
            print(f"  {name}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
