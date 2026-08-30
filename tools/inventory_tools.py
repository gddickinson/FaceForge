"""Inventory every script in ``tools/``: what it is, and whether it still runs.

Written because ``tools/`` reached 39 scripts and no one could say which were
maintained.  The facts here are measured, not remembered:

* **importable** -- the module is imported in a subprocess.  A script whose
  imports have rotted (a renamed module, a deleted helper) fails here, which is
  the cheapest available proxy for "does it still run": actually *running* most
  of these needs the 1.2 GB asset set, a GPU, or several minutes.
* **entry point** -- whether it has an ``if __name__ == "__main__"`` block.  A
  script without one is a library other tools import, not something to run.
* **imported by** -- which other files import it.  A module with importers
  cannot be retired without moving its callers first, whatever its verdict.
* **ruff** -- findings from the project's own lint configuration.

Verdicts are *not* generated: they are editorial and live in ``tools/README.md``.
This script produces the evidence that a verdict should be based on.

Usage::

    python tools/inventory_tools.py                  # table to stdout
    python tools/inventory_tools.py --json out.json  # machine-readable
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = REPO_ROOT / "tools"
SEARCH_ROOTS = ("tools", "tests", "src", ".github")


@dataclass
class ScriptFacts:
    """Measured facts about one script.  No judgements."""

    name: str
    lines: int
    summary: str = ""
    has_entry_point: bool = False
    importable: bool = False
    import_error: str = ""
    imports_tools: list[str] = field(default_factory=list)
    imported_by: list[str] = field(default_factory=list)
    ruff_findings: int = -1


def parse_script(path: Path) -> ScriptFacts:
    source = path.read_text(encoding="utf-8", errors="replace")
    facts = ScriptFacts(name=path.name, lines=len(source.splitlines()))
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        facts.import_error = f"SyntaxError line {exc.lineno}"
        return facts

    doc = ast.get_docstring(tree) or ""
    facts.summary = doc.strip().splitlines()[0] if doc.strip() else ""
    facts.has_entry_point = any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and getattr(node.test.left, "id", "") == "__name__"
        for node in tree.body
    )
    tool_imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("tools."):
            tool_imports.add(node.module.split(".")[1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("tools."):
                    tool_imports.add(alias.name.split(".")[1])
    facts.imports_tools = sorted(tool_imports - {path.stem})
    return facts


def check_importable(module: str) -> tuple[bool, str]:
    """Import ``tools.<module>`` in a subprocess.

    A subprocess because a rotted import can raise at module scope, and because
    a few of these scripts touch global state (OpenGL, Qt) that should not leak
    into the inventory run.
    """
    result = subprocess.run(
        [sys.executable, "-c", f"import tools.{module}"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=180,
        env=_env_with_src(),
    )
    if result.returncode == 0:
        return True, ""
    tail = (result.stderr or "").strip().splitlines()
    return False, tail[-1][:160] if tail else f"exit {result.returncode}"


def _env_with_src() -> dict:
    import os
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"src:.{':' + existing if existing else ''}"
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    return env


def find_importers(names: list[str]) -> dict[str, list[str]]:
    """Which files import each ``tools.<name>``, by scanning the tree."""
    out: dict[str, list[str]] = {name: [] for name in names}
    for root in SEARCH_ROOTS:
        base = REPO_ROOT / root
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            text = path.read_text(encoding="utf-8", errors="replace")
            rel = str(path.relative_to(REPO_ROOT))
            for name in names:
                if path.stem == name:
                    continue
                if f"tools.{name}" in text or f"from tools import {name}" in text:
                    out[name].append(rel)
    return out


def ruff_counts() -> dict[str, int]:
    """Findings per file from the project's ruff configuration."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "--output-format", "json",
             "tools"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=180,
        )
        rows = json.loads(result.stdout or "[]")
    except (OSError, ValueError, subprocess.SubprocessError):
        return {}
    counts: dict[str, int] = {}
    for row in rows:
        name = Path(row.get("filename", "")).name
        counts[name] = counts.get(name, 0) + 1
    return counts


def build(skip_imports: bool = False) -> list[ScriptFacts]:
    scripts = sorted(p for p in TOOLS_DIR.glob("*.py")
                     if p.name != "__init__.py")
    facts = [parse_script(path) for path in scripts]
    importers = find_importers([p.stem for p in scripts])
    ruff = ruff_counts()
    for path, item in zip(scripts, facts, strict=True):
        item.imported_by = importers.get(path.stem, [])
        item.ruff_findings = ruff.get(path.name, 0)
        if not skip_imports and not item.import_error:
            item.importable, item.import_error = check_importable(path.stem)
    return facts


def render_table(facts: list[ScriptFacts]) -> str:
    header = (f"{'script':36s} {'lines':>6s} {'entry':>5s} {'imp':>4s} "
              f"{'used-by':>7s} {'ruff':>4s}  summary")
    lines = [header, "-" * len(header)]
    for item in facts:
        lines.append(
            f"{item.name:36s} {item.lines:6d} "
            f"{'yes' if item.has_entry_point else '-':>5s} "
            f"{'ok' if item.importable else 'FAIL':>4s} "
            f"{len(item.imported_by):7d} {item.ruff_findings:4d}  "
            f"{item.summary[:60]}")
    broken = [i for i in facts if not i.importable]
    lines.append("")
    lines.append(f"{len(facts)} scripts, {len(broken)} not importable")
    for item in broken:
        lines.append(f"  {item.name}: {item.import_error}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path,
                        help="also write the facts to this path")
    parser.add_argument("--skip-imports", action="store_true",
                        help="skip the subprocess import check (much faster)")
    args = parser.parse_args(argv)

    facts = build(skip_imports=args.skip_imports)
    print(render_table(facts))
    if args.json:
        args.json.write_text(
            json.dumps([asdict(f) for f in facts], indent=1) + "\n",
            encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
