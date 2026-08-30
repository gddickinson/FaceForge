"""Verify the BodyParts3D asset set and build the welded-mesh disk cache.

Why this exists
---------------
The 1.2 GB BodyParts3D STL dataset lives outside the repository, reached
through ``assets/stl``, a symlink.  Before the 2026-08 review that symlink
pointed at a path that did not exist, and *nothing said so*: the asset-gated
test modules carry ``skipif`` guards, so a dangling link turned 33 tests into
silent skips and left the suite green.  A missing dataset must be a loud,
specific diagnostic, not an absence of output.

So the emphasis here is verification, not downloading:

* ``verify``   -- is the dataset present, complete, and readable?  Exactly
                  which meshes are missing, and which of the several distinct
                  failure modes (dangling symlink, absent directory, empty
                  directory, partial copy, truncated files) is in play?
* ``cache``    -- build the ``.npz`` welded-geometry cache that
                  ``loaders.stl_parser`` reads, so the first real run is warm.
* ``manifest`` -- print the set of mesh ids the configs actually require.
* ``fetch``    -- **not implemented.**  See the note below.

On ``fetch``
------------
No download is implemented, and no URL is hardcoded.  This repository does not
record a retrieval URL for the dataset anywhere (checked: ``README.md``,
``assets/README.md``, ``extract_data.py``), the download host was not reachable
to test from the environment this tool was written in, and BodyParts3D is
CC BY-SA 2.1 Japan — a share-alike licence whose attribution requirement
travels with the files.  A confidently-worded URL that 404s is worse than an
honest gap: it converts "you need to fetch the dataset" into "the tool is
broken".  ``fetch`` therefore prints acquisition instructions and exits 3.

Safety
------
This tool never deletes, never overwrites and never writes anything under
``assets/``.  The only path it writes is the mesh cache directory, which
defaults to ``~/.cache/faceforge/meshes`` (outside the repo); a cache directory
that resolves inside the asset tree is refused outright, so a
``--cache-dir assets/stl`` typo cannot scribble into the dataset — nor through
the symlink into the dataset's real home outside the repo.  ``--dry-run``
reports the writes it would make and performs none.

Usage
-----
::

    python -m tools.fetch_assets verify
    python -m tools.fetch_assets verify --json
    python -m tools.fetch_assets verify --stl-dir /somewhere/else
    python -m tools.fetch_assets cache --dry-run
    python -m tools.fetch_assets cache
    python -m tools.fetch_assets manifest

Exit codes
----------
0   dataset present and complete (``verify``); cache built (``cache``)
1   dataset absent, incomplete or unreadable
2   usage or configuration error (e.g. an unsafe ``--cache-dir``)
3   ``fetch`` -- deliberately unimplemented
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# ── manifest ────────────────────────────────────────────────────────
#
# The required mesh set is derived from the JSON configs rather than frozen
# into a list here.  A frozen list is a second source of truth that drifts the
# first time someone adds a muscle: the configs are what the loaders read, so
# they are what "required" means.  Every config entry names its mesh in a
# "stl" key holding a bare id (e.g. "FMA49027" -> FMA49027.stl).

_MESH_KEY = "stl"
# coordinate_transform.json holds a directory prefix under this key, not a
# mesh id.  It is the only such key and must not be mistaken for a mesh.
_NOT_A_MESH_KEY = "stl_base"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _collect_mesh_ids(config_dir: Path) -> tuple[dict[str, list[str]], list[str]]:
    """Return ({mesh_id: [config files requiring it]}, [unreadable config files]).

    Walks every JSON file under *config_dir* recursively and collects the value
    of every ``"stl"`` key, at any nesting depth.
    """
    required: dict[str, list[str]] = {}
    unreadable: list[str] = []

    def walk(node: object, origin: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == _MESH_KEY and isinstance(value, str) and value:
                    required.setdefault(value, [])
                    if origin not in required[value]:
                        required[value].append(origin)
                elif key == _NOT_A_MESH_KEY:
                    continue
                else:
                    walk(value, origin)
        elif isinstance(node, list):
            for item in node:
                walk(item, origin)

    if not config_dir.is_dir():
        return required, [f"{config_dir} (config directory does not exist)"]

    for path in sorted(config_dir.rglob("*.json")):
        rel = str(path.relative_to(config_dir.parent.parent))
        try:
            walk(json.loads(path.read_text(encoding="utf-8")), rel)
        except (OSError, ValueError) as exc:
            unreadable.append(f"{rel}: {exc}")
    return required, unreadable


# ── per-file integrity ──────────────────────────────────────────────

_STL_HEADER_BYTES = 84          # 80-byte header + uint32 triangle count
_STL_TRIANGLE_BYTES = 50        # 3+9 float32 + uint16 attribute count


def _stl_defect(path: Path) -> str | None:
    """Return a one-line defect description, or None if the file looks sound.

    A binary STL is exactly ``84 + 50 * n`` bytes, and the declared triangle
    count must agree with the file length.  Reading 84 bytes catches the two
    failure modes a partial copy produces — zero-length placeholders and
    truncated transfers — without parsing 1.2 GB of geometry.
    """
    try:
        size = path.stat().st_size
    except OSError as exc:
        return f"unreadable: {exc}"
    if size == 0:
        return "zero bytes"
    if size < _STL_HEADER_BYTES:
        return f"{size} bytes — shorter than an STL header ({_STL_HEADER_BYTES})"
    try:
        with open(path, "rb") as handle:
            head = handle.read(_STL_HEADER_BYTES)
    except OSError as exc:
        return f"unreadable: {exc}"
    if len(head) < _STL_HEADER_BYTES:
        return "truncated header"
    declared = struct.unpack("<I", head[80:84])[0]
    expected = _STL_HEADER_BYTES + _STL_TRIANGLE_BYTES * declared
    if size != expected:
        return (
            f"declares {declared} triangles (expects {expected} bytes) "
            f"but file is {size} bytes"
        )
    return None


# ── report ──────────────────────────────────────────────────────────

STATE_OK = "complete"
STATE_MISSING_DIR = "missing-directory"
STATE_DANGLING_SYMLINK = "dangling-symlink"
STATE_EMPTY = "empty"
STATE_PARTIAL = "partial"
STATE_CORRUPT = "corrupt-files"


@dataclass
class AssetReport:
    """Everything ``verify`` learned, renderable as text or JSON."""

    stl_dir: str
    state: str
    resolved: str | None = None
    symlink_target: str | None = None
    required: int = 0
    present: int = 0
    missing: list[str] = field(default_factory=list)
    defective: list[tuple[str, str]] = field(default_factory=list)
    unreferenced: list[str] = field(default_factory=list)
    files_on_disk: int = 0
    bytes_on_disk: int = 0
    config_errors: list[str] = field(default_factory=list)
    required_by: dict[str, list[str]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.state == STATE_OK

    def as_dict(self) -> dict:
        return {
            "stl_dir": self.stl_dir,
            "state": self.state,
            "ok": self.ok,
            "resolved": self.resolved,
            "symlink_target": self.symlink_target,
            "required": self.required,
            "present": self.present,
            "missing_count": len(self.missing),
            "missing": self.missing,
            "defective": [{"file": f, "problem": p} for f, p in self.defective],
            "unreferenced_count": len(self.unreferenced),
            "files_on_disk": self.files_on_disk,
            "bytes_on_disk": self.bytes_on_disk,
            "config_errors": self.config_errors,
        }


def _human(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n} B"


def verify(stl_dir: Path, config_dir: Path, *, integrity: bool = True) -> AssetReport:
    """Inspect *stl_dir* against the mesh set required by *config_dir*."""
    required_by, config_errors = _collect_mesh_ids(config_dir)
    report = AssetReport(
        stl_dir=str(stl_dir),
        state=STATE_OK,
        required=len(required_by),
        config_errors=config_errors,
        required_by=required_by,
    )

    # A symlink is reported whether or not it resolves: the pre-review failure
    # was a link that existed, looked fine in `ls`, and pointed nowhere.
    if stl_dir.is_symlink():
        report.symlink_target = os.readlink(stl_dir)
    if stl_dir.exists():
        report.resolved = str(stl_dir.resolve())

    if not stl_dir.is_dir():
        report.state = (
            STATE_DANGLING_SYMLINK if stl_dir.is_symlink() else STATE_MISSING_DIR
        )
        report.missing = sorted(required_by)
        return report

    on_disk: dict[str, Path] = {}
    total = 0
    for path in stl_dir.glob("*.stl"):
        on_disk[path.stem] = path
        try:
            total += path.stat().st_size
        except OSError:
            pass
    report.files_on_disk = len(on_disk)
    report.bytes_on_disk = total

    if not on_disk:
        report.state = STATE_EMPTY
        report.missing = sorted(required_by)
        return report

    report.missing = sorted(set(required_by) - set(on_disk))
    report.present = report.required - len(report.missing)
    report.unreferenced = sorted(set(on_disk) - set(required_by))

    if integrity:
        for name in sorted(set(required_by) & set(on_disk)):
            defect = _stl_defect(on_disk[name])
            if defect is not None:
                report.defective.append((f"{name}.stl", defect))

    if report.missing:
        report.state = STATE_PARTIAL
    elif report.defective:
        report.state = STATE_CORRUPT
    return report


def render(report: AssetReport, *, limit: int = 20) -> str:
    """Human-readable verification report."""
    out: list[str] = []
    add = out.append
    add(f"BodyParts3D asset check: {report.state.upper()}")
    add(f"  asset path      {report.stl_dir}")
    if report.symlink_target is not None:
        arrow = "->" if report.resolved else "-> (BROKEN)"
        add(f"  symlink         {arrow} {report.symlink_target}")
    if report.resolved and report.resolved != report.stl_dir:
        add(f"  resolves to     {report.resolved}")
    add(f"  meshes required {report.required}  (from assets/config/**/*.json)")

    if report.state == STATE_DANGLING_SYMLINK:
        add("")
        add("  assets/stl is a symlink whose target does not exist.  This is the")
        add("  failure mode that silently disabled 33 asset-gated tests before the")
        add("  2026-08 review: the tests carry skipif guards, so a dead link reads")
        add("  as 'skipped', not 'broken'.  Repoint the symlink at your BodyParts3D")
        add("  checkout, or pass --stl-dir.")
    elif report.state == STATE_MISSING_DIR:
        add("")
        add("  The asset directory does not exist at all.")
    elif report.state == STATE_EMPTY:
        add("")
        add("  The asset directory exists but contains no .stl files.")
    else:
        add(f"  meshes present  {report.present} / {report.required}")
        add(f"  files on disk   {report.files_on_disk}  ({_human(report.bytes_on_disk)})")
        if report.unreferenced:
            add(
                f"  unreferenced    {len(report.unreferenced)} .stl file(s) on disk that "
                "no config names (harmless)"
            )

    if report.missing and report.state in (STATE_PARTIAL, STATE_CORRUPT):
        add("")
        add(f"  MISSING {len(report.missing)} required mesh(es):")
        for name in report.missing[:limit]:
            origins = ", ".join(report.required_by.get(name, [])) or "?"
            add(f"    {name}.stl   required by {origins}")
        if len(report.missing) > limit:
            add(f"    ... and {len(report.missing) - limit} more (--limit 0 for all)")

    if report.defective:
        add("")
        add(f"  DEFECTIVE {len(report.defective)} file(s) — present but not valid binary STL:")
        for name, problem in report.defective[:limit]:
            add(f"    {name}: {problem}")
        if len(report.defective) > limit:
            add(f"    ... and {len(report.defective) - limit} more")

    if report.config_errors:
        add("")
        add("  CONFIG PROBLEMS:")
        for err in report.config_errors:
            add(f"    {err}")

    if not report.ok:
        add("")
        add("  How to obtain the dataset:")
        for line in FETCH_INSTRUCTIONS:
            add(f"    {line}")
    return "\n".join(out)


FETCH_INSTRUCTIONS = (
    "BodyParts3D is published by the Database Center for Life Science (DBCLS),",
    "Japan, under CC BY-SA 2.1 Japan.  Download the STL distribution from the",
    "DBCLS BodyParts3D project pages, unpack it so that the per-structure files",
    "are named FMA<id>.stl in one flat directory, then either:",
    "",
    "  ln -s /path/to/BodyParts3D/stl  assets/stl",
    "",
    "or point the tools at it directly with --stl-dir.  Afterwards run:",
    "",
    "  python -m tools.fetch_assets verify",
    "  python -m tools.fetch_assets cache",
    "",
    "Attribution is a licence condition and must be reproduced wherever the",
    "meshes or renders derived from them appear:",
    "  BodyParts3D, (c) The Database Center for Life Science licensed under",
    "  CC Attribution-Share Alike 2.1 Japan.",
    "",
    "This tool does not download for you: no retrieval URL is recorded in this",
    "repository, and guessing one that 404s would be worse than saying so.",
)


# ── cache building ──────────────────────────────────────────────────


def _assert_safe_cache_dir(cache_dir: Path, stl_dir: Path, repo_root: Path) -> None:
    """Refuse a cache directory that could write into the asset tree.

    Checked on *resolved* paths so that a cache directory reached through the
    ``assets/stl`` symlink — i.e. pointing at the real dataset outside the
    repo — is caught too.
    """
    resolved = cache_dir.resolve()
    # Both the in-repo assets tree and wherever --stl-dir actually lands after
    # symlink resolution.  Deliberately *not* stl_dir.parent: for the in-repo
    # case that is `assets/`, already covered, and for an out-of-tree
    # --stl-dir it would forbid a sibling cache directory for no reason.
    forbidden = [(repo_root / "assets").resolve()]
    try:
        forbidden.append(stl_dir.resolve())
    except OSError:
        pass
    for bad in forbidden:
        if resolved == bad or bad in resolved.parents:
            raise ValueError(
                f"refusing --cache-dir {cache_dir}: it resolves inside the asset "
                f"tree ({bad}).  The cache must never be written into assets."
            )


def build_cache(
    stl_dir: Path,
    cache_dir: Path,
    *,
    names: list[str] | None = None,
    dry_run: bool = False,
    progress_every: int = 100,
    log=print,
) -> dict:
    """Populate the welded-geometry ``.npz`` cache for every present mesh.

    Reads STLs and writes only into *cache_dir*.  Entries already cached are
    re-read rather than recomputed (that is the cache doing its job), so a
    second call is cheap and idempotent.  Nothing under *stl_dir* is written.
    """
    files = sorted(stl_dir.glob("*.stl")) if names is None else [
        stl_dir / f"{n}.stl" for n in sorted(names) if (stl_dir / f"{n}.stl").is_file()
    ]
    stats = {
        "stl_dir": str(stl_dir),
        "cache_dir": str(cache_dir),
        "meshes": len(files),
        "dry_run": dry_run,
        "built": 0,
        "failed": [],
        "seconds": 0.0,
        "cache_bytes": 0,
    }
    if dry_run:
        log(f"[dry-run] would read {len(files)} .stl file(s) from {stl_dir}")
        log(f"[dry-run] would write .npz cache entries into {cache_dir}")
        log("[dry-run] no file under the asset directory would be written")
        return stats

    from faceforge.loaders.stl_parser import load_stl_file

    cache_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    for i, path in enumerate(files, 1):
        try:
            load_stl_file(path, indexed=True, use_cache=True, cache_dir=cache_dir)
            stats["built"] += 1
        except Exception as exc:                                    # noqa: BLE001
            # One malformed mesh must not abort a 934-file build; it is
            # reported and the rest continues.
            stats["failed"].append({"file": path.name, "error": str(exc)})
        if progress_every and i % progress_every == 0:
            log(f"  cached {i}/{len(files)}")
    stats["seconds"] = round(time.perf_counter() - start, 2)
    stats["cache_bytes"] = sum(
        p.stat().st_size for p in cache_dir.glob("*.npz") if p.is_file()
    )
    return stats


# ── CLI ─────────────────────────────────────────────────────────────


def _default_paths() -> tuple[Path, Path]:
    """(stl_dir, config_dir) from faceforge.constants, with a static fallback."""
    try:
        from faceforge.constants import CONFIG_DIR, STL_DIR
        return Path(STL_DIR), Path(CONFIG_DIR)
    except Exception:                                               # noqa: BLE001
        root = _repo_root()
        return root / "assets" / "stl", root / "assets" / "config"


def build_parser() -> argparse.ArgumentParser:
    default_stl, default_config = _default_paths()

    # The shared options are attached to the top-level parser *and* to every
    # subparser, so `verify --stl-dir X` and `--stl-dir X verify` both work.
    # argparse otherwise accepts only the second form, which is the one nobody
    # types.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--stl-dir", type=Path, default=default_stl,
        help=f"asset directory to check (default: {default_stl})",
    )
    common.add_argument(
        "--config-dir", type=Path, default=default_config,
        help="config tree the required mesh set is derived from",
    )
    common.add_argument("--json", action="store_true", help="machine-readable output")

    parser = argparse.ArgumentParser(
        prog="python -m tools.fetch_assets",
        description="Verify the BodyParts3D asset set and build the mesh cache.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[common],
    )
    sub = parser.add_subparsers(dest="command")

    p_verify = sub.add_parser(
        "verify", parents=[common],
        help="check the dataset is present and complete",
    )
    p_verify.add_argument(
        "--limit", type=int, default=20,
        help="how many missing meshes to list (0 = all)",
    )
    p_verify.add_argument(
        "--no-integrity", action="store_true",
        help="skip the per-file STL header check",
    )
    p_verify.add_argument(
        "--allow-missing", type=int, default=0, metavar="N",
        help=(
            "exit 0 even if up to N required meshes are absent (they are still "
            "listed).  Motivating case: this repo's expression_muscles.json names "
            "FMA49041/FMA49042 (levator palpebrae superioris L/R), which are not "
            "in the BodyParts3D STL distribution -- so a complete, correct install "
            "still reports 930/932.  Use --allow-missing 2 to gate on that."
        ),
    )

    p_cache = sub.add_parser(
        "cache", parents=[common], help="build the .npz welded-mesh cache",
    )
    p_cache.add_argument(
        "--cache-dir", type=Path, default=None,
        help="cache directory (default: the one loaders.stl_parser uses)",
    )
    p_cache.add_argument(
        "--dry-run", action="store_true",
        help="report what would be read and written; write nothing",
    )
    p_cache.add_argument(
        "--required-only", action="store_true",
        help="cache only meshes the configs reference, not every .stl present",
    )

    sub.add_parser(
        "manifest", parents=[common],
        help="print the mesh ids the configs require",
    )
    sub.add_parser(
        "fetch", parents=[common],
        help="(not implemented) print acquisition instructions",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = args.command or "verify"

    if command == "fetch":
        if args.json:
            print(json.dumps({"implemented": False,
                              "instructions": list(FETCH_INSTRUCTIONS)}, indent=2))
        else:
            print("fetch is not implemented — see below.\n")
            print("\n".join(FETCH_INSTRUCTIONS))
        return 3

    if command == "manifest":
        required_by, errors = _collect_mesh_ids(args.config_dir)
        if args.json:
            print(json.dumps(
                {"count": len(required_by),
                 "required": {k: v for k, v in sorted(required_by.items())},
                 "config_errors": errors},
                indent=2,
            ))
        else:
            for name in sorted(required_by):
                print(name)
            print(f"# {len(required_by)} meshes required", file=sys.stderr)
            for err in errors:
                print(f"# CONFIG ERROR {err}", file=sys.stderr)
        return 1 if errors else 0

    report = verify(
        args.stl_dir, args.config_dir,
        integrity=not getattr(args, "no_integrity", False),
    )

    if command == "verify":
        limit = args.limit if args.limit > 0 else len(report.missing) + len(report.defective)
        print(json.dumps(report.as_dict(), indent=2) if args.json
              else render(report, limit=limit))
        if report.ok:
            return 0
        tolerated = (
            report.state == STATE_PARTIAL
            and not report.defective
            and len(report.missing) <= args.allow_missing
        )
        return 0 if tolerated else 1

    # command == "cache"
    if report.state in (STATE_MISSING_DIR, STATE_DANGLING_SYMLINK, STATE_EMPTY):
        print(render(report), file=sys.stderr)
        print("\nnothing to cache: no assets present.", file=sys.stderr)
        return 1

    if args.cache_dir is not None:
        cache_dir = args.cache_dir
    else:
        from faceforge.loaders.stl_parser import mesh_cache_dir
        cache_dir = mesh_cache_dir()
    try:
        _assert_safe_cache_dir(Path(cache_dir), args.stl_dir, _repo_root())
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    stats = build_cache(
        args.stl_dir, Path(cache_dir),
        names=sorted(report.required_by) if args.required_only else None,
        dry_run=args.dry_run,
        log=(lambda *a: None) if args.json else print,
    )
    stats["verify"] = report.as_dict()
    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        if not args.dry_run:
            print(
                f"cached {stats['built']}/{stats['meshes']} meshes in "
                f"{stats['seconds']} s -> {cache_dir} "
                f"({_human(stats['cache_bytes'])})"
            )
            for failure in stats["failed"]:
                print(f"  FAILED {failure['file']}: {failure['error']}")
            if not report.ok:
                print(
                    f"note: dataset is {report.state} "
                    f"({report.present}/{report.required} required meshes present) "
                    "— run `verify` for detail"
                )
    # Cache-build success is judged on what it was asked to cache, not on
    # dataset completeness: a partial dataset can still be fully cached, and
    # `verify` is the command that reports completeness.
    return 1 if stats["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
