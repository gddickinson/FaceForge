"""Every action unit the product offers must be one the engine implements.

The defect this locks out: ``TargetAU.set`` and ``FaceState.set_au`` were bare
``setattr`` calls, so any string became an attribute.  The timeline editor
offered 18 AUs while ``faceforge.anatomy.facs`` reads 12, and the six extras
(AU7, AU10, AU14, AU17, AU23, AU24) could be keyframed, were stored without
complaint, and were then read by nothing -- so those tracks moved no muscle and
nothing reported why.  ``AU46`` in the wink preset is the same story.

Two directions are checked, because either drift is a bug:

* an AU offered to the user but not read by the engine is an inert control;
* an AU read by the engine but absent from ``AU_IDS`` could never be set
  through the validating path.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

from faceforge.core.state import AU_IDS, StateManager

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "faceforge"


def test_au_ids_has_no_duplicates_and_is_not_empty():
    assert AU_IDS, "AU_IDS is empty"
    assert len(set(AU_IDS)) == len(AU_IDS), f"duplicates in AU_IDS: {AU_IDS}"


def test_an_unknown_au_id_is_not_stored():
    """A mistyped id must be refused, not silently kept where nothing reads it."""
    target = StateManager().target_au
    target.set("AU999", 1.0)
    assert "AU999" not in vars(target), \
        "an unknown AU id was stored as an attribute; a typo would be silent"
    assert target.get("AU999") == 0.0


def test_a_declared_au_still_round_trips():
    """The validation must not have broken the normal path."""
    target = StateManager().target_au
    for au in AU_IDS:
        target.set(au, 0.5)
        assert target.get(au) == pytest.approx(0.5), f"{au} did not round trip"


def test_every_au_the_facs_engine_reads_is_declared():
    """The engine's reads define what actually works; AU_IDS must cover them."""
    facs = (SRC / "anatomy" / "facs.py").read_text()
    read = set(re.findall(r'au\.get\(\s*"(AU\d+)"', facs))
    assert read, "found no AU reads in facs.py -- has the access pattern changed?"
    missing = sorted(read - set(AU_IDS), key=lambda s: int(s[2:]))
    assert not missing, (
        f"facs.py reads {missing}, which are absent from AU_IDS, so they can "
        f"never be set through the validating path"
    )


def test_no_module_offers_an_au_the_engine_cannot_use():
    """An AU named in a UI or preset literal but unimplemented is an inert control.

    Scans string literals rather than behaviour, which is the level the original
    defect lived at: an 18-entry literal list beside a 12-entry engine.
    """
    known = set(AU_IDS)
    offenders: dict[str, list[str]] = {}

    for path in SRC.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        found = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and re.fullmatch(r"AU\d+", node.value)
        }
        unknown = sorted(found - known, key=lambda s: int(s[2:]))
        if unknown:
            offenders[str(path.relative_to(SRC))] = unknown

    assert not offenders, (
        "these modules name action units the FACS engine does not implement, so "
        "the corresponding controls or preset entries do nothing:\n  "
        + "\n  ".join(f"{k}: {v}" for k, v in sorted(offenders.items()))
    )
