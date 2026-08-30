"""Anatomy configs must describe the structures they actually reference.

Every config entry pairs an FMA id with a display name and, in some files, a
behavioural flag such as ``jaw: upper|lower`` which decides what the structure
is parented to.  Nothing checked those against the ontology, and the result was
teeth.json: 20 of 34 entries wrong, including six that were not teeth at all
(nasalis, depressor septi nasi and procerus were configured as lower incisors
and canines, so they were attached to the mandible and swung with the mouth).

The class of bug matters more than that instance.  A wrong id in an anatomy
config is invisible: the mesh loads, renders and is labelled confidently with
the wrong name, and for a teaching tool a confident wrong label is the worst
possible failure.  These tests compare the authored configs against
assets/config/fma_labels.json, which is generated from the ontology and is the
only independent authority available.
"""

from __future__ import annotations

import json

import pytest

from faceforge.constants import CONFIG_DIR
from faceforge.loaders.stl_batch_loader import load_fma_labels


@pytest.fixture(scope="module")
def fma():
    labels = load_fma_labels()
    if not labels:
        pytest.skip("FMA crosswalk unavailable")
    return labels


def _entries(filename):
    path = CONFIG_DIR / filename
    if not path.exists():
        pytest.skip(f"{filename} not present")
    d = json.loads(path.read_text())
    return d if isinstance(d, list) else d.get("structures", d)


def test_teeth_config_contains_only_teeth(fma):
    """The defect that put nasal muscles on the jaw pivot."""
    offenders = []
    for it in _entries("teeth.json"):
        pref = fma.get(it["stl"], {}).get("preferred_label", "")
        if pref and "tooth" not in pref.lower():
            offenders.append(f"{it['stl']} configured as {it['name']!r} "
                             f"is actually {pref!r}")
    assert not offenders, (
        "teeth.json references structures that are not teeth:\n  "
        + "\n  ".join(offenders)
    )


def test_teeth_jaw_assignment_matches_the_ontology(fma):
    """``jaw`` decides whether a tooth is parented under jawPivot.

    Getting it wrong is silent: the tooth renders in the right place at rest
    and moves with the wrong bone as soon as the mouth opens.
    """
    wrong = []
    for it in _entries("teeth.json"):
        pref = fma.get(it["stl"], {}).get("preferred_label", "").lower()
        if not pref:
            continue
        true_jaw = ("upper" if "upper" in pref else
                    "lower" if "lower" in pref else None)
        if true_jaw and it.get("jaw") != true_jaw:
            wrong.append(f"{it['stl']} ({pref}) flagged jaw={it.get('jaw')!r}, "
                         f"ontology says {true_jaw!r}")
    assert not wrong, "teeth.json jaw assignments disagree with the ontology:\n  " \
                      + "\n  ".join(wrong)


def test_teeth_names_agree_with_the_ontology(fma):
    """A display name must not contradict the structure it labels.

    Checked on the load-bearing tokens (side, jaw, tooth type) rather than by
    string equality, so the file may keep its short house style.
    """
    TYPES = ("molar", "premolar", "canine", "incisor")
    wrong = []
    for it in _entries("teeth.json"):
        pref = fma.get(it["stl"], {}).get("preferred_label", "").lower()
        if not pref:
            continue
        name = it["name"].lower()
        for token, aliases in (("upper", ("upper",)), ("lower", ("lower",)),
                               ("right", ("right", " r ", "r ")),
                               ("left", ("left", " l ", "l "))):
            if token in pref and not any(a in f" {name} " for a in aliases):
                wrong.append(f"{it['stl']}: name {it['name']!r} omits/contradicts "
                             f"{token!r} from {pref!r}")
        kind = next((k for k in TYPES if k in pref), None)
        if kind and kind not in name:
            wrong.append(f"{it['stl']}: name {it['name']!r} is not a {kind} "
                         f"but the ontology says {pref!r}")
    assert not wrong, "teeth.json names contradict the ontology:\n  " \
                      + "\n  ".join(wrong)


def test_only_the_mandible_is_jaw_attached(fma):
    """Exactly one skull bone may rotate with the jaw."""
    attached = [it for it in _entries("skull_bones.json")
                if it.get("jaw_attached")]
    assert len(attached) == 1, \
        f"expected exactly one jaw_attached bone, got {[a['name'] for a in attached]}"
    pref = fma.get(attached[0]["stl"], {}).get("preferred_label", "").lower()
    if pref:
        assert "mandible" in pref, (
            f"the jaw_attached bone is {pref!r}, not the mandible"
        )


@pytest.mark.parametrize("filename", [
    "teeth.json", "skull_bones.json", "brain.json", "organs.json",
])
def test_config_ids_resolve_in_the_crosswalk(fma, filename):
    """An id absent from the crosswalk cannot be verified or cited."""
    entries = _entries(filename)
    missing = [it.get("stl") for it in entries
               if isinstance(it, dict) and it.get("stl")
               and it["stl"] not in fma]
    # Not a hard failure for every file -- report the proportion so a
    # regression is visible without pinning an exact number.
    assert len(missing) <= len(entries) * 0.05, (
        f"{filename}: {len(missing)}/{len(entries)} ids absent from the "
        f"crosswalk: {missing[:8]}"
    )
