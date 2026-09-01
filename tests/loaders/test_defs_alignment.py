"""A failed STL must not shift every later structure onto another's definition.

``load_stl_batch`` skips an entry it cannot load, so ``result.nodes`` and
``result.meshes`` are shorter than the input ``defs``.  Thirteen loops were
affected: twelve zipped loader output directly against that input list, and a
thirteenth (``DemandLoaders._register_items``) received both lists as
parameters, with all five of its callers passing the input list.  Either way the
result is that every structure after the first failure is paired with a
DIFFERENT structure's definition -- and those definitions carry behavioural
flags, not just names:

  * ``skull_bones.json`` marks exactly one bone ``jaw_attached``, so a shifted
    pairing can rotate an arbitrary cranial bone with the mandible;
  * ``teeth.json`` marks each tooth ``upper``/``lower``, deciding whether it is
    parented under jawPivot;
  * ``vertebrae`` definitions carry the spinal level.

None of it raises.  The meshes load, render, and are labelled confidently with
another structure's name and flags -- which for an anatomy tool is the worst
available failure mode.  ``defs_loaded`` is the aligned list; these tests pin
the alignment and the strictness that now surrounds it.
"""

from __future__ import annotations

import pytest

from faceforge.constants import STL_DIR
from faceforge.loaders.stl_batch_loader import load_stl_batch

pytestmark = pytest.mark.skipif(
    not (STL_DIR.exists() and any(STL_DIR.glob("*.stl"))),
    reason="BodyParts3D STL dataset not present",
)

# Two real ids, with a deliberately absent one between them.  The missing
# entry carries jaw_attached so a misalignment is unmistakable: the bone AFTER
# it would inherit the flag and rotate with the jaw.
DEFS = [
    {"stl": "FMA52734", "name": "Frontal Bone"},
    {"stl": "FMA_DEFINITELY_ABSENT", "name": "Missing", "jaw_attached": True},
    {"stl": "FMA52788", "name": "R Parietal Bone"},
]


@pytest.fixture(scope="module")
def result():
    return load_stl_batch(DEFS, label="alignment_probe")


def test_the_missing_entry_is_reported(result):
    assert result.failed, "a missing STL must be reported, not swallowed"
    assert len(result.nodes) == len(DEFS) - 1


def test_defs_loaded_is_aligned_with_nodes_and_meshes(result):
    assert len(result.defs_loaded) == len(result.nodes) == len(result.meshes)
    for node, defn in zip(result.nodes, result.defs_loaded, strict=True):
        assert node.name == defn["name"], (
            f"node {node.name!r} paired with definition {defn['name']!r}"
        )


def test_defs_loaded_omits_the_failed_entry(result):
    names = [d["name"] for d in result.defs_loaded]
    assert "Missing" not in names
    assert names == ["Frontal Bone", "R Parietal Bone"]


def test_zipping_against_the_input_defs_would_have_misaligned(result):
    """The negative control: without defs_loaded the bug is real, not theoretical.

    If this ever stops holding, the loader has started emitting a placeholder
    for failures and the tests above are no longer proving anything.
    """
    mispaired = [
        (node.name, defn["name"])
        for node, defn in zip(result.nodes, DEFS, strict=False)
        if node.name != defn["name"]
    ]
    assert mispaired, (
        "expected the naive zip to mispair; if it no longer does, these tests "
        "are vacuous and need rewriting"
    )
    # And specifically: the surviving bone would have inherited jaw_attached.
    naive = dict(zip([n.name for n in result.nodes], DEFS, strict=False))
    assert naive["R Parietal Bone"].get("jaw_attached") is True, (
        "the naive pairing should hand a cranial bone the mandible's flag"
    )


def test_no_call_site_zips_loader_output_against_an_input_defs_list():
    """Guards the fix across the codebase, not just this loader.

    Parsed rather than grepped.  A regex over source text also matches prose:
    the docstring on ``STLBatchResult.defs_loaded`` explains the bug and
    necessarily contains the literal ``zip(result.nodes, defs)``, which an
    earlier version of this test dutifully reported as a defect.
    """
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parents[2] / "src" / "faceforge"
    offenders = []
    for path in src.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:                       # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "zip"):
                continue
            args = [ast.unparse(a) for a in node.args]
            touches_loader = any(
                a.endswith((".nodes", ".meshes")) for a in args)
            if touches_loader and not any("defs_loaded" in a for a in args):
                # Only a concern when a definition list is the other operand;
                # zipping nodes against meshes is fine.
                if any("def" in a for a in args):
                    offenders.append(f"{path.relative_to(src)}:{node.lineno}")
    assert not offenders, (
        "these zip loader output against something other than defs_loaded:\n  "
        + "\n  ".join(offenders)
    )
