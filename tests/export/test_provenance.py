"""The attribution string and the per-structure provenance records.

The first test in this module is the one that matters legally: the BodyParts3D
attribution is a licence condition, and it is asserted byte-for-byte so that a
well-meaning reflow, a curly-quote autocorrect or a "modernised" copyright sign
fails here rather than in someone's redistribution.
"""

from __future__ import annotations

import json

from faceforge.export import provenance as prov


def test_attribution_string_is_exact():
    """Byte-for-byte. Do not "fix" this string; fix whatever changed it."""
    assert prov.BODYPARTS3D_ATTRIBUTION == (
        "BodyParts3D, © The Database Center for Life Science licensed under "
        "CC Attribution-Share Alike 2.1 Japan"
    )
    # The STL exporter's docstring quotes this length as its reason for using a
    # marker instead of the full string; pin it so the two cannot disagree.
    assert len(prov.BODYPARTS3D_ATTRIBUTION) == 103
    assert len(prov.BODYPARTS3D_ATTRIBUTION.encode("utf-8")) == 104
    assert "©" in prov.BODYPARTS3D_ATTRIBUTION
    assert "CC Attribution-Share Alike 2.1 Japan" in prov.BODYPARTS3D_ATTRIBUTION


def test_stl_header_marker_fits_in_eighty_bytes():
    """Binary STL's header is exactly 80 bytes and the marker must fit it."""
    from faceforge.export.mesh_export import STL_HEADER_MARKER

    assert len(STL_HEADER_MARKER) <= 80
    assert len(prov.BODYPARTS3D_ATTRIBUTION.encode("utf-8")) > 80, (
        "if the attribution ever fits in 80 bytes, write it there instead of "
        "the marker"
    )
    assert not STL_HEADER_MARKER.startswith(b"solid"), (
        "an STL header starting with 'solid' makes readers treat a binary "
        "file as ASCII"
    )


def test_every_format_declares_its_provenance_capability():
    """No format may be silently absent from the honesty table."""
    from faceforge.export.mesh_export import MESH_FORMATS

    for fmt in (*MESH_FORMATS, "dicom", "nifti"):
        assert fmt in prov.PROVENANCE_CHANNELS, fmt
        channel = prov.PROVENANCE_CHANNELS[fmt]
        assert channel["attribution"] in ("structured", "comment", "none")
        assert channel["per_structure"] in ("structured", "comment", "none")
        assert channel["detail"]

    # The one claim that must stay pessimistic.
    assert prov.PROVENANCE_CHANNELS["stl"]["per_structure"] == "none"
    assert prov.PROVENANCE_CHANNELS["glb"]["per_structure"] == "structured"


def test_collect_provenance_reads_the_mesh_fields(synthetic_scene):
    records = prov.collect_provenance(synthetic_scene.collect_meshes())
    assert len(records) == 2
    by_source = {r.source_id: r for r in records}
    assert set(by_source) == {"FMA52748", "FMA7088"}

    mandible = by_source["FMA52748"]
    assert mandible.ontology_id == "FMA:52748"
    assert mandible.preferred_label == "Mandible"
    assert mandible.vertex_count == 24
    assert mandible.triangle_count == 12
    assert mandible.is_bodyparts3d

    extras = mandible.extras()
    assert extras["source_id"] == "FMA52748"
    assert extras["ontology_id"] == "FMA:52748"
    assert extras["dataset"] == "BodyParts3D"


def test_a_mesh_without_a_source_id_is_not_claimed_as_bodyparts3d(mixed_scene):
    records = prov.collect_provenance(mixed_scene.collect_meshes())
    assert len(records) == 3
    orphans = prov.unattributed(records)
    assert [r.name for r in orphans] == ["procedural_marker"]

    extras = orphans[0].extras()
    assert "source_id" not in extras
    assert "ontology_id" not in extras
    assert "dataset" not in extras, (
        "a mesh with no source_id must not be labelled BodyParts3D"
    )
    # ...but the file still needs the notice, because the other two are.
    assert prov.needs_attribution(records) is True


def test_provenance_document_is_json_serialisable_and_complete(mixed_scene):
    records = prov.collect_provenance(mixed_scene.collect_meshes())
    doc = prov.provenance_document(records, fmt="stl", target="x.stl")
    reloaded = json.loads(json.dumps(doc, ensure_ascii=False))

    assert reloaded["attribution"] == prov.BODYPARTS3D_ATTRIBUTION
    assert reloaded["attribution_required"] is True
    assert reloaded["structure_count"] == 3
    assert reloaded["bodyparts3d_structures"] == 2
    assert reloaded["structures_without_source_id"] == [2]
    assert reloaded["in_file_provenance"]["per_structure"] == "none"


def test_comment_header_carries_attribution_and_one_line_per_structure(
    synthetic_scene,
):
    records = prov.collect_provenance(synthetic_scene.collect_meshes())
    lines = prov.comment_header(records, fmt="obj", prefix="#")
    assert all(line.startswith("# ") for line in lines)
    assert any(prov.BODYPARTS3D_ATTRIBUTION in line for line in lines)
    assert sum("FMA:52748" in line for line in lines) == 1
    assert sum("FMA:7088" in line for line in lines) == 1
