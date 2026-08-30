"""OBJ / PLY / STL / GLB round-tripped through an independent reader.

Every geometry claim here is checked by reading the written file back with
**trimesh**, not by inspecting the bytes FaceForge just wrote.  That is the
point: a hand-rolled writer that both writes and validates its own format
proves nothing, because a misunderstanding of the format is present on both
sides.  trimesh has its own OBJ, PLY, STL and GLB parsers and no knowledge of
FaceForge.

The one thing trimesh cannot check is the glTF ``asset`` block -- its loader
does not surface ``asset.copyright`` or ``asset.extras``.  That is verified by
parsing the GLB container's JSON chunk directly with :mod:`json`, which is
still an independent read of the file (the container layout is fixed by the
glTF 2.0 specification and the chunk is plain JSON), just not through trimesh.
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from faceforge.export import mesh_export as me            # noqa: E402
from faceforge.export.provenance import BODYPARTS3D_ATTRIBUTION  # noqa: E402

#: The synthetic scene: two boxes, 24 vertices and 12 triangles each.
EXPECTED_VERTICES = 48
EXPECTED_TRIANGLES = 24


def _read_glb_json(path) -> dict:
    """Parse a GLB's JSON chunk, per the glTF 2.0 container layout."""
    blob = path.read_bytes()
    magic, version, total = struct.unpack("<4sII", blob[:12])
    assert magic == b"glTF" and version == 2
    assert total == len(blob)
    chunk_len, chunk_type = struct.unpack("<II", blob[12:20])
    assert chunk_type == 0x4E4F534A, "first GLB chunk must be JSON"
    return json.loads(blob[20:20 + chunk_len])


# ---------------------------------------------------------------------------
# Geometry survives the round-trip, in all four formats
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["obj", "ply", "stl", "glb"])
def test_geometry_round_trips_through_trimesh(synthetic_scene, tmp_path, fmt):
    out = tmp_path / f"scene.{fmt}"
    result = me.export_mesh(synthetic_scene, out, fmt)

    assert result.vertices == EXPECTED_VERTICES
    assert result.triangles == EXPECTED_TRIANGLES
    assert out.stat().st_size > 0

    loaded = trimesh.load(out, force="mesh")
    assert int(loaded.faces.shape[0]) == EXPECTED_TRIANGLES, (
        f"{fmt}: trimesh read {loaded.faces.shape[0]} faces, expected "
        f"{EXPECTED_TRIANGLES}"
    )
    # Vertex counts legitimately differ between formats: STL is a facet soup
    # (3 vertices per facet, 72) and trimesh merges duplicates on load, so the
    # invariant that holds everywhere is the *bounding box* of the geometry.
    expected_min = np.array([-70.0, -20.0, -20.0])
    expected_max = np.array([70.0, 20.0, 20.0])
    assert np.allclose(loaded.bounds[0], expected_min, atol=1e-3), fmt
    assert np.allclose(loaded.bounds[1], expected_max, atol=1e-3), fmt


@pytest.mark.parametrize("fmt", ["obj", "ply", "stl", "glb"])
def test_every_format_writes_a_provenance_sidecar(synthetic_scene, tmp_path, fmt):
    out = tmp_path / f"scene.{fmt}"
    result = me.export_mesh(synthetic_scene, out, fmt)

    assert result.sidecar is not None and result.sidecar.is_file()
    doc = json.loads(result.sidecar.read_text(encoding="utf-8"))
    assert doc["attribution"] == BODYPARTS3D_ATTRIBUTION
    assert doc["structure_count"] == 2
    assert {s["ontology_id"] for s in doc["structures"]} == {
        "FMA:52748", "FMA:7088"}


def test_format_is_inferred_from_the_suffix(synthetic_scene, tmp_path):
    result = me.export_mesh(synthetic_scene, tmp_path / "scene.ply")
    assert result.fmt == "ply"
    with pytest.raises(me.MeshExportError, match="cannot tell what format"):
        me.export_mesh(synthetic_scene, tmp_path / "scene.wat")


def test_an_empty_scene_is_refused_rather_than_written(tmp_path):
    from faceforge.core.scene_graph import Scene

    with pytest.raises(me.MeshExportError, match="nothing to export"):
        me.export_mesh(Scene(), tmp_path / "empty.obj")
    assert not (tmp_path / "empty.obj").exists()


# ---------------------------------------------------------------------------
# glTF provenance: the format that can carry it properly
# ---------------------------------------------------------------------------


def test_glb_asset_block_carries_the_required_attribution(
    synthetic_scene, tmp_path,
):
    out = tmp_path / "scene.glb"
    me.export_mesh(synthetic_scene, out, "glb")
    doc = _read_glb_json(out)

    assert doc["asset"]["copyright"] == BODYPARTS3D_ATTRIBUTION
    extras = doc["asset"]["extras"]
    assert extras["attribution"] == BODYPARTS3D_ATTRIBUTION
    assert extras["dataset"] == "BodyParts3D"
    assert extras["licence"] == "CC BY-SA 2.1 JP"
    assert extras["licence_url"].startswith("https://creativecommons.org/")


def test_glb_nodes_and_meshes_carry_fma_ids_and_labels(synthetic_scene, tmp_path):
    out = tmp_path / "scene.glb"
    me.export_mesh(synthetic_scene, out, "glb")
    doc = _read_glb_json(out)

    for collection in ("nodes", "meshes"):
        found = {}
        for entry in doc[collection]:
            extras = entry["extras"]
            found[extras["source_id"]] = extras
        assert set(found) == {"FMA52748", "FMA7088"}, collection
        assert found["FMA52748"]["ontology_id"] == "FMA:52748"
        assert found["FMA52748"]["preferred_label"] == "Mandible"
        assert found["FMA7088"]["ontology_id"] == "FMA:7088"


def test_trimesh_surfaces_the_glb_mesh_extras(synthetic_scene, tmp_path):
    """An independent reader must be able to *retrieve* the provenance.

    Attribution buried where no library will hand it back is not much better
    than no attribution.  trimesh's glTF loader copies mesh-level ``extras``
    into ``geometry[...].metadata``, so this is the check that the provenance is
    reachable through somebody else's parser.
    """
    out = tmp_path / "scene.glb"
    me.export_mesh(synthetic_scene, out, "glb")

    scene = trimesh.load(out, force="scene")
    ontology = set()
    labels = set()
    for geom in scene.geometry.values():
        assert geom.metadata.get("dataset") == "BodyParts3D"
        ontology.add(geom.metadata.get("ontology_id"))
        labels.add(geom.metadata.get("preferred_label"))
    assert ontology == {"FMA:52748", "FMA:7088"}
    assert labels == {"Mandible", "Heart"}


def test_glb_marks_a_mesh_with_no_source_id_as_not_bodyparts3d(
    mixed_scene, tmp_path,
):
    out = tmp_path / "mixed.glb"
    result = me.export_mesh(mixed_scene, out, "glb")
    assert result.unattributed_indices == (2,)

    doc = _read_glb_json(out)
    orphan = [m for m in doc["meshes"] if m["name"] == "procedural_marker"]
    assert len(orphan) == 1
    extras = orphan[0]["extras"]
    assert "source_id" not in extras
    assert "not from BodyParts3D" in extras["provenance"]


# ---------------------------------------------------------------------------
# OBJ: groups carry the source id, comments carry the notice
# ---------------------------------------------------------------------------


def test_obj_groups_name_the_bodyparts3d_source_ids(synthetic_scene, tmp_path):
    out = tmp_path / "scene.obj"
    me.export_mesh(synthetic_scene, out, "obj")

    # trimesh merges OBJ objects by default; split_objects=True makes it honour
    # the 'o' statements, which is how an independent reader recovers which
    # facets belong to which BodyParts3D structure.
    scene = trimesh.load(out, force="scene", split_objects=True)
    assert set(scene.geometry) == {"FMA52748", "FMA7088"}, (
        f"trimesh reported geometry names {list(scene.geometry)}"
    )
    assert {k: len(v.faces) for k, v in scene.geometry.items()} == {
        "FMA52748": 12, "FMA7088": 12,
    }


def test_obj_header_comments_hold_the_attribution(synthetic_scene, tmp_path):
    out = tmp_path / "scene.obj"
    result = me.export_mesh(synthetic_scene, out, "obj")
    text = out.read_text(encoding="utf-8")

    assert BODYPARTS3D_ATTRIBUTION in text
    assert "# ontology_id FMA:52748" in text
    assert result.attribution_in_file == "comment", (
        "OBJ cannot do better than a comment; do not claim it can"
    )


# ---------------------------------------------------------------------------
# PLY: a custom face property, read back by an independent parser
# ---------------------------------------------------------------------------


def test_ply_face_property_maps_every_facet_to_its_structure(
    synthetic_scene, tmp_path,
):
    out = tmp_path / "scene.ply"
    me.export_mesh(synthetic_scene, out, "ply")

    loaded = trimesh.load(out, force="mesh")
    raw = loaded.metadata["_ply_raw"]
    assert "structure_index" in raw["face"]["properties"], (
        "trimesh did not surface the custom face property"
    )
    structure = np.asarray(raw["face"]["data"]["structure_index"])
    assert structure.shape == (EXPECTED_TRIANGLES,)
    counts = np.bincount(structure, minlength=2)
    assert counts.tolist() == [12, 12], (
        "each box should own 12 facets in the concatenated PLY"
    )


def test_ply_comments_hold_the_attribution(synthetic_scene, tmp_path):
    out = tmp_path / "scene.ply"
    me.export_mesh(synthetic_scene, out, "ply")
    header = out.read_bytes().split(b"end_header")[0].decode("utf-8")

    assert BODYPARTS3D_ATTRIBUTION in header
    assert "comment " in header
    assert "FMA:52748" in header


def test_ply_normals_survive_the_round_trip(synthetic_scene, tmp_path):
    """A box's face normals are known exactly, so this is checkable."""
    out = tmp_path / "scene.ply"
    me.export_mesh(synthetic_scene, out, "ply")

    loaded = trimesh.load(out, force="mesh", process=False)
    normals = np.asarray(loaded.vertex_normals)
    # Every vertex of an axis-aligned box has a unit normal along one axis.
    lengths = np.linalg.norm(normals, axis=1)
    assert np.allclose(lengths, 1.0, atol=1e-4)
    axis_aligned = (np.abs(np.abs(normals) - 1.0) < 1e-4).sum(axis=1)
    assert (axis_aligned == 1).all()


# ---------------------------------------------------------------------------
# STL: what is lost, stated rather than glossed
# ---------------------------------------------------------------------------


def test_stl_loses_per_structure_identity_and_says_so(synthetic_scene, tmp_path):
    out = tmp_path / "scene.stl"
    result = me.export_mesh(synthetic_scene, out, "stl")

    assert result.per_structure_in_file == "none"
    assert any("cannot carry per-structure provenance" in n for n in result.notes)

    loaded = trimesh.load(out, force="mesh")
    assert len(loaded.faces) == EXPECTED_TRIANGLES
    # The geometry is all there; the identity is not.  Two separated boxes read
    # back as two disconnected bodies, and nothing in the file says which is
    # the mandible.  (trimesh fills metadata['name'] from the *filename*, which
    # is not provenance.)
    assert loaded.body_count == 2
    metadata_text = " ".join(str(v) for v in loaded.metadata.values())
    assert "FMA" not in metadata_text, (
        "an STL round-trip must not appear to preserve ontology ids"
    )
    assert "Mandible" not in metadata_text and "Heart" not in metadata_text


def test_stl_header_holds_the_marker_and_the_facet_count_is_correct(
    synthetic_scene, tmp_path,
):
    out = tmp_path / "scene.stl"
    me.export_mesh(synthetic_scene, out, "stl")
    blob = out.read_bytes()

    assert blob[:len(me.STL_HEADER_MARKER)] == me.STL_HEADER_MARKER
    count = struct.unpack("<I", blob[80:84])[0]
    assert count == EXPECTED_TRIANGLES
    assert len(blob) == 84 + count * 50, "binary STL is 84 + 50 bytes per facet"


def test_stl_facet_normals_point_outward(synthetic_scene, tmp_path):
    """STL stores a facet normal; a wrong sign flips shading in every reader."""
    out = tmp_path / "scene.stl"
    me.export_mesh(synthetic_scene, out, "stl")

    loaded = trimesh.load(out, force="mesh", process=False)
    centres = loaded.triangles.mean(axis=1)
    # Each facet belongs to one of two boxes; the outward direction is away
    # from that box's centre.
    box_centre = np.where(centres[:, :1] < 0, -50.0, 50.0)
    outward = centres - np.hstack([box_centre, np.zeros((len(centres), 2))])
    dots = (loaded.face_normals * outward).sum(axis=1)
    assert (dots > 0).all(), "some STL facet normals point into the solid"


# ---------------------------------------------------------------------------
# The four formats must agree with each other
# ---------------------------------------------------------------------------


def test_all_four_formats_describe_the_same_geometry(synthetic_scene, tmp_path):
    """The shared baking path exists so this is true; assert it, not trust it."""
    volumes, areas = {}, {}
    for fmt in ("obj", "ply", "stl", "glb"):
        out = tmp_path / f"scene.{fmt}"
        me.export_mesh(synthetic_scene, out, fmt)
        loaded = trimesh.load(out, force="mesh")
        volumes[fmt] = float(loaded.volume)
        areas[fmt] = float(loaded.area)

    # Two 40 mm boxes: 2 * 40^3 = 128000 mm^3, 2 * 6 * 40^2 = 19200 mm^2.
    for fmt, value in volumes.items():
        assert value == pytest.approx(128000.0, rel=1e-4), fmt
    for fmt, value in areas.items():
        assert value == pytest.approx(19200.0, rel=1e-4), fmt
