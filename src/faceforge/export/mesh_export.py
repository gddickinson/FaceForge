"""OBJ, PLY and STL export of scene geometry, alongside the GLB exporter.

GLB is the right format for handing a scene to Blender, and the wrong format
for almost everything else a lab does with a mesh: meshlab, CloudCompare,
FreeCAD, a slicer, a finite-element preprocessor and most Python tooling all
want OBJ, PLY or STL.  This module writes all three from the same baked
world-space geometry the GLB exporter uses, so the four formats cannot describe
different shapes.

Provenance is the reason this is not a fifty-line file.  The BodyParts3D
licence obligation travels with the geometry, and the four formats can carry it
to four different depths -- see :data:`faceforge.export.provenance.
PROVENANCE_CHANNELS`.  Rather than write the notice where it fits and stay
quiet where it does not, every export here also writes a
``<name>.provenance.json`` sidecar and reports, in
:attr:`MeshExportResult.provenance_channel`, exactly how much of the
provenance made it into the file itself.

What is *not* claimed
---------------------
* STL carries no per-structure identity.  Binary STL is a flat facet list with
  an 80-byte header; a truncated attribution marker fits there and the full
  string does not.  Exporting a 16-structure scene to STL loses which facet
  belongs to which bone, irretrievably.  The sidecar is the only record.
* OBJ and PLY carry the attribution as a *comment*.  Comments survive a
  round-trip through a text editor and through every reader that preserves
  headers, but no reader is obliged to surface them, and several strip them.
* Only glTF carries provenance the format's own schema knows about.
"""

from __future__ import annotations

import json
import logging
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from faceforge.export.baking import BakedGeometry, bake_world_geometry
from faceforge.export.provenance import (
    BODYPARTS3D_ATTRIBUTION,
    PROVENANCE_CHANNELS,
    StructureProvenance,
    collect_provenance,
    comment_header,
    needs_attribution,
    provenance_document,
    unattributed,
)

logger = logging.getLogger(__name__)

#: Formats :func:`export_mesh` dispatches on.  ``glb`` delegates to
#: :mod:`faceforge.export.glb_exporter` so one call site covers all four.
MESH_FORMATS: tuple[str, ...] = ("obj", "ply", "stl", "glb")

#: Suffix -> format, for inferring the format from the output path.
_SUFFIX_FORMAT = {
    ".obj": "obj", ".ply": "ply", ".stl": "stl",
    ".glb": "glb", ".gltf": "glb",
}

#: What fits in binary STL's 80-byte header.  Deliberately short of the real
#: attribution string: writing a truncated licence notice and calling it the
#: notice would be worse than pointing at the sidecar that has the whole thing.
STL_HEADER_MARKER = b"FaceForge BodyParts3D CC-BY-SA-2.1-JP see .provenance.json"


class MeshExportError(RuntimeError):
    """An export that cannot produce a correct file."""


@dataclass(frozen=True)
class MeshExportResult:
    """What was written, and how much provenance survived."""

    fmt: str
    path: Path
    meshes: int
    vertices: int
    triangles: int
    bytes_written: int
    sidecar: Path | None
    attribution_in_file: str
    per_structure_in_file: str
    attribution_required: bool
    unattributed_indices: tuple[int, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": self.fmt,
            "out": str(self.path),
            "meshes": self.meshes,
            "vertices": self.vertices,
            "triangles": self.triangles,
            "bytes": self.bytes_written,
            "sidecar": None if self.sidecar is None else str(self.sidecar),
            "provenance": {
                "attribution_in_file": self.attribution_in_file,
                "per_structure_in_file": self.per_structure_in_file,
                "attribution_required": self.attribution_required,
                "structures_without_source_id": list(self.unattributed_indices),
            },
            "notes": list(self.notes),
        }

    def summary(self) -> str:
        return (
            f"{self.fmt}: {self.meshes} mesh(es), {self.vertices} vertices, "
            f"{self.triangles} triangles, {self.bytes_written / 1e6:.2f} MB; "
            f"attribution in file: {self.attribution_in_file}, "
            f"per-structure: {self.per_structure_in_file}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def format_for_path(path: Path | str) -> str:
    """Infer the export format from a filename suffix."""
    suffix = Path(path).suffix.lower()
    try:
        return _SUFFIX_FORMAT[suffix]
    except KeyError:
        raise MeshExportError(
            f"cannot tell what format {Path(path).name!r} should be; known "
            f"suffixes: {sorted(_SUFFIX_FORMAT)}"
        ) from None


def export_mesh(
    scene: Any,
    path: Path | str,
    fmt: str | None = None,
    *,
    sidecar: bool = True,
) -> MeshExportResult:
    """Export every visible mesh in *scene* to *path*.

    *fmt* defaults to the format implied by the suffix.  World transforms are
    baked in, so the file is in scene coordinates with the hierarchy flattened.

    Raises :class:`MeshExportError` rather than writing an empty file: a
    zero-mesh export that exits successfully is how a broken figure pipeline
    stays broken for a week.
    """
    path = Path(path)
    fmt = (fmt or format_for_path(path)).lower()
    if fmt not in MESH_FORMATS:
        raise MeshExportError(
            f"unknown mesh format {fmt!r}; known: {list(MESH_FORMATS)}"
        )

    scene.update()
    mesh_pairs = [
        (mesh, mat) for mesh, mat in scene.collect_meshes()
        if getattr(mesh.geometry, "vertex_count", 0) > 0
    ]
    if not mesh_pairs:
        raise MeshExportError(
            "no visible mesh in the scene has any geometry, so there is "
            "nothing to export.  Refusing to write an empty file."
        )

    if fmt == "glb":
        from faceforge.export.glb_exporter import export_glb

        count = export_glb(scene, path)
        baked = [bake_world_geometry(m, w) for m, w in mesh_pairs]
        records = collect_provenance(mesh_pairs, baked)
        result_notes: tuple[str, ...] = (
            "glTF extras carry per-structure provenance; asset.copyright "
            "carries the attribution.",
        )
        written = count
    else:
        baked = [bake_world_geometry(m, w) for m, w in mesh_pairs]
        records = collect_provenance(mesh_pairs, baked)
        writer = {"obj": _write_obj, "ply": _write_ply, "stl": _write_stl}[fmt]
        result_notes = writer(path, baked, records)
        written = len(baked)

    channel = PROVENANCE_CHANNELS[fmt]
    sidecar_path: Path | None = None
    if sidecar:
        sidecar_path = path.with_suffix(path.suffix + ".provenance.json")
        doc = provenance_document(records, fmt=fmt, target=path.name)
        sidecar_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False),
                                encoding="utf-8")

    missing = unattributed(records)
    if missing:
        logger.warning(
            "%d of %d exported meshes carry no BodyParts3D source_id "
            "(indices %s); they are recorded in the sidecar as unattributed "
            "rather than claimed as BodyParts3D",
            len(missing), len(records), [r.index for r in missing],
        )

    result = MeshExportResult(
        fmt=fmt,
        path=path,
        meshes=written,
        vertices=sum(b.vertex_count for b in baked),
        triangles=sum(b.triangle_count for b in baked),
        bytes_written=path.stat().st_size,
        sidecar=sidecar_path,
        attribution_in_file=channel["attribution"],
        per_structure_in_file=channel["per_structure"],
        attribution_required=needs_attribution(records),
        unattributed_indices=tuple(r.index for r in missing),
        notes=result_notes,
    )
    logger.info("%s", result.summary())
    return result


# ---------------------------------------------------------------------------
# OBJ
# ---------------------------------------------------------------------------


def _write_obj(
    path: Path,
    baked: Sequence[BakedGeometry],
    records: Sequence[StructureProvenance],
) -> tuple[str, ...]:
    """Wavefront OBJ, one ``g``/``o`` group per structure.

    OBJ vertex indices are 1-based and *file*-global, which is the one thing
    that reliably goes wrong when writing a multi-object OBJ by hand: each
    group's faces have to be offset by the number of vertices already written.
    """
    lines: list[str] = comment_header(records, fmt="obj", prefix="#")
    lines.append("#")

    vertex_base = 1
    for geom, rec in zip(baked, records, strict=True):
        group = rec.source_id or f"structure_{rec.index}"
        label = rec.preferred_label or rec.name or group
        lines.append(f"# structure {rec.index}: {rec.one_line()}")
        lines.append(f"o {group}")
        lines.append(f"g {group}")
        if rec.ontology_id:
            # Not a standard OBJ statement, but '#' keeps it legal and a
            # grep-able ontology id in the file is worth more than nothing.
            lines.append(f"# ontology_id {rec.ontology_id}  label {label}")

        for x, y, z in geom.positions.tolist():
            lines.append(f"v {x:.6g} {y:.6g} {z:.6g}")
        for x, y, z in geom.normals.tolist():
            lines.append(f"vn {x:.6g} {y:.6g} {z:.6g}")
        for a, b, c in (geom.indices.astype(np.int64) + vertex_base).tolist():
            lines.append(f"f {a}//{a} {b}//{b} {c}//{c}")
        vertex_base += geom.vertex_count

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return (
        "attribution and the structure table are '#' comments; group names "
        "carry the BodyParts3D source id.",
    )


# ---------------------------------------------------------------------------
# PLY
# ---------------------------------------------------------------------------


def _write_ply(
    path: Path,
    baked: Sequence[BakedGeometry],
    records: Sequence[StructureProvenance],
) -> tuple[str, ...]:
    """Binary little-endian PLY with a custom ``structure_index`` face property.

    PLY has no group concept, so all structures concatenate into one vertex and
    one face element.  The custom per-face ``int structure_index`` is what makes
    that reversible: it maps every facet back to the structure table in the
    header comments (and in the sidecar).  It is a legal PLY property -- readers
    that do not know it skip it, and ones that keep raw element data (trimesh
    does) hand it straight back.
    """
    total_vertices = sum(b.vertex_count for b in baked)
    total_faces = sum(b.triangle_count for b in baked)

    header = ["ply", "format binary_little_endian 1.0"]
    header += comment_header(records, fmt="ply", prefix="comment")
    header += [
        f"element vertex {total_vertices}",
        "property float x", "property float y", "property float z",
        "property float nx", "property float ny", "property float nz",
        f"element face {total_faces}",
        "property list uchar int vertex_indices",
        "property int structure_index",
        "end_header",
    ]

    vertex_dtype = np.dtype([(n, "<f4") for n in ("x", "y", "z", "nx", "ny", "nz")])
    vertices = np.empty(total_vertices, dtype=vertex_dtype)
    face_dtype = np.dtype([
        ("count", "u1"), ("i0", "<i4"), ("i1", "<i4"), ("i2", "<i4"),
        ("structure_index", "<i4"),
    ])
    faces = np.empty(total_faces, dtype=face_dtype)

    v_at = f_at = 0
    for geom, rec in zip(baked, records, strict=True):
        n = geom.vertex_count
        block = vertices[v_at:v_at + n]
        block["x"], block["y"], block["z"] = geom.positions.T
        block["nx"], block["ny"], block["nz"] = geom.normals.T

        m = geom.triangle_count
        fblock = faces[f_at:f_at + m]
        fblock["count"] = 3
        idx = geom.indices.astype(np.int64) + v_at
        fblock["i0"], fblock["i1"], fblock["i2"] = idx.T
        fblock["structure_index"] = rec.index

        v_at += n
        f_at += m

    with open(path, "wb") as fh:
        fh.write(("\n".join(header) + "\n").encode("utf-8"))
        fh.write(vertices.tobytes())
        fh.write(faces.tobytes())

    return (
        "attribution and the structure table are 'comment' header lines; the "
        "per-face 'structure_index' property assigns every facet to a "
        "structure.",
    )


# ---------------------------------------------------------------------------
# STL
# ---------------------------------------------------------------------------


def _write_stl(
    path: Path,
    baked: Sequence[BakedGeometry],
    records: Sequence[StructureProvenance],
) -> tuple[str, ...]:
    """Binary STL: one flat facet soup, structure identity lost.

    This is the honest summary of what STL can do.  The 80-byte header takes a
    marker pointing at the sidecar; the full attribution string is 104 UTF-8
    bytes and does not fit.  There is no per-facet attribute in the format (the
    2-byte "attribute byte count" is required to be zero and is not a
    general-purpose field), so nothing distinguishes a mandible facet from a
    maxilla facet once written.
    """
    triangles = sum(b.triangle_count for b in baked)

    header = bytearray(80)
    header[:len(STL_HEADER_MARKER)] = STL_HEADER_MARKER

    record_dtype = np.dtype([
        ("normal", "<f4", (3,)),
        ("v0", "<f4", (3,)), ("v1", "<f4", (3,)), ("v2", "<f4", (3,)),
        ("attr", "<u2"),
    ])
    facets = np.zeros(triangles, dtype=record_dtype)

    at = 0
    for geom in baked:
        m = geom.triangle_count
        v0, v1, v2 = geom.triangle_vertices()
        block = facets[at:at + m]
        block["normal"] = geom.face_normals()
        block["v0"], block["v1"], block["v2"] = v0, v1, v2
        at += m

    with open(path, "wb") as fh:
        fh.write(bytes(header))
        fh.write(struct.pack("<I", triangles))
        fh.write(facets.tobytes())

    lost = len(records)
    return (
        f"STL cannot carry per-structure provenance: the {lost} structures are "
        "written as one facet soup and their identity is only in the sidecar. "
        f"The 80-byte header holds a {len(STL_HEADER_MARKER)}-byte marker; the "
        f"{len(BODYPARTS3D_ATTRIBUTION.encode('utf-8'))}-byte attribution "
        "string does not fit.",
    )
