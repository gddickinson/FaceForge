"""Attribution and per-structure provenance for everything FaceForge exports.

The geometry FaceForge renders is BodyParts3D, and BodyParts3D is CC BY-SA 2.1
Japan.  The share-alike clause travels with the geometry: a GLB handed to a
collaborator, an OBJ dropped into a figure pipeline, a DICOM series loaded into
a teaching PACS all carry the obligation, and an export that drops the notice
is a licence breach, not a cosmetic omission.  So the attribution string lives
here, once, and every exporter in :mod:`faceforge.export` writes it.

What each format can actually carry is *not* uniform, and this module does not
pretend otherwise -- :data:`PROVENANCE_CHANNELS` records, per format, whether
provenance is structured, comment-only, or impossible.  Where a format cannot
carry it, the exporters write a ``.provenance.json`` sidecar and say so in
their result object.  A sidecar next to the file is a weaker guarantee than a
notice inside it; claiming otherwise would be worse than the sidecar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

#: The attribution the BodyParts3D licence requires, verbatim.  Do not
#: reflow, abbreviate or "modernise" this string: it is a licence condition,
#: and tests/export/test_provenance.py asserts it byte-for-byte.
BODYPARTS3D_ATTRIBUTION = (
    "BodyParts3D, © The Database Center for Life Science licensed under "
    "CC Attribution-Share Alike 2.1 Japan"
)

#: The licence itself, for formats with somewhere to put a URL.
BODYPARTS3D_LICENCE_URL = "https://creativecommons.org/licenses/by-sa/2.1/jp/"

#: What FaceForge writes as the producing tool.
GENERATOR = "FaceForge anatomical viewer (faceforge.export)"

#: Per-format honesty table.  ``structured`` means a reader can retrieve the
#: provenance through the format's own data model; ``comment`` means it is in
#: the file but only as free text a reader is not obliged to surface; ``none``
#: means the format has nowhere to put it and the sidecar is the only copy.
PROVENANCE_CHANNELS: dict[str, dict[str, str]] = {
    "glb": {
        "attribution": "structured",
        "per_structure": "structured",
        "detail": "asset.copyright plus asset.extras, and mesh/node extras "
                  "per structure (glTF 2.0 extras are part of the schema).",
    },
    "obj": {
        "attribution": "comment",
        "per_structure": "structured",
        "detail": "'#' comment header carries the attribution and a "
                  "structure table; 'g'/'o' group names carry the "
                  "BodyParts3D source id, which OBJ readers do surface.",
    },
    "ply": {
        "attribution": "comment",
        "per_structure": "structured",
        "detail": "'comment' header lines carry the attribution and a "
                  "structure table; a custom int face property "
                  "'structure_index' assigns every facet to a structure.",
    },
    "stl": {
        "attribution": "comment",
        "per_structure": "none",
        "detail": "binary STL has an 80-byte header and nothing else: a "
                  "truncated attribution marker fits, the full string does "
                  "not, and STL has no concept of a named part, so "
                  "per-structure provenance is impossible in-file.",
    },
    "dicom": {
        "attribution": "structured",
        "per_structure": "comment",
        "detail": "ImageComments (0020,4000) and DerivationDescription "
                  "(0008,2111) carry the attribution and the SIMULATED "
                  "statement; the contributing structure list goes in the "
                  "sidecar, since a voxel grid has no per-structure element.",
    },
    "nifti": {
        "attribution": "structured",
        "per_structure": "comment",
        "detail": "a NIfTI-1 comment extension (ecode 6) carries the "
                  "attribution and the SIMULATED statement; nibabel reads it "
                  "back.  descrip is 80 bytes and only gets a short marker.",
    },
}


@dataclass(frozen=True)
class StructureProvenance:
    """Where one exported mesh came from.

    ``source_id`` is the BodyParts3D file stem (``FMA52748``), ``ontology_id``
    the canonical term (``FMA:52748``) and ``preferred_label`` the FMA
    preferred term.  All three can legitimately be empty: a scene may hold
    meshes FaceForge generated rather than loaded (the scan-plane quad, a
    procedural primitive in a test), and those are not BodyParts3D and must not
    be labelled as though they were.
    """

    index: int
    name: str
    source_id: str
    ontology_id: str
    preferred_label: str
    vertex_count: int
    triangle_count: int

    @property
    def is_bodyparts3d(self) -> bool:
        """True when this mesh came out of the BodyParts3D distribution."""
        return bool(self.source_id)

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "source_id": self.source_id,
            "ontology_id": self.ontology_id,
            "preferred_label": self.preferred_label,
            "vertex_count": self.vertex_count,
            "triangle_count": self.triangle_count,
        }

    def extras(self) -> dict[str, Any]:
        """The glTF ``extras`` payload for this structure.

        Empty fields are omitted rather than written as ``""``: a consumer
        reading ``extras["ontology_id"] == ""`` cannot tell "not from an
        ontology" from "we lost it", and the difference matters.
        """
        out: dict[str, Any] = {"faceforge_index": self.index}
        if self.source_id:
            out["source_id"] = self.source_id
        if self.ontology_id:
            out["ontology_id"] = self.ontology_id
        if self.preferred_label:
            out["preferred_label"] = self.preferred_label
        if self.source_id:
            out["dataset"] = "BodyParts3D"
            out["licence"] = "CC BY-SA 2.1 JP"
        return out

    def one_line(self) -> str:
        """A single comment line for OBJ/PLY headers."""
        parts = [f"[{self.index}]"]
        if self.source_id:
            parts.append(self.source_id)
        parts.append(self.preferred_label or self.name or "(unnamed)")
        if self.ontology_id:
            parts.append(f"<{self.ontology_id}>")
        parts.append(f"v={self.vertex_count} f={self.triangle_count}")
        return " ".join(parts)


def collect_provenance(
    mesh_pairs: Sequence[tuple[Any, Any]],
    baked: Sequence[Any] | None = None,
) -> list[StructureProvenance]:
    """Build one :class:`StructureProvenance` per mesh, in export order.

    *baked* is the matching sequence of :class:`~faceforge.export.baking.
    BakedGeometry`; when given, the counts come from the baked arrays (what is
    actually in the file) rather than from the source geometry.
    """
    records: list[StructureProvenance] = []
    for i, (mesh, _mat) in enumerate(mesh_pairs):
        geom = mesh.geometry
        if baked is not None:
            vertices = baked[i].vertex_count
            triangles = baked[i].triangle_count
        else:
            vertices = int(getattr(geom, "vertex_count", 0))
            triangles = int(getattr(geom, "triangle_count", 0))
        records.append(StructureProvenance(
            index=i,
            name=str(getattr(mesh, "name", "") or ""),
            source_id=str(getattr(mesh, "source_id", "") or ""),
            ontology_id=str(getattr(mesh, "ontology_id", "") or ""),
            preferred_label=str(getattr(mesh, "preferred_label", "") or ""),
            vertex_count=vertices,
            triangle_count=triangles,
        ))
    return records


def needs_attribution(records: Iterable[StructureProvenance]) -> bool:
    """True if any exported mesh is BodyParts3D, so the notice is required."""
    return any(r.is_bodyparts3d for r in records)


def unattributed(records: Iterable[StructureProvenance]) -> list[StructureProvenance]:
    """Meshes carrying no ``source_id`` -- reported, never silently dropped."""
    return [r for r in records if not r.is_bodyparts3d]


def asset_extras(fmt: str) -> dict[str, Any]:
    """The file-level provenance block, shared by every format that has one."""
    return {
        "generator": GENERATOR,
        "dataset": "BodyParts3D",
        "attribution": BODYPARTS3D_ATTRIBUTION,
        "licence": "CC BY-SA 2.1 JP",
        "licence_url": BODYPARTS3D_LICENCE_URL,
        "provenance_channel": PROVENANCE_CHANNELS.get(fmt, {}).get(
            "per_structure", "unknown"),
    }


def provenance_document(
    records: Sequence[StructureProvenance],
    *,
    fmt: str,
    target: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The ``.provenance.json`` sidecar contents.

    Written for *every* format, not only the ones that cannot carry provenance
    in-file: a machine-readable copy next to the geometry costs a few kilobytes
    and is the only thing a downstream script can rely on across four formats
    with four different capabilities.
    """
    channel = PROVENANCE_CHANNELS.get(fmt, {})
    doc: dict[str, Any] = {
        "generator": GENERATOR,
        "format": fmt,
        "target": target,
        "attribution": BODYPARTS3D_ATTRIBUTION,
        "attribution_required": needs_attribution(records),
        "licence": "CC BY-SA 2.1 JP",
        "licence_url": BODYPARTS3D_LICENCE_URL,
        "in_file_provenance": {
            "attribution": channel.get("attribution", "unknown"),
            "per_structure": channel.get("per_structure", "unknown"),
            "detail": channel.get("detail", ""),
        },
        "structures": [r.as_dict() for r in records],
        "structure_count": len(records),
        "bodyparts3d_structures": sum(1 for r in records if r.is_bodyparts3d),
        "structures_without_source_id": [
            r.index for r in records if not r.is_bodyparts3d
        ],
    }
    if extra:
        doc.update(extra)
    return doc


def comment_header(
    records: Sequence[StructureProvenance],
    *,
    fmt: str,
    prefix: str,
) -> list[str]:
    """Attribution + structure table as comment lines, ``prefix``-marked.

    ``prefix`` is ``"#"`` for OBJ and ``"comment"`` for PLY -- the two formats
    whose only attribution channel is a header comment.
    """
    lines = [
        f"{prefix} {GENERATOR}",
        f"{prefix} {BODYPARTS3D_ATTRIBUTION}",
        f"{prefix} licence: {BODYPARTS3D_LICENCE_URL}",
        f"{prefix} format: {fmt}; per-structure provenance: "
        f"{PROVENANCE_CHANNELS.get(fmt, {}).get('per_structure', 'unknown')}",
        f"{prefix} structures: {len(records)}",
    ]
    lines += [f"{prefix} {r.one_line()}" for r in records]
    return lines
