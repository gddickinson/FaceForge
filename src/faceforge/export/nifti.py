"""NIfTI-1 export of virtual-scanner volumes.

NIfTI is the format the analysis half of the world reads: FSL, SPM, ANTs,
FreeSurfer, nilearn, 3D Slicer.  What it needs from an exporter is one thing
above all others -- a correct affine.  A NIfTI whose affine is wrong loads
without complaint, displays convincingly, and gives wrong distances and a
flipped left-right forever after.

The affine written here maps voxel ``(i, j, k)`` to RAS millimetres, built from
the scanner's own ray geometry in :mod:`faceforge.export.volume`:

* ``i`` is the column index, along the scan plane's *right* vector;
* ``j`` is the row index, along the plane's *down* vector (``-up``, because row
  0 is the top of the image);
* ``k`` is the slice index, along ``row x col`` so the frame is right-handed.

Scene coordinates are BodyParts3D coordinates, which are LPS (measured -- see
:mod:`faceforge.export.volume`), and NIfTI's world frame is RAS, so the x and y
axes flip sign.  ``tests/export/test_nifti.py`` reloads the file with nibabel
and checks the affine against independently computed voxel positions, and
checks that ``nibabel.aff2axcodes`` reports the axis labels the geometry
implies.

Values, and honesty about them
------------------------------
Unlike DICOM, NIfTI has no rescale-type field to declare a unit, so the values
are written as they are and the declaration goes in the header extension:

* ``hu_mode="index"`` (default) writes float32 in ``0..1`` -- the scanner's own
  dimensionless tissue index.  Not Hounsfield units; see
  :mod:`faceforge.export.hounsfield`.
* ``hu_mode="class"`` writes int16 nominal Hounsfield units, available only for
  ``mode="ct"`` with ``reduction="max"``, with the same limitations documented
  there.

``descrip`` is 80 bytes and only gets a short marker.  The full statement --
simulation notice, BodyParts3D attribution, unit description, geometry -- goes
into a NIfTI-1 comment extension (ecode 6) as JSON, which nibabel reads back,
so it is machine-readable rather than a comment nobody can retrieve.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from faceforge.export.dicom import SIMULATION_STATEMENT
from faceforge.export.hounsfield import RescaleSpec, to_hounsfield
from faceforge.export.provenance import (
    BODYPARTS3D_ATTRIBUTION,
    GENERATOR,
    provenance_document,
)
from faceforge.export.volume import ScanVolume

logger = logging.getLogger(__name__)

#: NIfTI-1 extension code for a plain comment.  nibabel exposes extensions on
#: ``img.header.extensions``, so this survives a round-trip.
NIFTI_COMMENT_ECODE = 6

#: ``descrip`` is a fixed 80-byte field.  Anything longer is truncated by the
#: format, so a short marker goes here and the real statement in the extension.
_DESCRIP_LIMIT = 80


class NiftiExportError(RuntimeError):
    """A NIfTI volume that cannot be written correctly."""


@dataclass(frozen=True)
class NiftiResult:
    """What was written, and how it describes itself."""

    path: Path
    shape: tuple[int, int, int]
    dtype: str
    affine: np.ndarray
    axis_codes: tuple[str, str, str]
    rescale: RescaleSpec
    sidecar: Path | None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "out": str(self.path),
            "shape_i_j_k": list(self.shape),
            "dtype": self.dtype,
            "affine": [[float(v) for v in row] for row in self.affine],
            "axis_codes": list(self.axis_codes),
            "voxel_sizes_mm": [
                float(np.linalg.norm(self.affine[:3, c])) for c in range(3)],
            "pixel_values": self.rescale.as_dict(),
            "sidecar": None if self.sidecar is None else str(self.sidecar),
            "simulated": True,
            "notes": list(self.notes),
        }


def header_document(
    volume: ScanVolume,
    spec: RescaleSpec,
    affine: np.ndarray,
) -> dict[str, Any]:
    """The JSON payload written into the NIfTI comment extension."""
    return {
        "generator": GENERATOR,
        "simulated": True,
        "simulation": SIMULATION_STATEMENT,
        "attribution": BODYPARTS3D_ATTRIBUTION,
        "licence": "CC BY-SA 2.1 JP",
        "unit": spec.unit,
        "hu_mode": spec.mode,
        "value_description": spec.description,
        "value_notes": list(spec.notes),
        "scan": {"mode": volume.mode, "reduction": volume.reduction},
        "affine_voxel_to_ras_mm": [[float(v) for v in row] for row in affine],
        "geometry": volume.geometry.as_dict(),
    }


def export_nifti(
    volume: ScanVolume,
    path: Path | str,
    *,
    hu_mode: str = "index",
    provenance: Any = None,
    sidecar: bool = True,
) -> NiftiResult:
    """Write *volume* as a NIfTI-1 file (``.nii`` or ``.nii.gz``).

    ``hu_mode="index"`` writes the model's own float32 values in ``0..1``;
    ``hu_mode="class"`` writes int16 nominal HU and is only accepted for a CT
    scan reduced with ``max`` (see :mod:`faceforge.export.hounsfield`).
    """
    try:
        import nibabel as nib
        from nibabel.nifti1 import Nifti1Extension, Nifti1Image
    except ImportError as exc:                           # pragma: no cover
        raise NiftiExportError(
            "NIfTI export needs nibabel: pip install nibabel"
        ) from exc

    path = Path(path)
    geometry = volume.geometry
    spec = to_hounsfield(
        volume.data, hu_mode=hu_mode, mode=volume.mode,
        reduction=volume.reduction,
    )

    # ``spec.stored`` is (slice, row, col); NIfTI wants (i, j, k) = (col, row,
    # slice) to match the affine.  Reuse ScanVolume's transpose so the two
    # cannot disagree.
    if spec.mode == "class":
        real = spec.real_values().astype(np.int16)
        data = np.ascontiguousarray(np.transpose(real, (2, 1, 0)))
    else:
        data = volume.nifti_array().astype(np.float32)

    affine = geometry.nifti_affine()
    image = Nifti1Image(data, affine)
    header = image.header
    header.set_xyzt_units(xyz="mm")
    header["descrip"] = (
        f"FaceForge SIMULATED {volume.mode} {spec.mode}"
    ).encode("ascii")[:_DESCRIP_LIMIT]
    # scl_slope/inter are identity: the array already holds the real values, so
    # a reader that ignores them still gets the right numbers.
    header.set_slope_inter(1.0, 0.0)
    header.extensions.append(Nifti1Extension(
        NIFTI_COMMENT_ECODE,
        json.dumps(header_document(volume, spec, affine),
                   ensure_ascii=False).encode("utf-8"),
    ))

    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, path)

    axis_codes = tuple(nib.aff2axcodes(affine))
    notes = [
        f"values are {spec.unit}",
        "affine maps voxel (col, row, slice) to RAS mm; scene coordinates are "
        "BodyParts3D LPS, so x and y flip sign.",
        f"nibabel reports axis codes {axis_codes} for this affine.",
        "the simulation notice, the BodyParts3D attribution and the full unit "
        f"description are in a NIfTI comment extension (ecode "
        f"{NIFTI_COMMENT_ECODE}); descrip is 80 bytes and holds a marker only.",
    ]
    notes.extend(spec.notes)
    if geometry.transform_applied != "none":
        notes.append(
            f"mesh transform {geometry.transform_applied!r} was applied to the "
            "scene, so the RAS interpretation of the affine inherits it."
        )

    sidecar_path: Path | None = None
    if sidecar:
        sidecar_path = path.with_suffix(path.suffix + ".provenance.json")
        doc = provenance_document(
            list(provenance or []), fmt="nifti", target=path.name,
            extra={
                "simulation": SIMULATION_STATEMENT,
                "volume": volume.as_dict(),
                "pixel_values": spec.as_dict(),
                "nifti": {
                    "shape_i_j_k": [int(v) for v in data.shape],
                    "dtype": str(data.dtype),
                    "affine_voxel_to_ras_mm": [
                        [float(v) for v in row] for row in affine],
                    "axis_codes": list(axis_codes),
                },
            },
        )
        sidecar_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False),
                                encoding="utf-8")

    logger.info("wrote %s %s %s", path, data.shape, data.dtype)
    return NiftiResult(
        path=path,
        shape=tuple(int(v) for v in data.shape),         # type: ignore[arg-type]
        dtype=str(data.dtype),
        affine=affine,
        axis_codes=axis_codes,                           # type: ignore[arg-type]
        rescale=spec,
        sidecar=sidecar_path,
        notes=tuple(notes),
    )
