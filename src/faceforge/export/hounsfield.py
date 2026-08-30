"""What the virtual scanner's numbers are, and what they are not.

Read this before trusting a pixel value out of :mod:`faceforge.export.dicom`.

What the model actually is
--------------------------
:mod:`faceforge.scanner.tissue_map` holds ``_CT_TABLE``: a hand-authored
lookup from a tissue class to a number in ``0..1`` (bone 0.95, muscle 0.45,
fat 0.15, fluid 0.10, air 0.0).  Those numbers were chosen so that the
grayscale colour map produces a CT-looking picture.  They are not Hounsfield
units, not linear attenuation coefficients, and not derived from any physical
material model -- nothing in FaceForge knows the density or the effective
atomic number of anything.

:class:`~faceforge.scanner.engine.ScannerEngine` then casts rays through
*triangle surfaces*.  There is no filled volume anywhere in the pipeline: a
pixel's value is a reduction over the tissue-table values of whichever mesh
surfaces that ray crossed inside the slab.  With ``reduction="mean"`` the value
is an average over surface crossings, which corresponds to no physical
quantity at all -- it is not a path integral, because nothing integrates over a
material, and a mesh crossed twice contributes twice.

So a CT number cannot be *computed* from this model.  Writing one anyway --
picking values that look like HU and tagging them ``RescaleType = HU`` -- would
turn a teaching illustration into a measurement someone could report, which is
exactly the failure mode a DICOM wrapper invites.  This module therefore offers
two modes and refuses to blur them.

``index`` -- the default, and the honest floor
----------------------------------------------
Store the model's own dimensionless value.  Stored pixel value is
``round(v * 1000)`` with ``RescaleSlope = 0.001`` and ``RescaleIntercept = 0``,
so applying the DICOM rescale returns ``v`` in ``0..1`` exactly as the scanner
produced it.  ``RescaleType`` is ``"US"`` -- the DICOM defined term for
*unspecified* -- because the quantity genuinely has no unit.  Nothing here
claims to be CT numbers; a reader that computes a mean "HU" over a region gets
0..1 and can tell immediately that it is not HU.

``class`` -- nominal HU, opt-in, and only where it is invertible
---------------------------------------------------------------
There is one configuration in which the model does support a defensible HU
mapping.  With ``mode="ct"`` and ``reduction="max"``, every pixel value is the
maximum over surface crossings of table entries, so it is *exactly equal to one
entry of the table* -- and the table is (almost) injective, so the value
inverts to a tissue class with no interpolation and no guessing.  A tissue class
can then be assigned a nominal HU from :data:`TISSUE_NOMINAL_HU`.

The limits of that, stated plainly, and repeated in the DICOM
DerivationDescription of every file written this way:

* The HU values are **nominal reference values for a tissue class**, taken from
  the ranges standardly tabulated in radiology texts.  They are not measured,
  not patient-derived, and carry no intra-tissue variation: every bone voxel in
  the volume has the identical value.
* Only the *class assignment* is meaningful, and it is only as good as
  :meth:`~faceforge.scanner.tissue_map.TissueMapper.classify`, which matches
  mesh-name substrings and falls back to material brightness.
* There is no partial-volume mixing, no noise, no beam hardening, no scatter,
  no contrast agent and no reconstruction kernel.  Edges are exact.
* Because the sampling is of surfaces, a "voxel" reports the densest surface
  the ray crossed in the slab, not the material occupying that volume.  The
  interior of a solid structure is filled only where the ray crossed its
  bounding surface within the slab.

If the caller asks for ``class`` mode with any other reduction,
:func:`to_hounsfield` raises rather than silently mapping values that are not
table entries -- with ``reduction="mean"`` a pixel could sit halfway between
fat and bone, and there is no honest class for that.

The two tissue classes that collide
-----------------------------------
``nerve`` and ``skin`` share the CT-table value 0.35, so the value cannot
distinguish them.  Both are assigned 30 HU, which makes the inverse map
well-defined; :func:`check_table_consistency` asserts that no colliding pair
disagrees, and ``tests/export/test_hounsfield.py`` runs it against the live
table so a future edit to ``tissue_map`` that breaks the property fails loudly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

#: Tissue classes :class:`~faceforge.scanner.tissue_map.TissueMapper` can
#: return.  Mirrors ``tissue_map._TISSUE_KEYWORDS`` plus the two synthetic
#: entries (``air``, ``unknown``); the *values* are always read from
#: ``TissueMapper.get_value`` at call time rather than copied, and
#: ``tests/export/test_hounsfield.py`` checks this list against the module's own
#: table so a newly added tissue cannot go unnoticed.
TISSUE_CLASSES: tuple[str, ...] = (
    "bone", "cartilage", "muscle", "organ", "brain", "vessel", "nerve",
    "fat", "skin", "fluid", "ligament", "eye", "ear", "air", "unknown",
)

#: Nominal CT numbers per tissue class, in Hounsfield units.
#:
#: These are round numbers taken from the reference ranges standardly tabulated
#: for CT (air -1000, water 0, fat about -100, soft tissue +20 to +60,
#: cancellous-to-cortical bone +300 to +1900).  They are **nominal class
#: values, not measurements**, and FaceForge did not compute them from
#: anything -- see the module docstring.  ``bone`` is a single value for
#: everything from a nasal turbinate to a femoral shaft, which is the coarsest
#: approximation in the table.
TISSUE_NOMINAL_HU: dict[str, int] = {
    "air": -1000,
    "fat": -100,
    "fluid": 0,          # water / CSF
    "eye": 10,           # vitreous humour, essentially water
    "nerve": 30,
    "skin": 30,
    "brain": 35,
    "organ": 45,
    "muscle": 45,
    "vessel": 55,        # unenhanced blood
    "unknown": 40,
    "ligament": 90,
    "cartilage": 100,
    "ear": 100,          # auricular cartilage
    "bone": 700,         # nominal cortical/cancellous mix
}

#: HU modes :func:`to_hounsfield` accepts.
HU_MODES: tuple[str, ...] = ("index", "class")

#: ``index`` mode multiplies the 0..1 model value by this before storing it.
#: 1000 keeps three decimal places exactly representable and makes
#: ``RescaleSlope`` a clean 0.001.
INDEX_SCALE = 1000

#: Offset applied in ``class`` mode so stored values are unsigned.  -1024 is
#: the conventional CT intercept.
HU_INTERCEPT = -1024

#: How close a pixel must be to a table entry to be inverted, in model units.
#: The scanner works in float32 and the table entries are exact float64
#: constants, so the achievable agreement is around 1e-7; 1e-4 is loose enough
#: to survive a float32 round-trip and far tighter than the 0.02 minimum gap
#: between adjacent table entries.
CLASS_TOLERANCE = 1e-4


class HUMappingError(RuntimeError):
    """The requested HU mapping is not defensible for this data."""


@dataclass(frozen=True)
class RescaleSpec:
    """DICOM rescale tags plus an honest description of the stored values."""

    stored: np.ndarray            # integer array as it goes into PixelData
    slope: float
    intercept: float
    rescale_type: str             # DICOM defined term: "HU" or "US"
    unit: str                     # human-readable
    signed: bool
    mode: str
    real_min: float
    real_max: float
    description: str
    notes: tuple[str, ...] = ()
    class_report: dict[str, Any] | None = None

    def real_values(self) -> np.ndarray:
        """Apply the rescale, i.e. what a DICOM reader will compute."""
        return self.stored.astype(np.float64) * self.slope + self.intercept

    def as_dict(self) -> dict[str, Any]:
        return {
            "hu_mode": self.mode,
            "rescale_slope": self.slope,
            "rescale_intercept": self.intercept,
            "rescale_type": self.rescale_type,
            "unit": self.unit,
            "real_value_range": [self.real_min, self.real_max],
            "description": self.description,
            "notes": list(self.notes),
            "class_mapping": self.class_report,
        }


def ct_table() -> dict[str, float]:
    """The CT tissue table, read from :mod:`faceforge.scanner.tissue_map`.

    Read through the public ``get_value`` so this module never depends on the
    table's private storage.
    """
    from faceforge.scanner.tissue_map import TissueMapper

    return {t: float(TissueMapper.get_value(t, "ct")) for t in TISSUE_CLASSES}


def check_table_consistency() -> dict[str, Any]:
    """Verify the value -> HU inverse map is well defined.

    Two tissue classes sharing a CT-table value must agree on their nominal HU,
    or the value cannot be inverted.  Raises :class:`HUMappingError` naming the
    offending classes; otherwise returns the collision report.
    """
    table = ct_table()
    missing = sorted(set(table) - set(TISSUE_NOMINAL_HU))
    if missing:
        raise HUMappingError(
            f"tissue classes with no nominal HU assigned: {missing}.  A class "
            "with no HU cannot be exported in 'class' mode; add it to "
            "TISSUE_NOMINAL_HU with a justification, or use 'index' mode."
        )

    groups: dict[float, list[str]] = {}
    for tissue, value in table.items():
        groups.setdefault(round(value, 6), []).append(tissue)

    collisions = {}
    for value, tissues in sorted(groups.items()):
        if len(tissues) < 2:
            continue
        hus = {TISSUE_NOMINAL_HU[t] for t in tissues}
        collisions[value] = {"tissues": sorted(tissues), "nominal_hu": sorted(hus)}
        if len(hus) > 1:
            raise HUMappingError(
                f"CT-table value {value} is shared by {sorted(tissues)} but "
                f"they are assigned different nominal HU {sorted(hus)}, so a "
                "pixel with that value cannot be inverted to one CT number.  "
                "Give the colliding classes the same nominal HU or separate "
                "their table values."
            )
    return {
        "distinct_values": len(groups),
        "tissue_classes": len(table),
        "colliding_values": collisions,
    }


def value_to_hu_map() -> dict[float, int]:
    """CT-table value -> nominal HU.  Validated by :func:`check_table_consistency`."""
    check_table_consistency()
    table = ct_table()
    out: dict[float, int] = {}
    for tissue, value in table.items():
        out[round(value, 6)] = TISSUE_NOMINAL_HU[tissue]
    return out


def value_to_classes() -> dict[float, list[str]]:
    """CT-table value -> the tissue classes sharing it."""
    groups: dict[float, list[str]] = {}
    for tissue, value in ct_table().items():
        groups.setdefault(round(value, 6), []).append(tissue)
    return {v: sorted(t) for v, t in groups.items()}


# ---------------------------------------------------------------------------
# The two encodings
# ---------------------------------------------------------------------------


def _index_spec(image: np.ndarray) -> RescaleSpec:
    values = np.clip(np.asarray(image, dtype=np.float64), 0.0, 1.0)
    stored = np.rint(values * INDEX_SCALE).astype(np.uint16)
    return RescaleSpec(
        stored=stored,
        slope=1.0 / INDEX_SCALE,
        intercept=0.0,
        rescale_type="US",
        unit="dimensionless tissue-radiodensity index, 0..1",
        signed=False,
        mode="index",
        real_min=float(values.min()),
        real_max=float(values.max()),
        description=(
            "Stored values are the FaceForge virtual scanner's own "
            "dimensionless tissue index scaled by 1000. Applying "
            "RescaleSlope/RescaleIntercept returns the model value in 0..1. "
            "These are NOT Hounsfield units and NOT attenuation "
            "coefficients: the underlying model is a hand-authored lookup "
            "from tissue class to a display value, sampled over triangle "
            "surface crossings, with no material physics. RescaleType is US "
            "(unspecified) because the quantity has no unit."
        ),
        notes=(
            "0.0 means no surface was crossed in the slab, not air with a "
            "measured density.",
        ),
    )


def _class_spec(image: np.ndarray, *, reduction: str, mode: str) -> RescaleSpec:
    if mode != "ct":
        raise HUMappingError(
            f"hu_mode='class' needs the CT tissue table, but the scan mode is "
            f"{mode!r}.  MRI intensities have no HU equivalent at all -- HU is "
            "defined by X-ray attenuation.  Use hu_mode='index'."
        )
    if reduction != "max":
        raise HUMappingError(
            f"hu_mode='class' needs reduction='max', not {reduction!r}.  Only "
            "'max' leaves each pixel exactly equal to one entry of the tissue "
            "table, which is what makes the value invertible to a tissue "
            "class.  With 'mean' a pixel can sit between two tissues and there "
            "is no honest class for it; with 'sum' the value is a nonlinear "
            "function of the crossings.  Use hu_mode='index' instead, or "
            "rescan with reduction='max'."
        )

    values = np.asarray(image, dtype=np.float64)
    lut = value_to_hu_map()
    table_values = np.array(sorted(lut), dtype=np.float64)
    table_hu = np.array([lut[v] for v in sorted(lut)], dtype=np.int32)

    # Nearest table entry per pixel, then a hard tolerance check.  No
    # interpolation: a pixel that is not a table entry is a bug in the caller's
    # reduction, not something to round away.
    idx = np.abs(values[..., None] - table_values[None, ...]).argmin(axis=-1)
    residual = np.abs(values - table_values[idx])
    worst = float(residual.max())
    if worst > CLASS_TOLERANCE:
        bad = int((residual > CLASS_TOLERANCE).sum())
        raise HUMappingError(
            f"{bad} of {values.size} pixels are not equal to any CT tissue "
            f"table entry (worst mismatch {worst:.6g} > {CLASS_TOLERANCE}), so "
            "they cannot be inverted to a tissue class.  This means the scan "
            "was not produced with reduction='max'.  Refusing to invent a "
            "class for them."
        )

    hu = table_hu[idx].astype(np.int32)
    # A pixel no ray hit is a table value of 0.0 == air, which maps to
    # -1000 HU.  That is the correct HU for "nothing there" and is stated in
    # the notes rather than left implicit.
    stored = (hu - HU_INTERCEPT).astype(np.uint16)

    classes = value_to_classes()
    present = {}
    for j, value in enumerate(table_values):
        count = int((idx == j).sum())
        if count:
            present["|".join(classes[round(float(value), 6)])] = {
                "table_value": float(value),
                "nominal_hu": int(table_hu[j]),
                "voxels": count,
                "fraction": count / values.size,
            }

    return RescaleSpec(
        stored=stored,
        slope=1.0,
        intercept=float(HU_INTERCEPT),
        rescale_type="HU",
        unit="Hounsfield units (nominal per tissue class, NOT measured)",
        signed=False,
        mode="class",
        real_min=float(hu.min()),
        real_max=float(hu.max()),
        description=(
            "SIMULATED, NOT AN ACQUISITION. Pixel values are nominal "
            "Hounsfield units assigned per tissue class. The FaceForge "
            "virtual scanner casts rays through triangle surface meshes; with "
            "reduction=max each pixel equals exactly one entry of the "
            "hand-authored tissue table, which is inverted here to a tissue "
            "class and then assigned a nominal reference CT number for that "
            "class. Only the class assignment carries information: the HU "
            "values are round reference figures, identical for every voxel of "
            "a class, with no intra-tissue variation, no partial-volume "
            "mixing, no noise, no beam hardening, no scatter and no contrast. "
            "Do not use for dosimetry, densitometry or any quantitative "
            "measurement."
        ),
        notes=(
            "Voxels no ray crossed are mapped to air, -1000 HU.",
            "Surface sampling, not volume sampling: a voxel reports the "
            "densest surface crossed within the slab, not the material "
            "occupying it.",
            "nerve and skin share the tissue-table value 0.35 and are both "
            "assigned 30 HU; the value cannot distinguish them.",
        ),
        class_report={
            "tolerance": CLASS_TOLERANCE,
            "worst_residual": worst,
            "classes_present": present,
        },
    )


def to_hounsfield(
    image: np.ndarray,
    *,
    hu_mode: str = "index",
    mode: str = "ct",
    reduction: str = "max",
) -> RescaleSpec:
    """Encode a scan for DICOM, in the mode the caller asked for.

    See the module docstring for what each mode claims.  ``index`` is the
    default because it is true of any scan; ``class`` is opt-in because it is
    only defensible for ``mode="ct"`` with ``reduction="max"``.
    """
    if hu_mode not in HU_MODES:
        raise HUMappingError(
            f"unknown hu_mode {hu_mode!r}; known: {list(HU_MODES)}")
    if hu_mode == "index":
        return _index_spec(image)
    return _class_spec(image, reduction=reduction, mode=mode)
