"""Stack virtual-scanner slices into a volume with defensible geometry.

:func:`faceforge.session.scan_scene` produces one cross-section.  A DICOM
series or a NIfTI volume needs a stack, and -- far more importantly -- needs
every millimetre of that stack described correctly.  A volume that loads with
the wrong voxel size or a flipped axis is worse than no volume at all: it looks
plausible, it measures wrong, and nothing about it announces the problem.  So
the geometry is derived here, once, and validated by reconstructing voxel
positions from the written file's own tags and comparing them against the
scanner's ray grid (``tests/export/test_dicom.py``, ``test_nifti.py``).

Scene coordinates are BodyParts3D coordinates
---------------------------------------------
With ``transform=None`` (the CLI default, ``--transform none``) FaceForge loads
BodyParts3D STL vertices unmodified, so scene coordinates *are* BodyParts3D
coordinates.  The axis convention was measured from the meshes themselves
rather than assumed -- mesh centroids, ``assets/stl``:

===================  ==========================  ==========================
axis                 evidence                    conclusion
===================  ==========================  ==========================
X                    right femur (FMA24474)      +X is patient LEFT
                     centroid x = -87.4;
                     left femur (FMA24475)
                     centroid x = +86.3
Y                    body of sternum (FMA7487)   +Y is patient POSTERIOR
                     centroid y = -187.3;
                     T9 vertebra (FMA10014)
                     centroid y = -35.6
Z                    mandible (FMA52748)         +Z is patient SUPERIOR
                     centroid z = 1472.9;
                     sacrum (FMA16202)
                     centroid z = 878.2
===================  ==========================  ==========================

That is (Left, Posterior, Superior) -- the DICOM patient coordinate system
exactly -- so :data:`SCENE_TO_LPS` is the identity, and the units are
millimetres (the femur meshes span 440 mm head to condyle).  This is asserted
against the real assets in ``tests/export/test_volume.py`` so that a future
change to the loader breaks the test rather than silently rotating every
exported volume.  ``--transform bp3d`` applies a non-uniform, X-mirroring
transform to the meshes; a volume exported from a transformed scene is
therefore *not* in LPS, and :func:`scan_volume` records which was used.

Sampling geometry
-----------------
Two properties of :class:`~faceforge.scanner.engine.ScannerEngine` decide the
tags and are easy to get wrong:

1. The ray grid is ``linspace(-0.5, 0.5, resolution)`` across the field, so
   adjacent samples are ``field / (resolution - 1)`` apart -- *not*
   ``field / resolution``.  The outermost samples sit exactly on the field
   boundary.  Using ``field / resolution`` would put a 0.8 % scale error in
   every measurement taken off the export at resolution 128.
2. The slab runs from the plane origin *forward* along the ray direction by
   ``depth``; it is not centred on the origin.  So a slice whose reported
   position is the slab centre must be cast from
   ``centre - depth/2 * ray_direction``.  Skipping that offsets the whole
   series by half a slice thickness.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

logger = logging.getLogger(__name__)

#: Scene axes -> DICOM patient (LPS) axes.  Identity, measured; see the module
#: docstring.  Kept as an explicit matrix rather than left implicit so that the
#: assumption is visible, testable and overridable.
SCENE_TO_LPS: np.ndarray = np.eye(3, dtype=np.float64)

#: DICOM patient (LPS) -> NIfTI world (RAS).  x and y flip sign.
LPS_TO_RAS: np.ndarray = np.diag([-1.0, -1.0, 1.0])

#: Modes that describe a scalar tomographic slice, so a volume is meaningful.
#: ``xray`` is a projection through the whole slab, not a slice, and
#: ``anatomical`` is RGB; stacking either into a volume would be a category
#: error, so the exporters refuse them by name.
VOLUME_MODES: tuple[str, ...] = ("ct", "mri_t1", "mri_t2")


class VolumeError(RuntimeError):
    """A volume that cannot be built or described correctly."""


@dataclass(frozen=True)
class VolumeGeometry:
    """Where every voxel of a scanned volume is, in scene millimetres.

    Index order throughout is ``(slice, row, column)`` -- the order a DICOM
    series is stored in and the order :attr:`ScanVolume.data` uses.
    ``row_dir`` is the direction of increasing *column* index and ``col_dir``
    the direction of increasing *row* index, matching DICOM's
    ImageOrientationPatient, whose two vectors are named the same confusing way.
    """

    orientation: str
    centre: np.ndarray            # (3,) scene mm, centre of the whole volume
    row_dir: np.ndarray           # (3,) unit, increasing column index
    col_dir: np.ndarray           # (3,) unit, increasing row index
    stack_dir: np.ndarray         # (3,) unit, increasing slice index
    ray_dir: np.ndarray           # (3,) unit, direction rays are cast
    field_width: float            # mm, spanned by the outermost samples
    field_height: float
    resolution: int
    slices: int
    slice_spacing: float          # mm between adjacent slice centres
    slab_depth: float             # mm sampled per slice (SliceThickness)
    scene_to_lps: np.ndarray = field(default_factory=lambda: SCENE_TO_LPS.copy())
    transform_applied: str = "none"

    # -- in-plane spacing -------------------------------------------------

    @property
    def column_spacing(self) -> float:
        """mm between adjacent columns, i.e. along :attr:`row_dir`."""
        return self.field_width / (self.resolution - 1)

    @property
    def row_spacing(self) -> float:
        """mm between adjacent rows, i.e. along :attr:`col_dir`."""
        return self.field_height / (self.resolution - 1)

    @property
    def pixel_spacing(self) -> tuple[float, float]:
        """DICOM PixelSpacing: ``(row spacing, column spacing)``, in that order."""
        return (self.row_spacing, self.column_spacing)

    # -- positions --------------------------------------------------------

    def slice_centre(self, k: int) -> np.ndarray:
        """Scene position of the centre of slice *k*'s slab."""
        if not 0 <= k < self.slices:
            raise IndexError(f"slice {k} outside 0..{self.slices - 1}")
        offset = (k - (self.slices - 1) / 2.0) * self.slice_spacing
        return self.centre + offset * self.stack_dir

    def scan_origin(self, k: int) -> np.ndarray:
        """Where to put the scan plane so slice *k*'s slab is centred on it.

        The engine sweeps ``[origin, origin + depth * ray_dir]``, so the origin
        sits half a slab behind the reported position.
        """
        return self.slice_centre(k) - 0.5 * self.slab_depth * self.ray_dir

    def voxel_position(self, k: int, row: int, col: int) -> np.ndarray:
        """Scene position of voxel ``(slice k, row, col)``."""
        return (
            self.slice_centre(k)
            + (col - (self.resolution - 1) / 2.0) * self.column_spacing * self.row_dir
            + (row - (self.resolution - 1) / 2.0) * self.row_spacing * self.col_dir
        )

    def image_position_patient(self, k: int) -> np.ndarray:
        """DICOM ImagePositionPatient for slice *k*: voxel ``(0, 0)`` in LPS."""
        return self.to_lps(self.voxel_position(k, 0, 0))

    def image_orientation_patient(self) -> list[float]:
        """DICOM ImageOrientationPatient: row cosines then column cosines, LPS."""
        row = self.direction_to_lps(self.row_dir)
        col = self.direction_to_lps(self.col_dir)
        return [float(v) for v in (*row, *col)]

    # -- frame conversions ------------------------------------------------

    def to_lps(self, point: Sequence[float]) -> np.ndarray:
        return self.scene_to_lps @ np.asarray(point, dtype=np.float64)

    def direction_to_lps(self, vector: Sequence[float]) -> np.ndarray:
        out = self.scene_to_lps @ np.asarray(vector, dtype=np.float64)
        norm = np.linalg.norm(out)
        return out / norm if norm > 0 else out

    def to_ras(self, point: Sequence[float]) -> np.ndarray:
        return LPS_TO_RAS @ self.to_lps(point)

    def direction_to_ras(self, vector: Sequence[float]) -> np.ndarray:
        out = LPS_TO_RAS @ self.direction_to_lps(vector)
        norm = np.linalg.norm(out)
        return out / norm if norm > 0 else out

    def nifti_affine(self) -> np.ndarray:
        """4x4 voxel ``(i, j, k)`` -> RAS mm, for an array indexed ``(col, row, slice)``.

        NIfTI's world frame is RAS and its first array axis is the fastest
        varying, so the array handed to nibabel is transposed relative to
        :attr:`ScanVolume.data`: ``i`` is the column index, ``j`` the row index,
        ``k`` the slice index.  :meth:`ScanVolume.nifti_array` does that
        transpose, and this affine matches it.
        """
        affine = np.eye(4, dtype=np.float64)
        affine[:3, 0] = self.direction_to_ras(self.row_dir) * self.column_spacing
        affine[:3, 1] = self.direction_to_ras(self.col_dir) * self.row_spacing
        affine[:3, 2] = self.direction_to_ras(self.stack_dir) * self.slice_spacing
        affine[:3, 3] = self.to_ras(self.voxel_position(0, 0, 0))
        return affine

    # -- reporting --------------------------------------------------------

    @property
    def right_handed(self) -> bool:
        """True when ``row x col`` agrees with the slice stacking direction.

        A left-handed stack is legal DICOM -- ImagePositionPatient describes
        each slice regardless -- but several converters silently reorder such a
        series, so the exporters build a right-handed stack and this asserts it.
        """
        return bool(np.dot(np.cross(self.row_dir, self.col_dir),
                           self.stack_dir) > 0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "orientation": self.orientation,
            "centre_scene_mm": [float(v) for v in self.centre],
            "row_dir_scene": [float(v) for v in self.row_dir],
            "col_dir_scene": [float(v) for v in self.col_dir],
            "stack_dir_scene": [float(v) for v in self.stack_dir],
            "ray_dir_scene": [float(v) for v in self.ray_dir],
            "field_mm": {"width": self.field_width, "height": self.field_height},
            "resolution": self.resolution,
            "slices": self.slices,
            "pixel_spacing_mm": list(self.pixel_spacing),
            "slice_spacing_mm": self.slice_spacing,
            "slice_thickness_mm": self.slab_depth,
            "image_orientation_patient": self.image_orientation_patient(),
            "image_position_patient_first": [
                float(v) for v in self.image_position_patient(0)],
            "image_position_patient_last": [
                float(v) for v in self.image_position_patient(self.slices - 1)],
            "right_handed_stack": self.right_handed,
            "scene_to_lps": [[float(v) for v in row] for row in self.scene_to_lps],
            "mesh_transform_applied": self.transform_applied,
            "sample_spacing_note": (
                "in-plane spacing is field/(resolution-1): the scanner's ray "
                "grid is linspace(-0.5, 0.5, resolution) so the outermost "
                "samples lie on the field boundary."
            ),
        }


@dataclass(frozen=True)
class ScanVolume:
    """A stacked scan plus the geometry that describes it."""

    data: np.ndarray              # (slices, rows, cols) float32, 0..1
    geometry: VolumeGeometry
    mode: str
    reduction: str
    structures: int
    hit_fraction: float

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.data.shape)      # type: ignore[return-value]

    def nifti_array(self) -> np.ndarray:
        """``(col, row, slice)`` view, matching :meth:`VolumeGeometry.nifti_affine`."""
        return np.ascontiguousarray(np.transpose(self.data, (2, 1, 0)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "reduction": self.reduction,
            "shape_slice_row_col": list(self.shape),
            "structures": self.structures,
            "hit_fraction": self.hit_fraction,
            "value_range": [float(self.data.min()), float(self.data.max())],
            "geometry": self.geometry.as_dict(),
        }


def build_geometry(
    *,
    orientation: str,
    centre: Sequence[float],
    field_width: float,
    field_height: float,
    resolution: int,
    slices: int,
    slice_spacing: float,
    slab_depth: float,
    scene_to_lps: np.ndarray | None = None,
    transform_applied: str = "none",
) -> VolumeGeometry:
    """Derive a :class:`VolumeGeometry` from scanner parameters.

    The slice stack is deliberately built along ``row x col`` rather than along
    the scanner's ray direction, so the series is right-handed with respect to
    its own ImageOrientationPatient.  Because each slab is centred on its
    reported position, the sign of the ray direction does not affect which
    geometry a slice samples.
    """
    from faceforge.session import plane_frame

    if resolution < 8:
        raise VolumeError(f"resolution {resolution} is too small to be an image")
    if slices < 1:
        raise VolumeError(f"{slices} slices is not a volume")
    if slice_spacing <= 0:
        raise VolumeError(f"slice spacing {slice_spacing} must be positive")
    if slab_depth <= 0:
        raise VolumeError(f"slab depth {slab_depth} must be positive")
    if field_width <= 0 or field_height <= 0:
        raise VolumeError(
            f"field {field_width}x{field_height} mm must be positive")

    normal, right, up = plane_frame(orientation)
    row_dir = right / np.linalg.norm(right)
    col_dir = -up / np.linalg.norm(up)
    stack_dir = np.cross(row_dir, col_dir)
    stack_norm = np.linalg.norm(stack_dir)
    if stack_norm < 1e-9:                                # pragma: no cover
        raise VolumeError(
            f"orientation {orientation!r} has parallel row and column "
            "directions, so it does not define a plane"
        )
    stack_dir = stack_dir / stack_norm

    geometry = VolumeGeometry(
        orientation=orientation,
        centre=np.asarray(centre, dtype=np.float64),
        row_dir=row_dir,
        col_dir=col_dir,
        stack_dir=stack_dir,
        ray_dir=normal / np.linalg.norm(normal),
        field_width=float(field_width),
        field_height=float(field_height),
        resolution=int(resolution),
        slices=int(slices),
        slice_spacing=float(slice_spacing),
        slab_depth=float(slab_depth),
        scene_to_lps=(SCENE_TO_LPS.copy() if scene_to_lps is None
                      else np.asarray(scene_to_lps, dtype=np.float64)),
        transform_applied=transform_applied,
    )
    if not geometry.right_handed:                        # pragma: no cover
        raise VolumeError(
            "internal error: the derived slice stack is left-handed with "
            "respect to its own ImageOrientationPatient"
        )
    return geometry


def scan_volume(
    scene: Any,
    *,
    orientation: str = "axial",
    centre: Sequence[float] = (0.0, 0.0, 0.0),
    field_width: float = 400.0,
    field_height: float = 400.0,
    resolution: int = 128,
    slices: int = 8,
    slice_spacing: float = 5.0,
    slab_depth: float | None = None,
    mode: str = "ct",
    reduction: str = "max",
    transform_applied: str = "none",
    progress: Callable[[float], None] | None = None,
) -> ScanVolume:
    """Cast *slices* cross-sections through *scene* and return them as a volume.

    ``slab_depth`` defaults to ``slice_spacing``, which makes the slabs
    contiguous and non-overlapping -- the only choice that makes
    SliceThickness and SpacingBetweenSlices agree in the DICOM output.

    ``reduction`` defaults to ``"max"``, not ``"mean"``: only ``max`` leaves
    each pixel equal to a single entry of the tissue table, which is what makes
    the value invertible to a tissue class and therefore mappable to nominal
    Hounsfield units (see :mod:`faceforge.export.dicom`).
    """
    from faceforge.session import scan_scene

    if mode not in VOLUME_MODES:
        raise VolumeError(
            f"mode {mode!r} does not describe a tomographic slice, so stacking "
            f"it into a volume would be wrong.  Volume modes: "
            f"{list(VOLUME_MODES)}.  'xray' is a projection through the whole "
            "slab and 'anatomical' is RGB; export either as an image instead."
        )
    depth = float(slice_spacing if slab_depth is None else slab_depth)
    geometry = build_geometry(
        orientation=orientation, centre=centre,
        field_width=field_width, field_height=field_height,
        resolution=resolution, slices=slices,
        slice_spacing=slice_spacing, slab_depth=depth,
        transform_applied=transform_applied,
    )

    scene.update()
    structures = len(scene.collect_meshes())
    stack = np.empty((geometry.slices, resolution, resolution), dtype=np.float32)
    for k in range(geometry.slices):
        image = scan_scene(
            scene,
            origin=geometry.scan_origin(k),
            orientation=orientation,
            width=geometry.field_width,
            height=geometry.field_height,
            depth=geometry.slab_depth,
            resolution=geometry.resolution,
            mode=mode,
            reduction=reduction,
        )
        if image.ndim != 2:                              # pragma: no cover
            raise VolumeError(
                f"mode {mode!r} returned a {image.ndim}-D slice; a volume "
                "needs scalar slices"
            )
        stack[k] = image
        if progress:
            progress((k + 1) / geometry.slices)

    hit_fraction = float((stack > 0).mean())
    if hit_fraction == 0.0:
        raise VolumeError(
            "no ray hit any geometry in any slice: the volume is empty.  The "
            f"field is centred at {list(np.asarray(centre, dtype=float))} in "
            "scene coordinates -- check it against the scene's own extent "
            "rather than assuming the origin is inside the subject."
        )
    logger.info("scanned %d slices at %dx%d, hits %.2f%%", geometry.slices,
                resolution, resolution, hit_fraction * 100)
    return ScanVolume(
        data=stack, geometry=geometry, mode=mode, reduction=reduction,
        structures=structures, hit_fraction=hit_fraction,
    )
