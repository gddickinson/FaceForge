"""L4: identify a tagged structure on a simulated cross-section.

This is the level that makes the tool resemble a real anatomy examination
rather than a flashcard deck: the stimulus is a CT/MRI-like slice produced by
the project's own ray-cast scanner, and the candidate answers are
anatomically adjacent structures drawn from the FMA graph.

How the tag position is derived
-------------------------------
``faceforge.scanner.engine.ScannerEngine.scan`` returns pixels, not identities:
it accumulates tissue values along rays and has no notion of which structure a
pixel belongs to.  Rather than change the scanner (which this module does not
own), the tag is obtained by scanning **twice through the same plane**:

1. the whole scene, giving the image the learner sees;
2. the focus structure alone, giving a mask of exactly where that structure
   appears in the plane.

The tag is the mask pixel nearest the mask's centroid -- nearest rather than
the centroid itself, because a C-shaped or bilobed section has a centroid that
lies outside the structure, and a tag that points at nothing is worse than no
question.  If the mask is empty, the structure does not intersect the plane and
:meth:`RadiologyItemBuilder.build` returns ``None``.

A change worth making in the scanner (described in the report, not done here,
since ``scanner/engine.py`` is not this module's file) would be an optional
per-pixel "which cached mesh was nearest along the ray" output.  That would
make the second scan unnecessary and would let a single scan tag several
structures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

from faceforge.anatomy.exam_items import ExamItem, Option, Provenance

logger = logging.getLogger(__name__)

#: Scan modes the scanner supports for a grayscale slice.  "anatomical" is
#: excluded: it renders mesh colours, which would make identification a colour
#: matching exercise rather than a radiological one.
GRAYSCALE_MODES = ("ct", "mri", "xray")

#: Ray reductions the scanner implements (engine.py: mean/max/min/sum).  "max"
#: is the maximum-intensity projection a radiologist would recognise, and it is
#: the default here because a mean over a 4 mm slab washes out small structures.
REDUCTIONS = ("mean", "max", "min", "sum")


@dataclass(frozen=True)
class ScanPlane:
    """A cross-section plane, in world units.

    ``origin`` is the centre of the near face of the slab; ``normal`` is the
    ray direction; ``right``/``up`` span the image.  This mirrors the argument
    list of ``ScannerEngine.scan`` exactly, so no translation layer can drift.
    """

    origin: tuple[float, float, float]
    normal: tuple[float, float, float]
    right: tuple[float, float, float]
    up: tuple[float, float, float]
    width: float
    height: float
    depth: float
    label: str = ""

    def as_scan_args(self) -> dict:
        return {
            "origin": np.asarray(self.origin, dtype=np.float32),
            "normal": np.asarray(self.normal, dtype=np.float32),
            "right": np.asarray(self.right, dtype=np.float32),
            "up": np.asarray(self.up, dtype=np.float32),
            "width": float(self.width),
            "height": float(self.height),
            "depth": float(self.depth),
        }


#: Standard radiological planes as (normal, right, up) in this project's world
#: axes (x right, y up, z forward).  Named for the plane, not the axis, because
#: that is what an examiner would say.
PLANE_BASES = {
    "axial": ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
    "coronal": ((0.0, 0.0, -1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "sagittal": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
}


def mesh_world_bounds(mesh, world_matrix) -> tuple[np.ndarray, np.ndarray]:
    """World-space AABB of one mesh instance."""
    positions = np.asarray(mesh.geometry.positions,
                           dtype=np.float32).reshape(-1, 3)
    rot = np.asarray(world_matrix, dtype=np.float32)[:3, :3]
    trans = np.asarray(world_matrix, dtype=np.float32)[:3, 3]
    world = positions @ rot.T + trans
    return world.min(axis=0), world.max(axis=0)


def plane_through_mesh(mesh, world_matrix, plane: str = "axial",
                       margin: float = 1.35, slab_margin: float = 1.2,
                       depth: Optional[float] = None) -> ScanPlane:
    """A plane cutting through the centre of ``mesh``.

    Derived from the mesh's own world-space bounding box.  The in-plane field
    of view is the box's extent times ``margin``, so the structure appears with
    surrounding context rather than filling the frame (filling it would itself
    give the answer away).

    The slab **spans the structure along the view axis** rather than being a
    thin slice centred on it, and that is deliberate.  The scanner intersects
    *surfaces*: BodyParts3D meshes are closed shells, so a 4 mm slab centred
    inside a structure contains only whatever shell wall happens to cross those
    4 mm -- for a convex structure, nothing at all, because the rays start
    inside the shell and exit past the far wall.  Spanning the structure makes
    the result a projection through it (radiographically, a thick-slab MIP),
    which is what the ``max`` reduction is for.  Pass ``depth`` explicitly for a
    true thin slice when the caller knows the geometry supports one.
    """
    normal, right, up = PLANE_BASES[plane]
    lo, hi = mesh_world_bounds(mesh, world_matrix)
    centre = (lo + hi) * 0.5
    extent = hi - lo
    right_v = np.asarray(right, dtype=np.float32)
    up_v = np.asarray(up, dtype=np.float32)
    normal_v = np.asarray(normal, dtype=np.float32)
    width = float(max(abs(float(extent @ right_v)), 1.0) * margin)
    height = float(max(abs(float(extent @ up_v)), 1.0) * margin)
    if depth is None:
        along = max(abs(float(extent @ normal_v)), 1.0)
        depth = along * slab_margin
    origin = centre - normal_v * (float(depth) * 0.5)
    return ScanPlane(
        origin=tuple(float(v) for v in origin),
        normal=tuple(float(v) for v in normal_v),
        right=tuple(float(v) for v in right_v),
        up=tuple(float(v) for v in up_v),
        width=width, height=height, depth=float(depth), label=plane,
    )


@dataclass
class RadiologyItem:
    """An exam item plus the image it refers to.

    ``image`` is the scanner's output array (``(res, res)`` for grayscale
    modes).  ``tag_px`` is the pixel the stem points at, in ``(x, y)`` with the
    origin at the top-left, matching the scanner's own image convention.
    """

    item: ExamItem
    image: np.ndarray
    tag_px: tuple[int, int]
    plane: ScanPlane
    mode: str
    mask_pixels: int

    @property
    def tag_xy(self) -> tuple[float, float]:
        res = self.image.shape[0]
        return (self.tag_px[0] / max(res - 1, 1), self.tag_px[1] / max(res - 1, 1))


class RadiologyItemBuilder:
    """Builds L4 items by calling the scanner engine.

    Parameters
    ----------
    engine_factory:
        Zero-argument callable returning a fresh ``ScannerEngine``.  A factory
        rather than an instance because ``cache_meshes`` replaces the engine's
        entire cache: the full-scene scan and the focus-only scan cannot share
        one engine without re-caching the whole scene for every item.
    generator:
        An :class:`~faceforge.anatomy.item_generators.ItemGenerator`, used for
        the option list so L4 and L1 draw distractors from the same
        neighbourhood ladder.
    """

    def __init__(self, engine_factory: Callable[[], object], generator=None):
        self._engine_factory = engine_factory
        if generator is None:
            from faceforge.anatomy.item_generators import ItemGenerator
            generator = ItemGenerator()
        self._gen = generator

    # -- masks -------------------------------------------------------------

    def _scan(self, meshes: Sequence[tuple], plane: ScanPlane, resolution: int,
              mode: str, reduction: str) -> np.ndarray:
        engine = self._engine_factory()
        engine.cache_meshes([(m, w) for m, w, *_ in meshes])
        return engine.scan(resolution=resolution, mode=mode,
                           reduction=reduction, **plane.as_scan_args())

    @staticmethod
    def _tag_from_mask(mask: np.ndarray) -> Optional[tuple[int, int]]:
        """Mask pixel nearest the mask centroid, or None for an empty mask."""
        ys, xs = np.nonzero(mask > 0)
        if len(xs) == 0:
            return None
        cx, cy = xs.mean(), ys.mean()
        d2 = (xs - cx) ** 2 + (ys - cy) ** 2
        best = int(np.argmin(d2))
        return int(xs[best]), int(ys[best])

    # -- build -------------------------------------------------------------

    def build(self, meshes: Sequence[tuple], focus_id: str,
              plane: Optional[ScanPlane] = None, plane_name: str = "axial",
              resolution: int = 128, mode: str = "ct",
              reduction: str = "max", options: int = 5, seed: int = 0,
              seconds: float = 0.0, min_mask_pixels: int = 4
              ) -> Optional[RadiologyItem]:
        """Build one L4 item, or ``None`` if the data cannot support it.

        ``meshes`` is a sequence of ``(mesh_instance, world_matrix, mesh_id)``;
        ``mesh_id`` is the BodyParts3D id used to match ``focus_id`` and to
        look up labels.  Returns ``None`` when the focus structure is absent
        from the scene, does not intersect the plane, or covers fewer than
        ``min_mask_pixels`` pixels (too small to tag honestly).
        """
        if reduction not in REDUCTIONS:
            raise ValueError(f"reduction {reduction!r} is not one of {REDUCTIONS}")
        if mode not in GRAYSCALE_MODES:
            raise ValueError(
                f"mode {mode!r} is not one of {GRAYSCALE_MODES}; anatomical "
                "mode renders mesh colours and would turn identification into "
                "colour matching")

        focus = [entry for entry in meshes if entry[2] == focus_id]
        if not focus:
            return None
        focus_mesh, focus_world = focus[0][0], focus[0][1]
        if plane is None:
            plane = plane_through_mesh(focus_mesh, focus_world, plane_name)

        mask = self._scan(focus, plane, resolution, mode, reduction)
        if mask.ndim == 3:
            mask = mask.max(axis=2)
        tag = self._tag_from_mask(mask)
        if tag is None or int(np.count_nonzero(mask > 0)) < min_mask_pixels:
            return None

        image = self._scan(meshes, plane, resolution, mode, reduction)

        base = self._gen.identification(focus_id, options=options, seed=seed,
                                        fmt="spot")
        if base is None:
            return None
        render_prov = Provenance(
            kind="scanner_render",
            reference=(f"{mode}/{reduction} {plane.label or 'plane'} "
                       f"{resolution}x{resolution}"),
            detail=(f"tag pixel {tag} is the mask pixel nearest the centroid "
                    f"of {focus_id}'s own single-mesh scan through the same "
                    f"plane ({int(np.count_nonzero(mask > 0))} px)"),
        )
        options_with_prov = tuple(base.options)
        item = ExamItem(
            level="L4", fmt="spot",
            stem=(f"This is a simulated {mode.upper()} "
                  f"{plane.label or 'cross-section'} slice. Identify the "
                  f"tagged structure."),
            options=options_with_prov,
            answer_index=base.answer_index,
            focus_id=focus_id,
            provenance=tuple(base.provenance) + (render_prov, ),
            tags=("radiology", mode, plane.label or "plane"),
            seconds=seconds,
            tag_xy=(tag[0] / max(resolution - 1, 1),
                    tag[1] / max(resolution - 1, 1)),
        )
        return RadiologyItem(
            item=item, image=image, tag_px=tag, plane=plane, mode=mode,
            mask_pixels=int(np.count_nonzero(mask > 0)),
        )

    def build_many(self, meshes: Sequence[tuple], focus_ids: Iterable[str],
                   count: int, **kwargs) -> list[RadiologyItem]:
        """Up to ``count`` items, skipping structures the plane misses."""
        out: list[RadiologyItem] = []
        for focus_id in focus_ids:
            if len(out) >= count:
                break
            item = self.build(meshes, focus_id, **kwargs)
            if item is not None:
                out.append(item)
        return out
