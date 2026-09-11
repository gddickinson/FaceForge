"""Apply a sex morph to the soft tissue: the skeleton's change, then its own.

Three things happen to a muscle or a patch of skin when the body changes sex,
and they compose in this order:

1. **The skeleton moves it.**  A shorter humerus carries the biceps with it.
   This is a smooth spatial warp built from how far every joint moved
   (:mod:`faceforge.body.skeleton_field`) and not a skinning pass, because a
   proportion change is not a pose -- see that module for the measurements.
2. **Muscle bellies thin.**  Length comes from the bones; girth does not
   (:mod:`faceforge.body.muscle_morph`).
3. **Soft tissue redistributes.**  Breast, gluteal and thigh fat, and a waist
   narrower relative to the hip (:mod:`faceforge.body.skin_morph`).

Everything is recomputed from a captured copy of the original rest pose, so
the slider can be dragged in any order and 0 restores the body exactly.  The
result is written to each mesh's *rest* positions: the skinning is re-bound
afterwards, so the morphed body becomes the pose-neutral body and joint
animation continues to work from there.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, Optional

import numpy as np
from numpy.typing import NDArray

from faceforge.body.muscle_morph import MuscleMorph
from faceforge.body.skin_morph import SkinShapeMorph

logger = logging.getLogger(__name__)


class SoftTissueMorph:
    """Owns the original rest pose of every soft-tissue mesh and rebuilds it."""

    def __init__(self, muscle: Optional[MuscleMorph] = None) -> None:
        self._muscle = muscle if muscle is not None else MuscleMorph()
        self._rest: dict[int, NDArray[np.float32]] = {}
        self._skin_field: dict[int, NDArray[np.float64]] = {}

    @property
    def muscle(self) -> MuscleMorph:
        return self._muscle

    def base_of(self, mesh: Any) -> Optional[NDArray[np.float32]]:
        """The captured original rest positions of ``mesh``."""
        if mesh is None or getattr(mesh, "rest_positions", None) is None:
            return None
        key = id(mesh)
        rest = self._rest.get(key)
        if rest is None:
            rest = self._rest[key] = np.asarray(
                mesh.rest_positions, dtype=np.float32).copy()
        return rest

    def apply(self, bindings: Iterable[Any], gender: float,
              warp: Optional[Callable[[NDArray], NDArray]] = None,
              skin_field: Optional[SkinShapeMorph] = None,
              chain_names: Optional[dict[int, str]] = None,
              chain_of_joint: Optional[NDArray] = None) -> dict[str, int]:
        """Rebuild every binding's rest pose at ``gender``.  Returns counts."""
        stats = {"muscles": 0, "other": 0}
        g = float(max(0.0, min(1.0, gender)))
        for binding in bindings:
            mesh = getattr(binding, "mesh", None)
            base = self.base_of(mesh)
            if base is None:
                continue
            pts = base.reshape(-1, 3).astype(np.float64)
            is_muscle = bool(getattr(binding, "is_muscle", False))

            if not is_muscle and skin_field is not None and g > 0.0:
                pts = pts + self._skin_delta(mesh, skin_field, base) * g

            if warp is not None:
                pts = pts + warp(pts)

            if is_muscle:
                if chain_of_joint is not None:
                    binding._chain_of_joint = chain_of_joint
                region = self._muscle.region_of(binding, chain_names)
                factor = self._muscle.bulk_factor(region, g)
                pts = self._muscle.thin_belly(mesh, pts, factor)
                stats["muscles"] += 1
            else:
                stats["other"] += 1

            flat = pts.reshape(-1).astype(np.float32)
            if len(flat) != len(np.asarray(mesh.rest_positions).ravel()):
                logger.warning("Rest pose length changed for %s; skipped", mesh.name)
                continue
            mesh.rest_positions = flat
            # geometry.positions is deliberately left alone: the skinning
            # rewrites it from the rest pose on the next update, and resizing
            # it here would desynchronise anything holding vertex indices into
            # it -- the fibre fields index the muscle's own vertices.
            # The skinning caches the rest pose in float64 and as homogeneous
            # points; both must be dropped or the new shape never reaches a frame.
            binding._rest_f64 = None
            binding._pos_h = None
            mesh.needs_update = True
        return stats

    def _skin_delta(self, mesh: Any, field: SkinShapeMorph,
                    base: NDArray) -> NDArray:
        key = id(mesh)
        cached = self._skin_field.get(key)
        if cached is None:
            pts = base.reshape(-1, 3).astype(np.float64)
            delta = field.delta_for(pts)
            delta = field.smooth_on_mesh(delta, mesh.geometry.indices, len(pts))
            delta = field.constrain(delta, pts, mesh.geometry.indices)
            cached = self._skin_field[key] = delta
            mag = np.linalg.norm(delta, axis=1)
            logger.info("Soft-tissue sex field on %s: median %.2f max %.2f",
                        mesh.name, float(np.median(mag)), float(mag.max()))
        return cached
