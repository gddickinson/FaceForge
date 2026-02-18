"""FACS (Facial Action Coding System) engine -- applies AU displacements to face mesh vertices.

Ported from the ``applyFACS()`` function in faceforge-muscles.html.
Each Action Unit affects specific vertex regions loaded from ``face_regions.json``.
Displacement vectors are accumulated on top of rest positions and written into
the face mesh geometry in-place.

This module has ZERO GL imports; all vertex math is done with NumPy.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray

from faceforge.core.mesh import MeshInstance
from faceforge.core.state import FaceState
from faceforge.constants import FACE_VERT_COUNT


class FACSEngine:
    """Apply FACS Action Unit displacements to a face mesh.

    All region loops are vectorized with NumPy fancy indexing for performance.
    """

    def __init__(
        self,
        face_mesh: MeshInstance,
        face_regions: dict[str, list[int]],
        jaw_pivot_angle_callback: Optional[Callable[[float], None]] = None,
    ) -> None:
        self._mesh = face_mesh
        self._regions = face_regions
        self._jaw_cb = jaw_pivot_angle_callback

        if face_mesh.rest_positions is None:
            face_mesh.store_rest_pose()

        self._rest: NDArray[np.float32] = face_mesh.rest_positions.copy()
        self._vert_count = face_mesh.geometry.vertex_count

        rest_2d = self._rest.reshape(-1, 3)
        self._x_sign = np.sign(rest_2d[:, 0]).astype(np.float32)

        # Pre-compute merged index arrays for each AU (avoids per-frame region lookups)
        self._au_indices: dict[str, NDArray[np.intp]] = {}
        self._init_region_indices()

        self._last_au_hash: Optional[int] = None

    def _init_region_indices(self) -> None:
        """Pre-merge region vertex indices into NumPy arrays per AU group."""
        def _merge(*region_names: str) -> NDArray[np.intp]:
            all_idx = []
            for rn in region_names:
                idx = self._regions.get(rn, [])
                all_idx.extend(i for i in idx if i < self._vert_count)
            return np.array(all_idx, dtype=np.intp) if all_idx else np.array([], dtype=np.intp)

        self._au_indices["brow_inner"] = _merge("leftBrowInner", "rightBrowInner")
        self._au_indices["brow_outer"] = _merge("leftBrowOuter", "rightBrowOuter")
        self._au_indices["brow_all"] = _merge("leftBrowInner", "rightBrowInner",
                                               "leftBrowOuter", "rightBrowOuter")
        self._au_indices["eye_upper"] = _merge("leftEyeUpper", "rightEyeUpper")
        self._au_indices["cheek"] = _merge("leftCheek", "rightCheek")
        self._au_indices["nose"] = _merge("noseBridge", "noseBottom")
        self._au_indices["lips"] = _merge("upperLip", "lowerLip")
        self._au_indices["upper_lip"] = _merge("upperLip")
        self._au_indices["lower_lip"] = _merge("lowerLip")
        self._au_indices["jaw"] = _merge("jawLine", "chin", "lowerLip")

    def apply(self, face_state: FaceState) -> float:
        """Apply all AU displacements and return the jaw angle in radians."""
        au = face_state.get_au_dict()

        au_hash = hash(tuple(sorted(au.items())))
        if au_hash == self._last_au_hash:
            return au.get("AU26", 0.0) * 0.28 + au.get("AU25", 0.0) * 0.06
        self._last_au_hash = au_hash

        pos = self._rest.copy()
        p = pos.reshape(-1, 3)
        xs = self._x_sign
        idx = self._au_indices

        # AU1: Inner Brow Raise
        v = au.get("AU1", 0.0)
        if v > 0:
            ix = idx["brow_inner"]
            p[ix, 1] += 0.8 * v

        # AU2: Outer Brow Raise
        v = au.get("AU2", 0.0)
        if v > 0:
            ix = idx["brow_outer"]
            p[ix, 1] += 0.6 * v

        # AU4: Brow Lower + inward pull
        v = au.get("AU4", 0.0)
        if v > 0:
            ix = idx["brow_all"]
            p[ix, 1] -= 0.5 * v
            p[ix, 0] -= 0.1 * v * xs[ix]

        # AU5: Upper Lid Raise
        v = au.get("AU5", 0.0)
        if v > 0:
            ix = idx["eye_upper"]
            p[ix, 1] += 0.4 * v

        # AU6: Cheek Raise
        v = au.get("AU6", 0.0)
        if v > 0:
            ix = idx["cheek"]
            p[ix, 1] += 0.3 * v
            p[ix, 2] -= 0.1 * v

        # AU9: Nose Wrinkle
        v = au.get("AU9", 0.0)
        if v > 0:
            ix = idx["nose"]
            p[ix, 1] += 0.25 * v
            p[ix, 2] -= 0.1 * v

        # AU12: Lip Corner Pull (smile)
        v = au.get("AU12", 0.0)
        if v > 0:
            ix = idx["lips"]
            p[ix, 1] += 0.5 * v
            p[ix, 0] += 0.15 * v * xs[ix]

        # AU15: Lip Corner Drop (frown)
        v = au.get("AU15", 0.0)
        if v > 0:
            ix = idx["lips"]
            p[ix, 1] -= 0.4 * v

        # AU20: Lip Stretch
        v = au.get("AU20", 0.0)
        if v > 0:
            ix = idx["lips"]
            p[ix, 0] += 0.3 * v * xs[ix]

        # AU22: Lip Funneler
        v = au.get("AU22", 0.0)
        if v > 0:
            ix = idx["lips"]
            p[ix, 2] += 0.4 * v
            p[ix, 0] -= 0.15 * v * xs[ix]

        # AU25: Lips Part
        v = au.get("AU25", 0.0)
        if v > 0:
            p[idx["upper_lip"], 1] += 0.15 * v
            p[idx["lower_lip"], 1] -= 0.15 * v

        # AU26: Jaw Drop
        au26 = au.get("AU26", 0.0)
        au25 = au.get("AU25", 0.0)
        jaw_angle_rad = au26 * 0.28 + au25 * 0.06
        if au26 > 0:
            p[idx["jaw"], 1] -= 0.8 * au26

        self._mesh.geometry.positions[:] = pos
        self._mesh.needs_update = True

        if self._jaw_cb is not None:
            self._jaw_cb(jaw_angle_rad)

        return jaw_angle_rad

    def reset(self) -> None:
        """Reset face mesh to rest positions."""
        self._mesh.geometry.positions[:] = self._rest
        self._mesh.needs_update = True
        self._last_au_hash = None
