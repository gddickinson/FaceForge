"""How a neck muscle's lower end follows the body it hangs from.

Two things move a neck muscle's body end.  The *regional* anchors -- shoulder,
ribcage, thoracic -- are coarse averages of the pivots in that region, and the
*bone* anchors are the specific bones the muscle actually attaches to, named
per muscle as ``lowerBones`` in ``assets/config/muscles/neck_muscles.json``.

Measured at full thoracic flexion: the top thoracic pivot travels 4.58 units,
but T1 -- where longus colli and longus capitis originate -- does not move at
all, because the cervical chain that carries T1 hangs off ``bodyRoot`` rather
than off the thoracic spine.  Driving those muscles from the regional anchor
therefore dragged their origins 3.77 units off the bone they are attached to,
which the pinning pass then spent its strength undoing.  Preferring the bone
displacement when the muscle names one removes the disagreement instead of
splitting the difference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover - typing only
    from faceforge.anatomy.neck_muscles import NeckMuscleData

#: Lowest spine fraction a muscle's body end is remapped to, by attachment.
#: Below this the vertex tracks the body rather than the head.
BODY_FOLLOW_BASE = {
    "shoulder": 0.12,
    "ribcage": 0.30,
    "thoracic": 0.18,
}
BODY_FOLLOW_MAX = {
    "shoulder": 0.25,
    "ribcage": 0.45,
    "thoracic": 0.30,
}

#: The anchor regions :func:`regional_deltas` reports.
ANCHOR_REGIONS = ("shoulder", "ribcage", "thoracic")

#: Overall pin strength (0 = disabled, 1 = hard pin to bone position).
PIN_STRENGTH = 0.6

_ZERO = np.zeros(3, dtype=np.float64)


def regional_deltas(
    rest: dict[str, NDArray],
    current: dict[str, NDArray],
) -> dict[str, NDArray[np.float64]]:
    """Body anchor displacement from rest to current, per region.

    Both dictionaries must be in the same frame; see
    :mod:`faceforge.coordination.body_anchors` for why that is not automatic.
    """
    deltas: dict[str, NDArray[np.float64]] = {}
    for region in ANCHOR_REGIONS:
        a = rest.get(region)
        b = current.get(region)
        deltas[region] = (np.asarray(b, dtype=np.float64)
                          - np.asarray(a, dtype=np.float64)
                          if a is not None and b is not None else _ZERO)
    return deltas


def bone_displacement(
    registry: Any,
    muscle_name: str,
    lower_bones: Optional[list[str]],
) -> Optional[NDArray[np.float64]]:
    """How far this muscle's named attachment bones have moved since rest.

    ``None`` when there is no registry, no ``lowerBones``, or none of the
    named bones is registered -- the caller then falls back to the region.
    """
    if registry is None or not lower_bones:
        return None
    current = registry.get_muscle_anchor_current(muscle_name, lower_bones)
    rest = registry.get_muscle_anchor(muscle_name, lower_bones)
    if current is None or rest is None:
        return None
    return np.asarray(current, dtype=np.float64) - np.asarray(rest, dtype=np.float64)


def body_delta_for(
    md: "NeckMuscleData",
    regional: dict[str, NDArray[np.float64]],
    registry: Any,
) -> NDArray[np.float64]:
    """The displacement this muscle's body end should follow.

    The muscle's own attachment bones win; the regional average is the
    fallback for the muscles that name none.
    """
    delta = bone_displacement(registry, md.defn.get("name", ""),
                              md.defn.get("lowerBones"))
    if delta is not None:
        return delta
    return regional.get(md.lower_attach, _ZERO)


def pin_weights(md: "NeckMuscleData") -> Optional[NDArray[np.float64]]:
    """1 at the body end, 0 at the skull end, quadratic in between.

    ``None`` when the muscle has no usable spread of spine fractions.
    """
    fracs = md.spine_fracs.astype(np.float64)
    frac_min = float(fracs.min())
    frac_max = float(fracs.max())
    frac_range = frac_max - frac_min
    if frac_range < 1e-6:
        return None
    weight = 1.0 - np.clip((fracs - frac_min) / frac_range, 0.0, 1.0)
    return weight ** 2


def apply_bone_pinning(
    out_pos: NDArray[np.float64],
    md: "NeckMuscleData",
    registry: Any,
    strength: float = PIN_STRENGTH,
) -> None:
    """Pin lower-end vertices toward their bone attachment positions.

    Uses the per-muscle ``lowerBones`` config to query the registry for the
    specific attachment bones rather than a shared global anchor.  A no-op
    when no registry, no bones, or no resolvable anchor.

    Pin strength is strongest at the body end (lowest spine fraction) and
    fades to zero at the skull end.
    """
    delta = bone_displacement(registry, md.defn.get("name", ""),
                              md.defn.get("lowerBones"))
    if delta is None:
        return

    weight = pin_weights(md)
    if weight is None:
        return

    # Target: each vertex's rest position carried by its bone.
    rest = md.rest_positions.reshape(-1, 3).astype(np.float64)
    target = rest + delta

    blend = weight[:, None] * strength
    out_pos[:] = out_pos * (1.0 - blend) + target * blend
