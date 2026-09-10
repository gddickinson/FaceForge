"""The kinematic chains the skinning binds to, built from the loaded skeleton.

One builder for the application and for every headless tool.  It used to be
a method of the asset load sequence, and ``tools/headless_loader.py`` kept
its own copy that fell behind: the app had started the arm chain at the
clavicle and scapula (so shoulder-girdle muscles have proximal joints to
bind to, and authored footprints on those bones resolve) while the headless
copy still started it at the shoulder.  Every headless render therefore
bound the rotator cuff 85-100 % to the humerus and skipped the footprints --
and measured muscles the app would not have torn.  There is now one function.

A *chain* is an ordered list of ``(name, node)`` joints that deform together;
segments and secondary joints are built only WITHIN a chain.  Chain ids are
assigned by construction order -- spine first, then limbs, then digits, then
ribs -- and recorded by name in ``chain_ids``.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

#: Arm chain joints, proximal to distal.  The clavicle comes first: it is the
#: most proximal girdle bone and the declared origin of deltoid clavicular and
#: pectoralis major clavicular.  Extending the arm chain rather than adding a
#: separate girdle chain is deliberate: a separate chain would be a hard
#: partition with no blending across the girdle -> arm boundary.
ARM_JOINTS = ("clavicle", "scapula", "shoulder", "elbow", "wrist")
LEG_JOINTS = ("hip", "knee", "ankle")
HAND_SEGMENTS = ("mc", "prox", "mid", "dist")
FOOT_SEGMENTS = ("mt", "prox", "mid", "dist")


def build_joint_chains(skeleton: Any, joint_setup: Any, rib_pivots: list | None,
                       chain_ids: dict[str, int]) -> list[list[tuple[str, Any]]]:
    """Build the chains, filling ``chain_ids`` (cleared first) with name -> id."""
    chains: list[list[tuple[str, Any]]] = []
    chain_ids.clear()

    def add(name: str, chain: list[tuple[str, Any]]) -> None:
        if chain:
            chain_ids[name] = len(chains)
            chains.append(chain)

    # Spine: thoracic top -> bottom, then lumbar.
    spine: list[tuple[str, Any]] = []
    if skeleton is not None:
        for region in ("thoracic", "lumbar"):
            for pinfo in skeleton.pivots.get(region, []):
                spine.append((f"{region}_{pinfo.get('level', 0)}", pinfo["group"]))
    add("spine", spine)

    n_hand = n_foot = 0
    if joint_setup is not None:
        pivots = joint_setup.pivots
        for side in ("R", "L"):
            for name, joints in (("arm", ARM_JOINTS), ("leg", LEG_JOINTS)):
                chain = [(f"{j}_{side}", pivots[f"{j}_{side}"])
                         for j in joints if pivots.get(f"{j}_{side}") is not None]
                add(f"{name}_{side}", chain)

        # Digits: one chain per digit per side.
        for side in ("R", "L"):
            for digit in range(1, 6):
                hand = [(f"finger_{side}_{digit}_{seg}", pivots[f"finger_{side}_{digit}_{seg}"])
                        for seg in HAND_SEGMENTS
                        if pivots.get(f"finger_{side}_{digit}_{seg}") is not None]
                if hand:
                    n_hand += 1
                add(f"hand_{side}_{digit}", hand)

                foot = [(f"toe_{side}_{digit}_{seg}", pivots[f"toe_{side}_{digit}_{seg}"])
                        for seg in FOOT_SEGMENTS
                        if pivots.get(f"toe_{side}_{digit}_{seg}") is not None]
                if foot:
                    n_foot += 1
                add(f"foot_{side}_{digit}", foot)
    logger.info("Digit chains built: %d hand, %d foot", n_hand, n_foot)

    # Ribs: one pivot per rib, for rib-attached muscles and breathing.
    if rib_pivots:
        ribs = [(f"rib_{i}", pivot) for i, pivot in enumerate(rib_pivots)]
        add("ribs", ribs)
        logger.info("Rib skinning chain added: %d rib pivots", len(ribs))

    return chains
