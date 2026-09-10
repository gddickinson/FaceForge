"""One chain builder for the app and the headless tools, starting the arm at the clavicle.

The headless loader kept its own copy of the chain builder, which fell behind
the app's (arm chain from the shoulder instead of the clavicle and scapula).
Every headless render then bound the rotator cuff 85-100 % to the humerus and
the shipped footprints -- authored against clavicle and scapula joints --
silently failed to resolve.  These tests pin the shared builder's shape and
that both callers use it.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

from faceforge.coordination import asset_load_sequence, joint_chains
from faceforge.core.scene_graph import SceneNode


def _stub_rig():
    pivots = {}
    for side in "RL":
        for j in ("clavicle", "scapula", "shoulder", "elbow", "wrist", "hip", "knee", "ankle"):
            pivots[f"{j}_{side}"] = SceneNode(f"{j}_{side}")
        for digit in (1, 2):
            for seg in ("mc", "prox"):
                pivots[f"finger_{side}_{digit}_{seg}"] = SceneNode(f"finger_{side}_{digit}_{seg}")
            pivots[f"toe_{side}_{digit}_mt"] = SceneNode(f"toe_{side}_{digit}_mt")
    skeleton = SimpleNamespace(pivots={
        "thoracic": [{"level": i, "group": SceneNode(f"t{i}")} for i in range(3)],
        "lumbar": [{"level": i, "group": SceneNode(f"l{i}")} for i in range(2)],
    })
    ribs = [SceneNode(f"rib{i}") for i in range(4)]
    return skeleton, SimpleNamespace(pivots=pivots), ribs


def test_arm_chain_starts_at_the_clavicle_and_ids_follow_construction_order():
    skeleton, joint_setup, ribs = _stub_rig()
    ids: dict[str, int] = {}
    chains = joint_chains.build_joint_chains(skeleton, joint_setup, ribs, ids)
    assert ids["spine"] == 0
    arm = chains[ids["arm_R"]]
    assert [n for n, _ in arm] == ["clavicle_R", "scapula_R", "shoulder_R", "elbow_R", "wrist_R"]
    assert [n for n, _ in chains[ids["leg_L"]]] == ["hip_L", "knee_L", "ankle_L"]
    assert ids["hand_R_1"] < ids["foot_R_1"] < ids["ribs"]
    assert len(chains[ids["ribs"]]) == 4


def test_missing_pivots_are_skipped_not_fatal():
    skeleton, joint_setup, ribs = _stub_rig()
    del joint_setup.pivots["clavicle_L"]
    ids: dict[str, int] = {}
    chains = joint_chains.build_joint_chains(skeleton, joint_setup, None, ids)
    assert [n for n, _ in chains[ids["arm_L"]]][0] == "scapula_L"
    assert "ribs" not in ids


def test_the_app_and_the_headless_loader_both_use_the_shared_builder():
    app_src = inspect.getsource(asset_load_sequence.AssetLoadSequence.build_joint_chains)
    assert "build_joint_chains(" in app_src and "clavicle" not in app_src, \
        "the app must delegate rather than carry its own chain list"
    import importlib.util
    from pathlib import Path
    src = Path(__file__).resolve().parents[2] / "tools" / "headless_loader.py"
    text = src.read_text()
    assert "from faceforge.coordination.joint_chains import build_joint_chains" in text
    assert '("shoulder", "elbow", "wrist")' not in text, "the stale private copy is back"


def test_the_headless_loader_registers_muscles_through_the_shared_function():
    """Attachments, footprints and lever damping must not be app-only."""
    from pathlib import Path
    text = (Path(__file__).resolve().parents[2] / "tools" / "headless_loader.py").read_text()
    assert "register_muscle_layer(" in text
    assert "MuscleAttachmentSystem(" in text and "BoneCollisionSystem(" in text
