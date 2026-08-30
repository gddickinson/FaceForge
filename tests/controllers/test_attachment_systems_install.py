"""The attachment and collision systems must actually get installed.

Both systems were built inside ``wire_attachments`` (load stage 6) and installed
onto ``simulation.soft_tissue``, which ``build_skinning`` (stage 7) creates.  The
guard ``sim.soft_tissue is not None`` was therefore always false, and neither
system was installed in any session: muscles deformed by skinning weights with
no origin/insertion constraint, and soft tissue passed through bone.

Nothing raised.  The suite was green.  That is why these tests assert on the
INSTALLED OBJECTS and on the stage ordering invariant, not on the code path.
"""

from __future__ import annotations

from types import SimpleNamespace

from faceforge.coordination.asset_load_sequence import AssetLoadSequence, LoadStage


class _Anchors:
    """Minimal stand-in for BoneAnchorRegistry.

    ``BoneCollisionSystem.build_capsules`` interrogates the registry per bone,
    so a double that only lists names is not enough -- it must answer
    ``has_bone``.  Returning False for every bone is deliberate: it exercises
    the "no capsules built" branch without needing real skeleton geometry, and
    the attachment system (the thing this module is really about) installs
    regardless.
    """

    def has_bone(self, name: str) -> bool:
        return False

    def bone_names(self):
        return ["humerus_l", "femur_r"]


class _Soft:
    """Minimal stand-in for SoftTissueSkinning."""

    def __init__(self):
        self.attachment_system = None
        self.collision_system = None


def _ctx(anchors=None, soft=None):
    return SimpleNamespace(
        pipeline=SimpleNamespace(bone_anchors=anchors),
        simulation=SimpleNamespace(soft_tissue=soft),
    )


def test_the_stage_runs_after_skinning_creates_soft_tissue():
    """The ordering invariant whose violation caused the defect.

    Asserting the stage exists is not enough: it has to come after the stage
    that creates the object it installs onto.
    """
    order = AssetLoadSequence(ctx=None).stage_order()
    assert LoadStage.BUILD_ATTACHMENT_SYSTEMS in order
    assert order.index(LoadStage.BUILD_ATTACHMENT_SYSTEMS) > \
        order.index(LoadStage.BUILD_SKINNING), (
            "attachment systems install onto simulation.soft_tissue, which "
            "BUILD_SKINNING creates -- running earlier silently installs nothing"
        )


def test_both_systems_are_installed_when_prerequisites_exist():
    soft = _Soft()
    seq = AssetLoadSequence(ctx=_ctx(anchors=_Anchors(), soft=soft))
    seq.build_attachment_systems()

    assert soft.attachment_system is not None, (
        "muscle attachment system was not installed: muscles would deform "
        "without any origin/insertion constraint"
    )
    assert type(soft.attachment_system).__name__ == "MuscleAttachmentSystem"


def test_missing_prerequisites_are_logged_not_silent(caplog):
    """A body with no skeleton has nothing to attach to -- but say so.

    The original defect was invisible precisely because the guard failed
    quietly, so absence must be distinguishable from success in the log.
    """
    seq = AssetLoadSequence(ctx=_ctx(anchors=None, soft=_Soft()))
    with caplog.at_level("WARNING"):
        seq.build_attachment_systems()
    assert any("NOT installed" in r.message or "NOT installed" in r.getMessage()
               for r in caplog.records), \
        "missing prerequisites must produce a warning, not silence"


def test_wire_attachments_no_longer_tries_to_install_them():
    """Guards against the construction drifting back into the early stage."""
    import inspect

    src = inspect.getsource(AssetLoadSequence.wire_attachments)
    assert "MuscleAttachmentSystem(" not in src, (
        "attachment construction is back in wire_attachments, which runs "
        "before soft_tissue exists"
    )
    assert "BoneCollisionSystem(" not in src
