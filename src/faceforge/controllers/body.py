"""Body: joint pose, pose presets, and sexual dimorphism morphing.

Gender is two events, not one, and the split is a responsiveness decision.
``GENDER_CHANGED`` fires continuously while the slider is dragged and does the
cheap thing: morph the body surface only.  ``GENDER_RELEASED`` fires once, on
release, and does the expensive thing: scale every bone mesh, rebuild the
kinematic chains and re-register every skinned mesh against them.  Doing the
expensive path per drag event would freeze the render loop for the length of
the drag.
"""

from __future__ import annotations

import logging
from typing import Any

from faceforge.core.events import EventType

logger = logging.getLogger(__name__)


class BodyController:
    """Handlers for body pose, pose presets and the gender morph."""

    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx

    def subscribe(self) -> None:
        bus = self.ctx.event_bus
        bus.subscribe(EventType.BODY_STATE_CHANGED, self.on_body_changed)
        bus.subscribe(EventType.BODY_POSE_SET, self.on_body_pose_set)
        bus.subscribe(EventType.GENDER_CHANGED, self.on_gender_changed)
        bus.subscribe(EventType.GENDER_RELEASED, self.on_gender_released)

    # -- Pose --------------------------------------------------------------

    def on_body_changed(self, field: str = "", value: float = 0.0, **kw) -> None:
        """Write one body DOF, translating the JS field name if needed.

        A value that arrives as a genuine ``bool`` is also written to live
        state, so a checkbox takes effect this frame rather than on the next
        simulation step.  The test is on the stored value's type, not on
        whether the field is in ``FLAG_FIELDS``: a flag sent as ``1.0`` reaches
        the target only, and the interpolator copies it across on the next
        step (it copies ``FLAG_FIELDS`` rather than lerping them).
        """
        if not field:
            return
        state = self.ctx.state
        py_field = state.target_body._JS_KEY_MAP.get(field, field)
        if not hasattr(state.target_body, py_field):
            return
        setattr(state.target_body, py_field, value)
        if isinstance(getattr(state.target_body, py_field), bool):
            setattr(state.body, py_field, bool(value))

    def on_body_pose_set(self, name: str = "", values: dict | None = None,
                         **kw) -> None:
        if values:
            self.ctx.state.target_body.set_from_js_dict(values)

    # -- Gender morph ------------------------------------------------------

    def on_gender_changed(self, gender: float = 0.0, **kw) -> None:
        """Live slider drag: morph the body surface only (cheap).

        ``gender`` is a ``LIVE_ONLY_FIELD``, so the interpolator never touches
        it and the live value is authoritative.  ``target_body`` is kept in
        step anyway so the two ``BodyState`` objects never disagree about a
        setting the user can see.
        """
        state = self.ctx.state
        state.body.gender = gender
        state.target_body.gender = gender
        morph = getattr(self.ctx.pipeline, "gender_morph", None)
        if morph is not None and morph.loaded:
            morph.set_gender(gender)

    def on_gender_released(self, gender: float = 0.0, **kw) -> None:
        """Slider release: bone scaling, then chain rebuild and re-registration."""
        state = self.ctx.state
        state.body.gender = gender
        state.target_body.gender = gender
        morph = getattr(self.ctx.pipeline, "gender_morph", None)
        if morph is None or not morph.loaded:
            return
        morph.set_gender(gender)

        bone_meshes = self.collect_bone_meshes()
        if bone_meshes:
            n_scaled = morph.scale_skeleton(bone_meshes)
            logger.info("Gender %.2f: scaled %d/%d bone meshes",
                        gender, n_scaled, len(bone_meshes))
            for _, mesh in bone_meshes:
                mesh.store_rest_pose()

        # setup_from_skeleton() is deliberately NOT re-run.  Pivots were built
        # during the initial load and the bones are already reparented under
        # them, so a second call would fail looking for bones in their original
        # skeleton groups.  Scaling above updated vertex positions in place,
        # which is what the pivots need.

        self.rebind_skinning()

    def collect_bone_meshes(self) -> list[tuple[str, Any]]:
        """Every bone mesh in the ``bodyRoot`` subtree, as ``(name, mesh)``.

        Walks the whole subtree rather than the skeleton's own groups because
        ``setup_from_skeleton()`` reparents bones under pivot nodes, so the
        original groups are no longer where the bones live.  The body surface
        mesh is excluded by name: it is skin, not bone, and scaling it here
        would fight the surface morph.
        """
        out: list[tuple[str, Any]] = []
        body_root = self.ctx.node("bodyRoot")
        if body_root is None:
            return out

        def walk(node: Any) -> None:
            if node.mesh is not None and node.name and node.name != "body_surface":
                out.append((node.name, node.mesh))
            for child in node.children:
                walk(child)

        walk(body_root)
        return out

    def rebind_skinning(self) -> None:
        """Rebuild the skin joints for the new skeleton and re-register meshes.

        Bone scaling moved every joint, so the rest matrices the skinning was
        solved against are stale.  Chains are rebuilt from the (unchanged)
        scene hierarchy and every previously registered mesh is re-registered
        against the new joints; a mesh that fails to re-register is warned
        about rather than dropping the whole rebuild.
        """
        skinning = getattr(self.ctx.simulation, "soft_tissue", None)
        if skinning is None:
            return
        body_root = self.ctx.node("bodyRoot")
        if body_root is not None:
            body_root.update_world_matrix(force=True)
        self.ctx.scene.update()

        builder = self.ctx.joint_chain_builder
        new_chains = builder() if builder else []
        if not new_chains:
            return

        old_bindings = list(skinning.bindings)
        skinning.clear_bindings()
        skinning.rebuild_skin_joints(new_chains)

        failed: list[str] = []
        for binding in old_bindings:
            try:
                # Rebind exactly as the mesh was originally bound.  Passing
                # only is_muscle/muscle_name here dropped allowed_chains,
                # spatial_limit, chain_z_margin and head_follow_config, so a
                # gender change re-solved every mesh against ALL chains: a
                # torso mesh constrained to the spine would then bind to the
                # arm chain (measured: 111 of 140 vertices on a shoulder-region
                # mesh), and torso geometry visibly followed arm motion.
                skinning.register_skin_mesh(
                    binding.mesh, **binding.rebind_kwargs()
                )
            except Exception as e:  # noqa: BLE001 - one mesh must not stop the rest
                failed.append(binding.mesh.name)
                logger.warning("Re-registration failed for %s: %s",
                               binding.mesh.name, e)

        # "One mesh must not stop the rest" is the right policy for a bad mesh;
        # it is the wrong policy for a bad interface.  When EVERY mesh fails the
        # cause is systemic (a changed signature, a missing attribute) and the
        # result is a body with no skinning at all -- which renders as a frozen
        # figure, not as an error.  Logging one warning per mesh made that
        # indistinguishable from two unlucky meshes, so say it once, loudly.
        if failed and len(failed) == len(old_bindings):
            logger.error(
                "Gender re-registration re-bound NOTHING: all %d meshes "
                "failed (first: %s). The body is now unskinned; this is a "
                "systemic failure, not a per-mesh one.",
                len(failed), failed[0],
            )
        elif failed:
            logger.warning("Gender re-registration: %d of %d meshes failed",
                           len(failed), len(old_bindings))

        logger.info("Gender re-registration complete: %d meshes",
                    len(skinning.bindings))
