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
        bus.subscribe(EventType.SKELETON_FIT_TOGGLED, self.on_skeleton_fit_toggled)

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
        """Slider release: scale the skeleton, then rebuild the soft tissue on it.

        The order matters and each step is there for a measured reason.

        1. The skeleton is scaled as an articulated hierarchy, so the joints
           move and the articulations stay shut
           (:mod:`faceforge.body.skeleton_morph`).
        2. The soft tissue's *rest* pose is rebuilt from how far those joints
           moved, plus the muscle and soft-tissue sex differences
           (:mod:`faceforge.body.soft_tissue_morph`).  It is deliberately not
           routed through the skinning: joints that translate rather than
           rotate tear the skin at every chain boundary, measured at 27747
           over-stretched edges on the skin alone.
        3. Only then is the skinning re-bound, which makes the morphed body
           the pose-neutral body and leaves joint animation working from there.
        """
        state = self.ctx.state
        state.body.gender = gender
        state.target_body.gender = gender
        morph = getattr(self.ctx.pipeline, "gender_morph", None)
        if morph is None or not morph.loaded:
            return
        morph.set_gender(gender)

        root = self.ctx.node("bodyRoot")
        if root is None:
            return
        joint_setup = getattr(getattr(self.ctx, "pipeline", None), "joint_setup", None)
        skinning = getattr(self.ctx.simulation, "soft_tissue", None)
        soft = {id(b.mesh) for b in getattr(skinning, "bindings", ())
                if getattr(b, "mesh", None) is not None}
        # The fit comes off first.  Both it and the sex morph rewrite the same
        # rest geometry from their own captured original, so scaling on top of
        # a fitted skeleton would leave the fit's snapshot describing a body
        # that no longer exists -- and taking the fit off afterwards would then
        # undo the scaling with it.
        fit = getattr(morph, "skeleton_fit", None)
        was_fitted = fit is not None and fit.applied
        if fit is not None:
            fit.reset(root, getattr(joint_setup, "joint_positions", None))
        n_scaled = morph.scale_skeleton(
            root, getattr(joint_setup, "joint_positions", None), exclude=soft)
        root.update_world_matrix(force=True)
        self.ctx.scene.update()

        warps = self.skeleton_warps(morph, root, gender, joint_setup, soft,
                                    fit_to_mesh=was_fitted)
        root.update_world_matrix(force=True)
        self.ctx.scene.update()
        stats = self.rebuild_soft_tissue(gender, morph, *warps)
        logger.info("Gender %.2f: %d bones scaled, %d muscles and %d other meshes rebuilt",
                    gender, n_scaled, stats.get("muscles", 0), stats.get("other", 0))

        self.resnapshot_skinning()
        self.refresh_after_skeleton_change()

    # -- Fitting the skeleton into the body-surface mesh --------------------

    def on_skeleton_fit_toggled(self, enabled: bool = False, **kw) -> None:
        """Put the skeleton inside the body-surface mesh, or take it back out.

        The surface mesh is a MakeHuman figure and the skeleton a
        BodyParts3D cadaver, so one of the two has to give.  Warping the
        surface onto the bones is measured in ``docs/sex_morph.md`` and it
        damages the mesh, which is why it is off; this moves the *bones*
        instead, region by region, and leaves the authored surface exactly as
        its authors drew it (:mod:`faceforge.body.skeleton_fit`).

        Everything after the move is the gender path's, for the same reasons:
        the soft tissue is carried by a smooth field rather than through the
        skinning, and only then is the result made the pose-neutral body.
        """
        morph = getattr(self.ctx.pipeline, "gender_morph", None)
        root = self.ctx.node("bodyRoot")
        if morph is None or not morph.loaded or root is None:
            return
        fit = getattr(morph, "skeleton_fit", None)
        if fit is None or bool(enabled) == fit.applied:
            return
        joint_setup = getattr(getattr(self.ctx, "pipeline", None), "joint_setup", None)
        skinning = getattr(self.ctx.simulation, "soft_tissue", None)
        soft = {id(b.mesh) for b in getattr(skinning, "bindings", ())
                if getattr(b, "mesh", None) is not None}
        gender = float(getattr(self.ctx.state.body, "gender", 0.0))

        warps = self.skeleton_warps(morph, root, gender, joint_setup, soft,
                                    fit_to_mesh=bool(enabled))
        root.update_world_matrix(force=True)
        self.ctx.scene.update()
        self.rebuild_soft_tissue(gender, morph, *warps)
        self.resnapshot_skinning()
        self.refresh_after_skeleton_change()
        logger.info("Skeleton fit to the body mesh: %s", "on" if enabled else "off")

    def skeleton_warps(self, morph: Any, root: Any, gender: float,
                       joint_setup: Any, soft: set[int],
                       fit_to_mesh: bool | None = None) -> tuple[Any, Any]:
        """Move the skeleton, and return the two fields that moved it.

        Two changes can be in force at once -- the sex morph and the fit into
        the body-surface mesh -- and each was measured on the skeleton the
        other had not yet moved.  So the sex morph's field is read with the
        fit taken off, the fit is then re-applied, and the two are returned
        separately because they do not apply to the same meshes.
        """
        joints = getattr(joint_setup, "joint_positions", None)
        fit = getattr(morph, "skeleton_fit", None)
        if fit_to_mesh is None:
            fit_to_mesh = fit is not None and fit.applied
        if fit is not None:
            fit.reset(root, joints)
        skeleton = getattr(morph, "skeleton_morph", None)
        gender_warp = (skeleton.displacement_field(root)
                       if skeleton is not None and skeleton.captured else None)
        fit_warp = None
        if fit is not None and fit_to_mesh:
            fit.apply(root, 1.0, gender, joints, exclude=soft)
            fit_warp = fit.displacement_field()
        return gender_warp, fit_warp

    def rebuild_soft_tissue(self, gender: float, morph: Any,
                            gender_warp: Any, fit_warp: Any) -> dict:
        """Carry the soft tissue onto the skeleton as it now is.

        Everything that hangs off a bone follows both changes, chained rather
        than added: the fit was measured on a skeleton the sex morph had
        already scaled.

        The skin that came with the skeleton is the exception.  It was scanned
        from these bones and already fits them; dragging it onto the surface
        mesh's proportions is the distortion the whole option exists to avoid,
        and it is visible -- the fitted skin comes out a head shorter and
        broad in the shoulders.  So it follows the sex morph and nothing else,
        and the bones simply sit a little further inside it.
        """
        from faceforge.body.skeleton_field import compose
        from faceforge.body.skeleton_fit import is_own_skin

        skinning = getattr(self.ctx.simulation, "soft_tissue", None)
        bindings = list(getattr(skinning, "bindings", ()))
        if fit_warp is None:
            return self.morph_soft_tissue(gender, morph,
                                          compose(gender_warp, fit_warp),
                                          bindings)
        own, carried = [], []
        for binding in bindings:
            (own if is_own_skin(getattr(binding, "mesh", None))
             else carried).append(binding)
        stats = self.morph_soft_tissue(
            gender, morph, compose(gender_warp, fit_warp), carried)
        # With no sex morph in force there is nothing for the skeleton's own
        # skin to be rebuilt *from*: the second pass would copy 791,729
        # vertices back onto themselves, and build the sex field to do it.
        if own and (gender_warp is not None or gender > 0.0):
            extra = self.morph_soft_tissue(gender, morph, gender_warp, own)
            stats = {k: stats.get(k, 0) + extra.get(k, 0)
                     for k in set(stats) | set(extra)}
        return stats

    def resnapshot_skinning(self) -> None:
        """Make the morphed body the pose-neutral body.

        Only the joints' rest transforms and the caches derived from them are
        refreshed.  Which bone a vertex follows does not change when the body
        changes size, and re-solving that costs 85 seconds against 0.4 for the
        re-snapshot.  If the joint list itself came back different the
        assignment really would be stale, and the full re-registration runs.
        """
        skinning = getattr(self.ctx.simulation, "soft_tissue", None)
        if skinning is None:
            return
        builder = self.ctx.joint_chain_builder
        chains = builder() if builder else []
        if not chains:
            return
        resnapshot = getattr(skinning, "resnapshot_rest", None)
        if resnapshot is not None and resnapshot(chains):
            logger.info("Skinning re-snapshotted: %d bindings kept",
                        len(skinning.bindings))
            return
        logger.warning("The joint list changed during the morph; re-solving "
                       "every binding, which is slow but correct")
        self.rebind_skinning()

    def morph_soft_tissue(self, gender: float, morph: Any, warp: Any,
                          bindings: Any = None) -> dict:
        """Rebuild a muscle's or skin mesh's rest pose for ``gender``.

        ``bindings`` defaults to every bound mesh; the caller passes a subset
        when two groups need different fields (see :meth:`rebuild_soft_tissue`).
        """
        skinning = getattr(self.ctx.simulation, "soft_tissue", None)
        if skinning is None:
            return {}
        from faceforge.body.soft_tissue_morph import SoftTissueMorph
        tissue = getattr(self, "_soft_tissue_morph", None)
        if tissue is None:
            tissue = self._soft_tissue_morph = SoftTissueMorph()
        # A muscle's region is read from the kinematic chain most of its
        # vertices bind to, so nothing has to be named twice.
        ids = getattr(self.ctx, "skin_chain_ids", None) or {}
        chain_names = {int(v): k for k, v in ids.items()} or None
        if bindings is None:
            bindings = getattr(skinning, "bindings", ())
        return tissue.apply(
            bindings, gender, warp=warp,
            skin_field=getattr(morph, "skin_shape", None),
            chain_names=chain_names,
            chain_of_joint=getattr(skinning, "_joint_chain_ids", None))

    def refresh_after_skeleton_change(self) -> None:
        """Invalidate what was measured against the old skeleton.

        The skinning early-exits on a signature made of joint DOFs, none of
        which a sex change touches, so without clearing it the body would keep
        the pose it was last deformed into.  The bone collision capsules and
        the shoulder-girdle rest geometry were both measured from bone
        vertices that have just moved.
        """
        skinning = getattr(self.ctx.simulation, "soft_tissue", None)
        if skinning is not None:
            skinning._last_signature = ()
            collision = getattr(skinning, "collision_system", None)
            if collision is not None:
                try:
                    collision.build_capsules()
                except Exception as exc:                     # noqa: BLE001 - logged
                    logger.warning("Bone capsules not rebuilt after morph: %s", exc)
            # A footprinted muscle is placed from its fibre field, and the
            # field holds the rest pose it was solved on.  The morph has just
            # rewritten that pose, so without this the field writes the old
            # geometry back over the new one every frame -- visible with the
            # skeleton fit on as the trapezius and deltoid in wings.
            attachments = getattr(skinning, "attachment_system", None)
            if attachments is not None:
                try:
                    attachments.refresh_rest_poses(
                        getattr(skinning, "bindings", ()))
                except Exception as exc:                     # noqa: BLE001 - logged
                    logger.warning("Fibre fields not refreshed after morph: %s",
                                   exc)
        anim = getattr(self.ctx.simulation, "body_animation", None)
        if anim is not None and hasattr(anim, "_girdle_cache"):
            anim._girdle_cache = {}

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
