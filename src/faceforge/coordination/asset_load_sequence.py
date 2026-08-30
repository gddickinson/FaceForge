"""The startup load, as an explicit ordered sequence of named stages.

What this replaces
------------------
The whole-scene load used to be one ~640-line closure armed with
``QTimer.singleShot(100, load_assets)``.  Two things followed from that, and
both were problems:

* **It ran inside somebody else's event loop.**  Whichever ``processEvents()``
  happened to be executing when the 100 ms timer expired paid for the entire
  load.  That is how a GUI sweep once attributed 3.18 s of startup work to an
  unrelated checkbox: the checkbox was simply the control being exercised when
  the timer fired.
* **It was unobservable.**  There was no way to ask what had loaded, what
  stage was running, or which stage failed -- and no way to test the ordering,
  which is load-bearing (chains cannot be built before the pivots exist, and
  the rib chain cannot be built before body animation has collected the rib
  pivots).

:class:`AssetLoadSequence` makes the ordering data: :data:`LoadStage` names the
stages, :meth:`AssetLoadSequence.stage_order` returns them without running
anything, and :attr:`AssetLoadSequence.stage` reports where the sequence is.
An observer callback fires on every transition, so a caller (or a test, or a
timing harness) can attribute cost to the stage that incurred it rather than to
the interaction that happened to be on screen.

Deferral is still deliberate
----------------------------
The sequence is still armed on a timer by the caller: the GL widget has to have
been shown and given a context before meshes can be uploaded, and there is no
Qt signal for "the first paint has happened".  What has changed is that the
work now has a name and a state while it runs.

Failure policy
--------------
The two asset phases (head, body skeleton) record failures on
``LoadingPipeline.report`` and continue: a missing body STL must not cost the
user the head.  Every other stage is wiring, where an exception means a bug
rather than a missing file, so it propagates -- with the stage name attached by
:class:`StageFailure`.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Callable, Iterator, Sequence

from faceforge.body.body_animation import BodyAnimationSystem
from faceforge.body.body_constraints import BodyConstraints
from faceforge.body.soft_tissue import SoftTissueSkinning
from faceforge.core.scene_graph import SceneNode

logger = logging.getLogger(__name__)


class LoadStage(Enum):
    """The stages of the startup load, in execution order.

    Ordering constraints that the sequence exists to make explicit:

    * ``WIRE_HEAD`` after ``LOAD_HEAD`` -- it wires systems the head phase built.
    * ``REGISTER_SKELETON_VISIBILITY`` after ``LOAD_BODY_SKELETON`` -- bones are
      reparented under pivot nodes during the load, so both the original groups
      and the pivots have to be registered, and neither exists earlier.
    * ``BUILD_SKINNING`` after ``WIRE_BODY_ANIMATION`` -- the rib chain is built
      from the rib pivots that body animation collects.
    * ``BUILD_ATTACHMENT_SYSTEMS`` after ``BUILD_SKINNING`` -- both systems are
      installed onto ``simulation.soft_tissue``, which ``BUILD_SKINNING``
      creates.  They used to be built inside ``WIRE_ATTACHMENTS``, which runs
      earlier, so the ``soft_tissue is not None`` guard was always false and
      neither system was ever installed: muscles received no origin/insertion
      constraints and soft tissue never collided with bone.
    * ``ATTACH_DIAGNOSTICS`` after ``BUILD_SKINNING`` -- the debug tab's tooling
      is constructed around the skinning system.
    * ``APPLY_STARTUP_PRESET`` last -- a preset drives the UI through the event
      bus, so every handler it can reach must already be wired.
    """

    IDLE = "idle"
    LOAD_HEAD = "load_head"
    WIRE_HEAD = "wire_head"
    LOAD_BODY_SKELETON = "load_body_skeleton"
    REGISTER_SKELETON_VISIBILITY = "register_skeleton_visibility"
    WIRE_BODY_ANIMATION = "wire_body_animation"
    WIRE_ATTACHMENTS = "wire_attachments"
    BUILD_SKINNING = "build_skinning"
    BUILD_ATTACHMENT_SYSTEMS = "build_attachment_systems"
    ATTACH_DIAGNOSTICS = "attach_diagnostics"
    FINALISE = "finalise"
    APPLY_STARTUP_PRESET = "apply_startup_preset"
    COMPLETE = "complete"


class StageFailure(RuntimeError):
    """A wiring stage raised.  Carries the stage so the traceback names it."""

    def __init__(self, stage: LoadStage, cause: BaseException) -> None:
        super().__init__(f"asset load stage {stage.value!r} failed: {cause}")
        self.stage = stage
        self.cause = cause


#: Skeleton group key -> visibility toggle id.
SKELETON_TOGGLES = {
    "thoracic": "thoracic",
    "lumbar": "lumbar",
    "ribs": "ribs",
    "pelvis": "pelvis",
}

#: Limb visibility toggles.  Each toggle controls a *pivot*, not a bone group:
#: after ``setup_from_skeleton()`` the bones hang off the pivots, so the pivot
#: is what has to be hidden.  ``upper_limb_skel`` is the shoulder pivot because
#: that pivot contains the whole humerus -> elbow -> wrist chain.
LIMB_PIVOT_TOGGLES = {
    "upper_limb_skel": "shoulder",
    "lower_limb_skel": "hip",
    "hands_skel": "wrist",
    "feet_skel": "ankle",
}


class AssetLoadSequence:
    """Runs the startup load as an ordered, observable sequence of stages."""

    def __init__(self, ctx: Any, controllers: Any = None,
                 *, on_stage: Callable[[LoadStage], None] | None = None) -> None:
        self.ctx = ctx
        self.controllers = controllers
        self.on_stage = on_stage
        self.stage = LoadStage.IDLE
        #: Stages that have completed, in the order they completed.
        self.completed: list[LoadStage] = []
        #: Skinning system built by ``BUILD_SKINNING``.
        self.skinning: Any = None

    # -- Ordering ----------------------------------------------------------

    def steps(self) -> Sequence[tuple[LoadStage, Callable[[], None]]]:
        """The stages and their implementations, in execution order."""
        return (
            (LoadStage.LOAD_HEAD, self.load_head),
            (LoadStage.WIRE_HEAD, self.wire_head),
            (LoadStage.LOAD_BODY_SKELETON, self.load_body_skeleton),
            (LoadStage.REGISTER_SKELETON_VISIBILITY,
             self.register_skeleton_visibility),
            (LoadStage.WIRE_BODY_ANIMATION, self.wire_body_animation),
            (LoadStage.WIRE_ATTACHMENTS, self.wire_attachments),
            (LoadStage.BUILD_SKINNING, self.build_skinning),
            (LoadStage.BUILD_ATTACHMENT_SYSTEMS, self.build_attachment_systems),
            (LoadStage.ATTACH_DIAGNOSTICS, self.attach_diagnostics),
            (LoadStage.FINALISE, self.finalise),
            (LoadStage.APPLY_STARTUP_PRESET, self.apply_startup_preset),
        )

    def stage_order(self) -> list[LoadStage]:
        """Just the stage names, without running anything.

        This is what lets the ordering itself be asserted in a test.
        """
        return [stage for stage, _ in self.steps()]

    # -- Running -----------------------------------------------------------

    def run(self) -> LoadStage:
        """Run every stage in order.  Returns the final stage."""
        for stage, step in self.steps():
            self._enter(stage)
            try:
                step()
            except Exception as exc:  # noqa: BLE001 - re-raised, named
                logger.exception("Asset load stage %s failed", stage.value)
                raise StageFailure(stage, exc) from exc
            self.completed.append(stage)
        self._enter(LoadStage.COMPLETE)
        return self.stage

    def _enter(self, stage: LoadStage) -> None:
        self.stage = stage
        if self.on_stage is not None:
            self.on_stage(stage)

    # -- Stage: head --------------------------------------------------------

    def load_head(self) -> None:
        """Load the skull, face and facial systems.

        Asset-level failures are already recorded on the pipeline's report by
        its per-phase handlers.  Anything reaching here is a bug or a broken
        dependency, so the traceback is logged and the scene is marked degraded
        rather than continuing blind.
        """
        try:
            self.ctx.pipeline.load_head()
        except Exception as e:  # noqa: BLE001 - recorded, load continues
            logger.exception("Head loading incomplete")
            self.ctx.pipeline.report.record("load_head", e)

    def wire_head(self) -> None:
        """Give the simulation the head systems and scene groups it drives."""
        ctx = self.ctx
        sim = ctx.simulation
        pipeline = ctx.pipeline

        for attr in ("facs_engine", "jaw_muscles", "expression_muscles",
                     "face_features", "head_rotation", "neck_muscles",
                     "neck_constraints", "vertebrae_pivots"):
            setattr(sim, attr, getattr(pipeline, attr))

        sim.skull_group = ctx.node("skullGroup")
        sim.face_group = ctx.node("faceGroup")
        sim.brain_group = ctx.node("brainGroup")

        # Group references let the simulation skip expensive per-frame work
        # for groups that are hidden.
        sim.jaw_muscle_group = ctx.node("stlMuscleGroup")
        sim.expr_muscle_group = ctx.node("exprMuscleGroup")
        sim.platysma_group = ctx.node("platysmaGroup")
        sim.neck_muscle_group = ctx.node("neckMuscleGroup")
        sim.face_feature_group = ctx.node("faceFeatureGroup")

        skull_grp = ctx.node("skullGroup")
        if skull_grp is not None:
            from faceforge.anatomy.skull import get_jaw_pivot_node
            sim.jaw_pivot_node = get_jaw_pivot_node(skull_grp)

    # -- Stage: body skeleton ----------------------------------------------

    def load_body_skeleton(self) -> None:
        try:
            self.ctx.pipeline.load_body_skeleton()
        except Exception as e:  # noqa: BLE001 - recorded, load continues
            logger.exception("Body skeleton loading incomplete")
            self.ctx.pipeline.report.record("load_body_skeleton", e)

        # One place a caller can ask "is this scene complete?".
        report = self.ctx.pipeline.report
        if report.degraded:
            logger.warning("%s", report.summary())
        else:
            logger.info("%s", report.summary())

        # Surface the report in the UI.  Logging it is not surfacing it: a
        # partially loaded body looks exactly like a complete one on screen,
        # which is the defect the load-failure badge exists to fix.  This is
        # the only point where the finished report and the window are both in
        # scope.  getattr keeps the headless Session path -- which builds a
        # context with window=None -- working unchanged.
        window = getattr(self.ctx, "window", None)
        if window is not None and hasattr(window, "set_load_report"):
            try:
                # The report only -- no mesh counts.  ``set_load_report``
                # accepts loaded/expected to render "930 of 932 structures
                # loaded", but neither LoadReport nor the pipeline records how
                # many structures were REQUESTED: that number is derived from
                # the anatomy configs by tools/fetch_assets.py and is not
                # available here.  Passing a plausible constant would put an
                # unsourced figure in front of the user, so the badge shows
                # which subsystems degraded and stays silent about totals.
                window.set_load_report(report)
            except Exception:                                   # noqa: BLE001
                # A badge that fails must never take the scene down with it.
                logger.exception("Load-status badge could not be updated")

    def register_skeleton_visibility(self) -> None:
        """Register skeleton groups and limb pivots with the visibility manager.

        Both are registered because ``setup_from_skeleton()`` reparents bones
        under pivot nodes, which leaves the original groups (upper_limb,
        lower_limb, hand, foot) empty -- hiding an empty group hides nothing.
        """
        ctx = self.ctx
        pipeline = ctx.pipeline

        if pipeline.skeleton is not None:
            for group_key, toggle in SKELETON_TOGGLES.items():
                group = pipeline.skeleton.groups.get(group_key)
                if group is not None:
                    ctx.visibility.register(toggle, group)

        if pipeline.joint_setup is None:
            return
        pivots = pipeline.joint_setup.pivots
        for toggle, pivot_prefix in LIMB_PIVOT_TOGGLES.items():
            for side in ("R", "L"):
                pivot = pivots.get(f"{pivot_prefix}_{side}")
                if pivot is not None:
                    ctx.visibility.register(toggle, pivot)

    def wire_body_animation(self) -> None:
        """Build the body animation system and give it its pivots."""
        ctx = self.ctx
        pipeline = ctx.pipeline
        if pipeline.joint_setup is None:
            return

        body_anim = BodyAnimationSystem(pipeline.joint_setup)
        body_anim.load_fractions()
        if pipeline.skeleton is not None:
            body_anim.set_thoracic_pivots(
                pipeline.skeleton.pivots.get("thoracic", []))
            body_anim.set_lumbar_pivots(
                pipeline.skeleton.pivots.get("lumbar", []))
            if pipeline.skeleton.rib_nodes:
                body_anim.set_rib_nodes(pipeline.skeleton.rib_nodes)
                logger.info("Rib nodes wired: %d nodes for breathing",
                            len(pipeline.skeleton.rib_nodes))
        ctx.simulation.body_animation = body_anim

    def wire_attachments(self) -> None:
        """Wire bone anchors, fascia, attachment/collision systems, constraints.

        NOTE -- the muscle attachment and bone collision systems are guarded on
        ``simulation.soft_tissue``, which is not created until the next stage.
        In the original single-function load these blocks sat at exactly the
        same point relative to the skinning construction, so at startup the
        guard is false and neither system is built.  That is preserved here
        rather than quietly corrected: this refactor is required not to change
        behaviour, and the ordering defect is reported separately.
        """
        ctx = self.ctx
        sim = ctx.simulation
        pipeline = ctx.pipeline

        sim.bone_anchors = pipeline.bone_anchors
        sim.platysma = pipeline.platysma
        sim.fascia = pipeline.fascia

        # The attachment and collision systems are NOT built here: they install
        # onto sim.soft_tissue, which does not exist until BUILD_SKINNING runs
        # (the next stage).  See build_attachment_systems().

        # Rest positions for neck body-delta tracking.
        sim.init_neck_body_anchors()

        constraints = BodyConstraints()
        constraints.load()
        sim.body_constraints = constraints

    def build_attachment_systems(self) -> None:
        """Install the muscle-attachment and bone-collision systems.

        Both attach to ``simulation.soft_tissue``, so this must run after
        ``build_skinning``.  It previously ran inside ``wire_attachments``,
        one stage too early, where ``soft_tissue`` was still ``None`` -- so the
        guard never passed and both systems were silently absent for the whole
        session.  The consequences were not visible as an error: muscles simply
        deformed by skinning weights alone with no origin/insertion constraint,
        and soft tissue was free to pass through bone.

        Returning quietly when the anchors are missing is deliberate -- a body
        that loaded without a skeleton has nothing to attach to -- but the
        absence is logged, because "no attachments" should never again be
        indistinguishable from "attachments working".
        """
        ctx = self.ctx
        sim = ctx.simulation
        anchors = getattr(ctx.pipeline, "bone_anchors", None)
        soft = getattr(sim, "soft_tissue", None)

        if anchors is None or soft is None:
            logger.warning(
                "Muscle attachments and bone collision NOT installed "
                "(bone_anchors=%s, soft_tissue=%s); muscles will deform "
                "without origin/insertion constraints",
                "present" if anchors is not None else "missing",
                "present" if soft is not None else "missing",
            )
            return

        from faceforge.anatomy.muscle_attachments import MuscleAttachmentSystem
        soft.attachment_system = MuscleAttachmentSystem(anchors)

        from faceforge.anatomy.bone_collision import BoneCollisionSystem
        collision = BoneCollisionSystem(anchors)
        capsules = collision.build_capsules()
        if capsules > 0:
            soft.collision_system = collision
        logger.info("Attachment system installed; bone collision capsules: %d",
                    capsules)

    # -- Stage: skinning ----------------------------------------------------

    def build_skinning(self) -> None:
        """Create the skinning system and solve the skin against the chains."""
        ctx = self.ctx
        skinning = SoftTissueSkinning()
        ctx.simulation.soft_tissue = skinning
        self.skinning = skinning

        # The gender slider re-runs this on release, so the callable itself is
        # published on the context rather than only its result.
        ctx.joint_chain_builder = self.build_joint_chains

        chains = self.build_joint_chains()
        if not chains:
            return
        # Rest matrices must be current before the solve.
        ctx.scene.update()
        skinning.build_skin_joints(chains)
        logger.info("Skin joints built: %d joints in %d chains",
                    len(skinning.joints), len(chains))

    def build_joint_chains(self) -> list:
        """Build the kinematic chains, filling ``ctx.skin_chain_ids``.

        A *chain* is an ordered list of ``(name, node)`` joints that deform
        together.  Chain ids are assigned by construction order -- spine first,
        then limbs, then digits, then ribs -- and recorded by name in
        ``ctx.skin_chain_ids`` so the on-demand loaders can name the chains a
        structure follows without knowing the numbering.

        Rebuilt wholesale (rather than patched) when the skeleton is rescaled,
        which is why it is a method and not inline in :meth:`build_skinning`.
        """
        ctx = self.ctx
        pipeline = ctx.pipeline
        chain_ids = ctx.skin_chain_ids
        chains: list[list[tuple[str, SceneNode]]] = []
        chain_ids.clear()

        def add(name: str, chain: list[tuple[str, SceneNode]]) -> None:
            if chain:
                chain_ids[name] = len(chains)
                chains.append(chain)

        # Spine: thoracic top -> bottom, then lumbar.
        spine: list[tuple[str, SceneNode]] = []
        if pipeline.skeleton is not None:
            for region in ("thoracic", "lumbar"):
                for pinfo in pipeline.skeleton.pivots.get(region, []):
                    spine.append(
                        (f"{region}_{pinfo.get('level', 0)}", pinfo["group"]))
        add("spine", spine)

        # Limbs: one 3-joint chain per limb.
        n_hand = n_foot = 0
        if pipeline.joint_setup is not None:
            pivots = pipeline.joint_setup.pivots
            for side in ("R", "L"):
                for name, joints in (("arm", ("shoulder", "elbow", "wrist")),
                                     ("leg", ("hip", "knee", "ankle"))):
                    chain = [(f"{j}_{side}", pivots[f"{j}_{side}"])
                             for j in joints if pivots.get(f"{j}_{side}") is not None]
                    add(f"{name}_{side}", chain)

            # Digits: one chain per digit per side.
            for side in ("R", "L"):
                for digit in range(1, 6):
                    hand = [(f"finger_{side}_{digit}_{seg}",
                             pivots[f"finger_{side}_{digit}_{seg}"])
                            for seg in ("mc", "prox", "mid", "dist")
                            if pivots.get(f"finger_{side}_{digit}_{seg}") is not None]
                    if hand:
                        n_hand += 1
                    add(f"hand_{side}_{digit}", hand)

                    foot = [(f"toe_{side}_{digit}_{seg}",
                             pivots[f"toe_{side}_{digit}_{seg}"])
                            for seg in ("mt", "prox", "mid", "dist")
                            if pivots.get(f"toe_{side}_{digit}_{seg}") is not None]
                    if foot:
                        n_foot += 1
                    add(f"foot_{side}_{digit}", foot)
        logger.info("Digit chains built: %d hand, %d foot", n_hand, n_foot)

        # Ribs: one pivot per rib, for rib-attached muscles and breathing.
        body_anim = getattr(ctx.simulation, "body_animation", None)
        if body_anim is not None and body_anim._rib_pivots:
            ribs = [(f"rib_{i}", pivot)
                    for i, pivot in enumerate(body_anim._rib_pivots)]
            add("ribs", ribs)
            if ribs:
                logger.info("Rib skinning chain added: %d rib pivots", len(ribs))

        return chains

    # -- Stage: diagnostics -------------------------------------------------

    def attach_diagnostics(self) -> None:
        """Hand the skinning system to the debug tab's tooling."""
        diagnostics = getattr(self.controllers, "diagnostics", None)
        if diagnostics is not None and self.skinning is not None:
            diagnostics.attach(self.skinning)

    # -- Stage: finalise ----------------------------------------------------

    def finalise(self) -> None:
        """Everything that needs the finished scene: search index, overlays."""
        ctx = self.ctx
        self._wire_eye_tracking_cursor()
        ctx.simulation.anim_player = ctx.anim_player
        ctx.control_panel.display_tab.set_animation_clips(
            list(ctx.builtin_clips.keys()))

        meshes = ctx.scene.collect_meshes()
        print(f"[FaceForge] Scene has {len(meshes)} renderable meshes")
        for mesh, _ in meshes[:5]:
            geom = mesh.geometry
            print(f"  - {mesh.name}: {geom.vertex_count} verts, "
                  f"{'indexed' if geom.has_indices else 'non-indexed'}, "
                  f"mode={mesh.material.render_mode.name}")

        mesh_names = [mesh.name for mesh, _ in meshes if mesh.name]
        ctx.search_index.build_from_names(mesh_names)
        print(f"[FaceForge] Search index built: "
              f"{len(ctx.search_index._entries)} entries")
        ctx.control_panel.layers_tab.set_pathology_targets(mesh_names)

        # Pathology can affect anything; the heatmap only muscles.
        for mesh, _ in meshes:
            if not mesh.name:
                continue
            ctx.pathology.register_mesh(mesh, mesh.name)
            if "muscle" in mesh.name.lower() or "Muscle" in mesh.name:
                ctx.muscle_activation.register_muscle(mesh, mesh.name)

    def _wire_eye_tracking_cursor(self) -> None:
        """Feed normalised cursor position to eye tracking.

        Normalised to [-1, 1] with Y inverted, because the eyes are aimed in
        view space and Qt's Y axis points down.
        """
        ctx = self.ctx
        gl = ctx.gl_widget

        def on_mouse_move(x: float, y: float) -> None:
            width, height = gl.width(), gl.height()
            if width > 0 and height > 0:
                ctx.simulation.eye_tracking.set_cursor_position(
                    (x / width) * 2.0 - 1.0,
                    -((y / height) * 2.0 - 1.0),
                )

        gl.mouse_move_callback = on_mouse_move

    # -- Stage: startup preset ---------------------------------------------

    def apply_startup_preset(self) -> None:
        """Apply whichever preset the startup dialog selected.

        Last, because a preset drives the UI through the event bus and through
        the layers tab: every handler it can reach has to be wired and the
        scene has to be built, or the preset silently half-applies.
        """
        ctx = self.ctx
        panel = ctx.control_panel
        if ctx.startup_illustration:
            from faceforge.ui.illustration_presets import apply_illustration_preset
            apply_illustration_preset(
                ctx.startup_illustration, panel.layers_tab, ctx.event_bus,
                ctx.gl_widget, panel.display_tab, ctx.label_overlay, ctx.scene,
            )
            print("[FaceForge] Applied illustration preset: "
                  f"{ctx.startup_illustration}")
        elif ctx.startup_preset and ctx.startup_preset != "Default":
            from faceforge.ui.startup_dialog import apply_preset
            apply_preset(ctx.startup_preset, panel.layers_tab, ctx.event_bus,
                         gl_widget=ctx.gl_widget)
            print(f"[FaceForge] Applied startup preset: {ctx.startup_preset}")

    def __iter__(self) -> Iterator[LoadStage]:
        """Iterating the sequence yields its stage order."""
        return iter(self.stage_order())
