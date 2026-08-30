"""The debug tab: skinning diagnostics, vertex selection and chain overrides.

This is the workbench for the hardest thing in the application to get right --
which kinematic chain each skin vertex follows.  Automatic assignment is
spatial and therefore wrong in the places where two chains meet, so the tab
provides: a stretch/chain colour visualisation to *see* the bad vertices, a
viewport selection tool to pick them, a reassignment control to move them to
the right chain, and persistence so the corrections survive a restart.

Attachment order
----------------
:meth:`DiagnosticsController.attach` runs after the skinning system exists,
which is why it is a step in the asset load sequence and not part of the
startup wiring.  Its two halves are deliberately separate: the region-label
half is wrapped in its own try/except and *replaces* three signal connections
made by the first half.  If region labels are unavailable, the simpler
selection handlers stay connected and the tab still works.  That is why the
connections are made twice rather than once with the final handler -- the
fallback is the point.

Overrides
---------
Saved vertex overrides cannot be applied at startup: the meshes they refer to
belong to layers that load on demand, so at startup there is usually nothing to
apply them to.  They are instead applied through a post-registration hook, so
whenever a layer registers meshes with skinning the pending overrides get
another chance to bind.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class DiagnosticsController:
    """Owns the debug tab's visualisation, selection and override tooling."""

    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx
        self.skinning: Any = None
        self.stretch_viz: Any = None
        self.reassigner: Any = None
        self.selection_tool: Any = None
        #: binding index -> vertex indices the user has reassigned.
        self.modified_vertices: dict[int, set[int]] = {}
        #: Overrides loaded from disk that have not found their meshes yet.
        self.pending_overrides: Any = None
        #: Region-label state (only populated if region labels are available).
        self.region_color_cache: dict[int, bytes] = {}
        self.region_viz_active = False
        self.region_modified: dict[int, int] = {}

    @property
    def debug_tab(self) -> Any:
        return self.ctx.control_panel.debug_tab

    # -- Attachment --------------------------------------------------------

    def attach(self, skinning: Any) -> None:
        """Build the tooling around *skinning* and wire the debug tab."""
        from faceforge.body.chain_reassignment import ChainReassigner
        from faceforge.body.stretch_viz import StretchVisualizer
        from faceforge.rendering.selection_tool import SelectionTool

        self.skinning = skinning
        self.stretch_viz = StretchVisualizer(skinning)
        self.reassigner = ChainReassigner(skinning)
        self.selection_tool = SelectionTool(skinning)
        self.ctx.gl_widget.selection_tool = self.selection_tool
        print(f"[DEBUG] Created StretchVisualizer, ChainReassigner, SelectionTool"
              f" (skinning id={id(skinning)}, bindings={len(skinning.bindings)})")

        self.debug_tab.set_diagnostic_callback(self.run_skinning_diagnostic)
        self.populate_chain_names()
        self.connect_visualisation()
        self.connect_selection()
        self.connect_overrides()
        self.install_viz_update_hook()
        self.attach_region_labels()

    def populate_chain_names(self) -> None:
        """Fill the chain selector with ``"<id>: <name>"`` entries.

        The friendly name comes from the first joint in each chain with its
        trailing segment index stripped, so ``shoulder_R`` names the arm chain
        rather than ``shoulder_R``/``elbow_R``/``wrist_R`` all appearing.
        """
        names: list[str] = []
        seen: set[int] = set()
        for joint in self.skinning.joints:
            if joint.chain_id in seen:
                continue
            seen.add(joint.chain_id)
            name = (joint.name.rsplit("_", 1)[0]
                    if "_" in joint.name else joint.name)
            names.append(f"{joint.chain_id}: {name}")
        self.debug_tab.set_chain_names(names)

    @staticmethod
    def parse_leading_id(label: str) -> int:
        """``"3: arm_R"`` -> ``3``.  Malformed labels fall back to chain 0."""
        try:
            return int(label.split(":")[0])
        except (ValueError, IndexError):
            return 0

    # -- Skinning diagnostic ----------------------------------------------

    def run_skinning_diagnostic(self) -> str:
        """Run every skinning check and format one report for the tab."""
        from faceforge.body.diagnostics import SkinningDiagnostic

        skinning = getattr(self.ctx.simulation, "soft_tissue", None)
        if skinning is None or not skinning.bindings:
            return "No soft tissue bindings registered."
        diag = SkinningDiagnostic(skinning)
        return diag.format_report(
            diag.analyze_bindings(),
            diag.check_displacements(max_displacement=5.0, relative=True),
            distortion=diag.check_mesh_distortion(),
            static_verts=diag.check_static_vertices(),
            neighbor_stretch=diag.check_neighbor_stretch(max_stretch=3.0),
        )

    # -- Stretch / chain visualisation ------------------------------------

    def connect_visualisation(self) -> None:
        tab = self.debug_tab
        tab.stretch_viz_toggled.connect(self.on_stretch_viz)
        tab.chain_viz_toggled.connect(self.on_chain_viz)
        print("[DEBUG] Connected stretch_viz_toggled and chain_viz_toggled signals")
        # Keep references on the tab so the bound methods are not collected.
        tab._viz_handler_stretch = self.on_stretch_viz
        tab._viz_handler_chain = self.on_chain_viz

    def on_stretch_viz(self, enabled: bool) -> None:
        """Toggle stretch colouring.

        Turning it *off* also turns chain colouring off: the two write the same
        vertex colour buffer, so "off" has to mean no colouring at all rather
        than falling back to whatever the other mode last left behind.  Turning
        it *on* applies the colours immediately instead of waiting for the next
        simulation tick, so a static pose still responds to the checkbox.
        """
        print(f"[DEBUG] _on_stretch_viz({enabled}), "
              f"bindings={len(self.skinning.bindings)}")
        viz = self.stretch_viz
        viz.stretch_enabled = enabled
        if not enabled:
            viz.chain_enabled = False
        else:
            viz.update()
        self._report_viz_state()

    def on_chain_viz(self, enabled: bool) -> None:
        """Toggle chain-id colouring.  See :meth:`on_stretch_viz`."""
        print(f"[DEBUG] _on_chain_viz({enabled}), "
              f"bindings={len(self.skinning.bindings)}")
        viz = self.stretch_viz
        viz.chain_enabled = enabled
        if not enabled:
            viz.stretch_enabled = False
        else:
            viz.update()
        self._report_viz_state()

    def _report_viz_state(self) -> None:
        print(f"[DEBUG]   stretch_enabled={self.stretch_viz.stretch_enabled}, "
              f"chain_enabled={self.stretch_viz.chain_enabled}")

    def install_viz_update_hook(self) -> None:
        """Wrap ``skinning.update`` so the visualisation follows the pose.

        Wrapping rather than calling from the simulation loop keeps the two in
        lockstep: the colours are recomputed from the same deformed positions
        the update just produced, in the same call, so a frame can never show
        stretch colours from the previous pose.  The wrapper also swallows (and
        reports once) exceptions from the skinning update, because a raising
        update on the paint path would otherwise stop the render loop dead.
        """
        original_update = self.skinning.update
        log_counter = [0]
        err_reported = [False]
        viz = self.stretch_viz

        def update_with_viz(body_state):
            try:
                original_update(body_state)
            except Exception as e:  # noqa: BLE001 - reported once, never fatal
                if not err_reported[0]:
                    print(f"[DEBUG] EXCEPTION in soft tissue update: {e}")
                    import traceback
                    traceback.print_exc()
                    err_reported[0] = True
            if viz.stretch_enabled or viz.chain_enabled:
                if log_counter[0] < 3:
                    print("[DEBUG] _skinning_update_with_viz: calling "
                          f"stretch_viz.update() (stretch={viz.stretch_enabled}, "
                          f"chain={viz.chain_enabled})")
                    log_counter[0] += 1
                viz.update()

        self.skinning.update = update_with_viz

    # -- Selection ---------------------------------------------------------

    def connect_selection(self) -> None:
        tab = self.debug_tab
        self.selection_tool.on_selection_changed = self.on_selection_changed
        tab.selection_mode_toggled.connect(self.on_selection_mode)
        tab.clear_selection_clicked.connect(self.on_clear_selection)
        tab.reassign_clicked.connect(self.on_reassign)
        tab.undo_clicked.connect(self.on_undo)

    def on_selection_mode(self, enabled: bool) -> None:
        self.selection_tool.active = enabled

    def on_selection_changed(self) -> None:
        self.debug_tab.update_selection_count(
            self.selection_tool.selection.total_count)

    def on_clear_selection(self) -> None:
        self.selection_tool.selection.clear()
        self.on_selection_changed()

    def on_reassign(self, chain_label: str) -> None:
        """Move every selected vertex to the chain named in the selector."""
        chain_id = self.parse_leading_id(chain_label)
        total = 0
        for bi, vis in self.selection_tool.selection.get_flat_indices():
            total += self.reassigner.reassign(bi, vis, chain_id)
            self.modified_vertices.setdefault(bi, set()).update(vis)
        if total > 0:
            self.stretch_viz.invalidate_chain_cache()
            self.debug_tab.set_undo_enabled(self.reassigner.can_undo)
            print(f"[FaceForge] Reassigned {total} vertices to chain {chain_id}")

    def on_undo(self) -> None:
        if self.reassigner.undo():
            self.stretch_viz.invalidate_chain_cache()
            self.debug_tab.set_undo_enabled(self.reassigner.can_undo)
            print("[FaceForge] Undo reassignment")

    # -- Overrides ---------------------------------------------------------

    def connect_overrides(self) -> None:
        from faceforge.body.chain_overrides import load_overrides

        tab = self.debug_tab
        tab.save_overrides_clicked.connect(self.on_save_overrides)
        tab.load_overrides_clicked.connect(self.on_load_overrides)

        # Applied event-driven, not now: the meshes these refer to belong to
        # layers that have not loaded yet.
        self.pending_overrides = load_overrides()
        self.ctx.after_registration_hooks.append(self.try_apply_pending_overrides)

    def on_save_overrides(self) -> None:
        from faceforge.body.chain_overrides import (
            collect_modified_overrides, save_overrides,
        )

        overrides = collect_modified_overrides(self.skinning,
                                               self.modified_vertices)
        if not overrides:
            print("[FaceForge] No modified vertices to save")
            return
        path = save_overrides(self.skinning, overrides)
        count = sum(len(v) for v in overrides.values())
        self.debug_tab.set_override_count(count)
        print(f"[FaceForge] Saved {count} overrides to {path}")

    def on_load_overrides(self) -> None:
        """Load overrides from disk and report precisely why none applied.

        Zero applied almost always means the layer they belong to is not
        loaded, so the message names the missing meshes rather than saying
        nothing happened.
        """
        from faceforge.body.chain_overrides import apply_overrides, load_overrides

        overrides = load_overrides()
        if not overrides:
            print("[FaceForge] No overrides file found")
            return
        count = apply_overrides(self.skinning, overrides)
        if count > 0:
            self._on_overrides_applied(count)
            print(f"[FaceForge] Loaded and applied {count} overrides")
            return
        mesh_names = {b.mesh.name for b in self.skinning.bindings}
        missing = set(overrides.keys()) - mesh_names
        if missing:
            print(f"[FaceForge] 0 overrides applied — mesh layers not loaded: "
                  f"{missing} (enable the Skin layer first)")
        else:
            print("[FaceForge] 0 overrides applied — no matching vertices")

    def try_apply_pending_overrides(self) -> None:
        """Post-registration hook: bind startup overrides once meshes exist."""
        if self.pending_overrides is None or not self.skinning.bindings:
            return
        from faceforge.body.chain_overrides import apply_overrides

        count = apply_overrides(self.skinning, self.pending_overrides)
        if count > 0:
            self._on_overrides_applied(count)
            print(f"[FaceForge] Auto-loaded {count} overrides on startup")
        # Applied, or matched nothing: either way stop retrying.
        self.pending_overrides = None

    def _on_overrides_applied(self, count: int) -> None:
        self.debug_tab.set_override_count(count)
        self.stretch_viz.invalidate_chain_cache()
        # Clearing the signature forces a skinning recompute next frame.
        self.skinning._last_signature = ()

    # -- Region labels ------------------------------------------------------

    def attach_region_labels(self) -> None:
        """Wire body-region colouring and region reassignment, if available.

        Isolated in its own try/except and applied *after* the simpler
        selection wiring, replacing three of its connections.  Region labels
        depend on the gender-morph skeleton landmarks, which a degraded load
        may not have produced; when that happens the tab must still offer
        chain-level selection.
        """
        try:
            from faceforge.body.region_labels import BodyRegion

            self.debug_tab.set_region_names(
                [f"{r.value}: {r.name}" for r in BodyRegion])
            self.debug_tab.region_viz_toggled.connect(self.on_region_viz)
            self.debug_tab.region_reassign_clicked.connect(self.on_region_reassign)

            # Replace the selection handlers with body-mesh-aware versions.
            self.selection_tool.on_selection_changed = (
                self.on_selection_changed_with_body)
            self.debug_tab.selection_mode_toggled.disconnect()
            self.debug_tab.selection_mode_toggled.connect(
                self.on_selection_mode_with_body)
            self.debug_tab.clear_selection_clicked.disconnect()
            self.debug_tab.clear_selection_clicked.connect(
                self.on_clear_selection_with_body)

            print("[FaceForge] Region label visualization wired successfully")
        except Exception as e:  # noqa: BLE001 - the tab must survive this
            print(f"[FaceForge] Region label features unavailable: {e}")
            import traceback
            traceback.print_exc()

    def on_region_viz(self, enabled: bool) -> None:
        """Colour skin meshes by anatomical region.

        Segmentation is cached per binding: it is a nearest-landmark
        classification over every vertex of a 790k-vertex mesh and must not run
        on every toggle.
        """
        from faceforge.body.region_labels import compute_region_colors, segment_mh_mesh

        print(f"[DEBUG] _on_region_viz({enabled}), "
              f"bindings={len(self.skinning.bindings)}")
        if not enabled:
            if self.region_viz_active:
                self.region_viz_active = False
                for binding in self.skinning.bindings:
                    if not binding.is_muscle:
                        binding.mesh.material.vertex_colors_active = False
            return

        # Stretch and chain viz write the same colour buffer.
        self.stretch_viz.stretch_enabled = False
        self.stretch_viz.chain_enabled = False

        morph = getattr(self.ctx.pipeline, "gender_morph", None)
        landmarks = morph.skel_landmarks if morph is not None else None
        if landmarks is None:
            print("[FaceForge] Region viz: no skeleton landmarks available yet")
            return

        self.region_viz_active = True
        for binding in self.skinning.bindings:
            if binding.is_muscle:
                continue
            key = id(binding)
            if key not in self.region_color_cache:
                pos = binding.mesh.geometry.positions.reshape(-1, 3)
                labels = segment_mh_mesh(pos.astype(float), landmarks)
                self.region_color_cache[key] = compute_region_colors(labels)
            binding.mesh.geometry.vertex_colors = self.region_color_cache[key]
            binding.mesh.geometry.colors_dirty = True
            binding.mesh.material.vertex_colors_active = True

    def on_region_reassign(self, region_label: str) -> None:
        """Relabel the selected body-mesh vertices to the chosen region."""
        region_id = self.parse_leading_id(region_label)
        morph = getattr(self.ctx.pipeline, "gender_morph", None)
        if morph is None or morph.mh_region_labels is None:
            print("[FaceForge] Region reassign: no region labels computed yet")
            return
        count = 0
        for vi in self.selection_tool.body_selection:
            if 0 <= vi < len(morph.mh_region_labels):
                morph.mh_region_labels[vi] = region_id
                self.region_modified[vi] = region_id
                count += 1
        if count == 0:
            return
        # The KD-trees and colour cache were built from the old labels.
        morph._region_kdtrees = None
        self.region_color_cache.clear()
        self.debug_tab.update_region_override_count(len(self.region_modified))
        print(f"[FaceForge] Set {count} body mesh vertices to region {region_id}")
        if self.region_viz_active:
            self.on_region_viz(True)

    def ensure_body_mesh_ref(self) -> None:
        """Give the selection tool the body mesh, once the morph has built it.

        Deferred because the body surface mesh does not exist until the gender
        morph has loaded, which is after the debug tab is wired.
        """
        if self.selection_tool.body_mesh is not None:
            return
        morph = getattr(self.ctx.pipeline, "gender_morph", None)
        if morph is not None and morph.body_mesh is not None:
            self.selection_tool.body_mesh = morph.body_mesh

    def on_selection_mode_with_body(self, enabled: bool) -> None:
        self.ensure_body_mesh_ref()
        self.selection_tool.active = enabled
        self.on_selection_changed_with_body()

    def on_selection_changed_with_body(self) -> None:
        self.debug_tab.update_selection_count(
            self.selection_tool.selection.total_count
            + len(self.selection_tool.body_selection))

    def on_clear_selection_with_body(self) -> None:
        self.selection_tool.selection.clear()
        self.selection_tool.body_selection.clear()
        self.on_selection_changed_with_body()
