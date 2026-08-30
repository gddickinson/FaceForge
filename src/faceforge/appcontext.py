"""The collaborators an assembled FaceForge application is made of.

Why this module exists
----------------------
``app.main()`` used to construct about forty objects as local variables and
then define ~100 nested closures over them.  Nothing was importable, so the
whole application layer -- every event handler, the tiered asset load, the
debug-tab wiring -- had no unit tests: there was no name to import and no way
to build one of those handlers without building a ``QApplication`` and loading
790k vertices of anatomy first.

:class:`AppContext` is that set of locals given a name.  Controllers take a
context and read the collaborators they need off it, which means a test can
construct a context out of two or three stubs and exercise a handler directly.

Field policy
------------
Every field defaults to ``None`` or to an empty container.  That is deliberate
and load-bearing for tests: ``AppContext(scene=FakeScene(), state=StateManager())``
is a valid context for exercising a handler that only touches those two.  It is
*not* an invitation to leave fields unset in the real application --
:func:`build_app_context` fills all of them, and a controller that finds a
collaborator missing at runtime is looking at a wiring bug.

Mutable shared state
--------------------
Three fields are shared mutable containers rather than values, because the
original code shared them between closures by ``nonlocal`` and the loading
order depends on that sharing:

``skin_chain_ids``
    name -> chain id, filled while the joint chains are built and read
    afterwards by every on-demand loader deciding which chains a mesh follows.
``after_registration_hooks``
    callbacks run after any loader registers meshes with soft-tissue skinning
    (this is how saved vertex overrides get applied to layers loaded later).
``joint_chain_builder``
    the chain-building callable itself, which the gender slider re-runs on
    release.  Set by :class:`~faceforge.coordination.asset_load_sequence.AssetLoadSequence`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class AppContext:
    """Everything an assembled FaceForge application is wired out of.

    Not frozen: the asset load sequence legitimately fills fields in
    (``joint_chain_builder``, ``startup_preset``) after construction, and the
    real application mutates the collaborators themselves throughout its life.
    """

    # ── Core ────────────────────────────────────────────────────────────
    event_bus: Any = None
    state: Any = None
    clock: Any = None

    # ── Assets and scene graph ──────────────────────────────────────────
    assets: Any = None
    visibility: Any = None
    scene: Any = None
    named_nodes: dict[str, Any] = field(default_factory=dict)
    pipeline: Any = None

    # ── Simulation ──────────────────────────────────────────────────────
    simulation: Any = None
    physiology: Any = None
    presets: Any = None

    # ── Qt / rendering ──────────────────────────────────────────────────
    gl_widget: Any = None
    window: Any = None
    label_overlay: Any = None

    # ── Scene mode and animation ────────────────────────────────────────
    scene_controller: Any = None
    anim_player: Any = None
    builtin_clips: dict[str, Any] = field(default_factory=dict)

    # ── Feature systems ─────────────────────────────────────────────────
    muscle_activation: Any = None
    search_index: Any = None
    pathology: Any = None
    speech_engine: Any = None
    quiz_engine: Any = None
    video_exporter: Any = None
    blood_particles: Any = None
    neural_particles: Any = None

    # ── Virtual scanner ─────────────────────────────────────────────────
    tissue_mapper: Any = None
    scanner_engine: Any = None
    scan_plane_viz: Any = None

    # ── Shared mutable loading state (see module docstring) ─────────────
    skin_chain_ids: dict[str, int] = field(default_factory=dict)
    after_registration_hooks: list[Callable[[], None]] = field(default_factory=list)
    joint_chain_builder: Callable[[], list] | None = None

    # ── Startup selections (set once the startup dialog has been shown) ──
    startup_preset: str | None = None
    startup_illustration: str | None = None

    #: The asset load sequence, once armed.  Held here for two reasons: a
    #: caller can ask which stage the load reached, and -- load-bearing --
    #: something must own a strong reference to it.  ``QTimer.singleShot``
    #: does not keep the bound method it is given alive, so a sequence held
    #: only by the timer is collected and the load silently never runs.
    load_sequence: Any = None

    # -- Convenience accessors ------------------------------------------
    # Controllers reach for these constantly; a missing collaborator should
    # read as "not wired" rather than raise AttributeError deep in a handler.

    @property
    def camera(self) -> Any:
        return getattr(self.gl_widget, "camera", None)

    @property
    def lights(self) -> Any:
        return getattr(self.gl_widget, "lights", None)

    @property
    def renderer(self) -> Any:
        return getattr(self.gl_widget, "renderer", None)

    @property
    def control_panel(self) -> Any:
        return getattr(self.window, "control_panel", None)

    def node(self, name: str) -> Any:
        """A named scene node, or ``None`` if the scene has not built it."""
        return self.named_nodes.get(name)

    def run_after_registration_hooks(self) -> None:
        """Run every post-registration hook, in registration order."""
        for hook in self.after_registration_hooks:
            hook()


def build_app_context(*, argv: list[str] | None = None) -> AppContext:
    """Construct the real application's collaborators, in dependency order.

    Qt widgets are created here, so a ``QApplication`` must already exist.
    Everything is constructed but nothing is *wired*: no event handlers are
    subscribed and no assets are loaded.  Wiring is
    :func:`faceforge.controllers.build_controllers`; loading is
    :class:`~faceforge.coordination.asset_load_sequence.AssetLoadSequence`.
    """
    from faceforge.animation.preset_manager import PresetManager
    from faceforge.animation.speech import SpeechEngine
    from faceforge.anatomy.pathology import PathologySystem
    from faceforge.anatomy.quiz_engine import QuizEngine
    from faceforge.anatomy.structure_search import AnatomySearchIndex
    from faceforge.body.muscle_activation import MuscleActivationSystem
    from faceforge.body.physiology import PhysiologySystem
    from faceforge.coordination.loading_pipeline import LoadingPipeline
    from faceforge.coordination.scene_builder import SceneBuilder
    from faceforge.coordination.simulation import Simulation
    from faceforge.coordination.visibility import VisibilityManager
    from faceforge.core.clock import DeltaClock
    from faceforge.core.events import EventBus
    from faceforge.core.state import StateManager
    from faceforge.export.video_export import VideoExporter
    from faceforge.loaders.asset_manager import AssetManager
    from faceforge.rendering.gl_widget import GLViewport
    from faceforge.rendering.particle_system import ParticleSystem
    from faceforge.scanner.engine import ScannerEngine
    from faceforge.scanner.scan_plane import ScanPlaneViz
    from faceforge.scanner.tissue_map import TissueMapper
    from faceforge.scene.builtin_animations import get_builtin_clips
    from faceforge.scene.scene_animation import AnimationPlayer
    from faceforge.scene.scene_mode_controller import SceneModeController
    from faceforge.ui.widgets.label_overlay import LabelOverlay

    ctx = AppContext()

    # Core systems
    ctx.event_bus = EventBus()
    ctx.state = StateManager()
    ctx.clock = DeltaClock()

    # Assets.  init_transform() must run before the scene builder: without it
    # the scene loads without error and renders in the wrong coordinate system.
    ctx.assets = AssetManager()
    ctx.assets.init_transform()
    ctx.visibility = VisibilityManager()

    # Scene graph
    builder = SceneBuilder(ctx.assets, ctx.visibility)
    ctx.scene, ctx.named_nodes = builder.build()

    # Simulation
    ctx.simulation = Simulation(ctx.state, ctx.scene)
    ctx.physiology = PhysiologySystem()
    ctx.simulation.physiology = ctx.physiology

    # Presets
    ctx.presets = PresetManager()
    try:
        ctx.presets.load()
    except FileNotFoundError:
        pass

    # GL viewport
    ctx.gl_widget = GLViewport()
    ctx.gl_widget.scene = ctx.scene

    # Main window (imported here to avoid circular imports with gl_widget)
    from faceforge.ui.main_window import MainWindow
    ctx.window = MainWindow(ctx.event_bus, ctx.state, ctx.gl_widget)
    ctx.label_overlay = LabelOverlay(ctx.gl_widget)

    # Virtual scanner
    ctx.tissue_mapper = TissueMapper()
    ctx.scanner_engine = ScannerEngine(ctx.tissue_mapper)
    ctx.scan_plane_viz = ScanPlaneViz(ctx.scene)

    # Scene mode + animation
    ctx.scene_controller = SceneModeController()
    ctx.anim_player = AnimationPlayer()
    ctx.builtin_clips = get_builtin_clips()

    # Feature systems
    ctx.muscle_activation = MuscleActivationSystem()
    ctx.simulation.muscle_activation = ctx.muscle_activation
    ctx.search_index = AnatomySearchIndex()
    ctx.pathology = PathologySystem()
    ctx.simulation.pathology = ctx.pathology
    ctx.speech_engine = SpeechEngine()
    ctx.quiz_engine = QuizEngine(ctx.search_index)
    ctx.video_exporter = VideoExporter(ctx.gl_widget)
    ctx.blood_particles = ParticleSystem(max_particles=3000)
    ctx.neural_particles = ParticleSystem(max_particles=2000)

    # Loading pipeline
    ctx.pipeline = LoadingPipeline(ctx.assets, ctx.event_bus, ctx.named_nodes)

    return ctx
