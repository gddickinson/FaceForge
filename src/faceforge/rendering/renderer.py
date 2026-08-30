"""Main OpenGL renderer -- traverses the scene graph and issues draw calls.

Uses OpenGL 3.3 core profile with Phong lighting.
"""

import logging

import numpy as np
from OpenGL.GL import (
    GL_BACK,
    GL_BLEND,
    GL_CLIP_DISTANCE0,
    GL_COLOR_BUFFER_BIT,
    GL_CULL_FACE,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_FILL,
    GL_FRONT_AND_BACK,
    GL_LESS,
    GL_LINE,
    GL_MULTISAMPLE,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_POINT,
    GL_PROGRAM_POINT_SIZE,
    GL_SRC_ALPHA,
    glBindVertexArray,
    glBlendFunc,
    glClear,
    glClearColor,
    glCullFace,
    glDepthFunc,
    glDepthMask,
    glDisable,
    glEnable,
    glPolygonMode,
    glViewport,
)

from faceforge.core.material import RenderMode
from faceforge.core.math_utils import Mat4
from faceforge.core.mesh import MeshInstance
from faceforge.core.scene_graph import Scene
from faceforge.rendering.camera import Camera
from faceforge.rendering.gl_material import needs_blending, restore_material_defaults
from faceforge.rendering.gl_mesh import GLMesh
from faceforge.rendering.lights import LightSetup
from faceforge.rendering.shader_program import ShaderProgram, load_shader_source

logger = logging.getLogger(__name__)

# Fragment shaders that read vWorldPos for something other than the clip-plane
# test, so uModelMatrix must still be uploaded when clipping is off.
_NEEDS_WORLD_POS = frozenset({RenderMode.HOLOGRAM, RenderMode.BLUEPRINT})


class _GLStateMirror:
    """Shadow copy of the GL state the material path touches.

    ``apply_material`` issues its blend, cull and polygon-mode calls on every
    invocation, taking one branch or the other in each of the three blocks: 4 to
    6 calls per mesh depending on the material, measured at 5.01 per mesh
    (2,506 state calls for 500 meshes) on the pre-fix tree.  With this mirror
    plus draws sorted by material, the same scene costs 11 state calls for the
    whole frame -- and, measured at 10 and 100 meshes, a count that does not
    scale with mesh count at all.
    """

    __slots__ = ("blend", "depth_mask", "cull", "poly")

    def __init__(self) -> None:
        self.invalidate()

    def invalidate(self) -> None:
        """Forget everything -- call whenever code outside this class touches
        the state (e.g. the selection overlay in gl_widget)."""
        self.blend = None
        self.depth_mask = None
        self.cull = None
        self.poly = None

    def set_blend(self, on: bool) -> None:
        if self.blend is not on:
            if on:
                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            else:
                glDisable(GL_BLEND)
            self.blend = on

    def set_depth_mask(self, on: bool) -> None:
        if self.depth_mask is not on:
            glDepthMask(on)
            self.depth_mask = on

    def set_cull(self, on: bool) -> None:
        if self.cull is not on:
            if on:
                glEnable(GL_CULL_FACE)
                glCullFace(GL_BACK)
            else:
                glDisable(GL_CULL_FACE)
            self.cull = on

    def set_poly(self, mode: int) -> None:
        if self.poly != mode:
            glPolygonMode(GL_FRONT_AND_BACK, mode)
            self.poly = mode


class GLRenderer:
    """Traverses a :class:`Scene`, uploads meshes on demand, and draws them.

    Usage
    -----
    1. Call :meth:`init_gl` once after a valid GL context is current.
    2. Call :meth:`resize` whenever the viewport changes.
    3. Call :meth:`render` each frame.
    4. Call :meth:`destroy` on shutdown.
    """

    # Background colour (dark blue-grey)
    CLEAR_COLOR = (0.12, 0.12, 0.15, 1.0)

    def __init__(self) -> None:
        self._shaders: dict[RenderMode, ShaderProgram] = {}
        # GLMesh ownership lives on the MeshInstance itself (mesh.gl_handle);
        # this dict exists only so destroy() can free everything.  It used to be
        # keyed by id(MeshInstance): CPython recycles id() values after GC, so a
        # freed mesh's VAO could be handed to an unrelated new mesh -- and
        # because nothing in the tree ever called remove_mesh(), the dict also
        # grew without bound as the skull/body hierarchies were rebuilt.  It is
        # now keyed by id(GLMesh), of which this dict holds the only reference,
        # so the key cannot be recycled while it is in use.
        self._gl_meshes: dict[int, GLMesh] = {}
        self._initialised: bool = False
        self._width: int = 1
        self._height: int = 1
        self._frame_count: int = 0
        self._bg_color_dirty: bool = False

        # Redundancy elimination for the per-mesh hot path.
        self._state = _GLStateMirror()
        # program id -> {uniform name: last uploaded value}; only cheap-to-compare
        # uniforms (scalars, vec3, vec4, int) are tracked -- matrices change per
        # mesh anyway and byte-comparing them costs more than re-uploading.
        self._uniform_state: dict[int, dict] = {}
        self._frame_programs: set[int] = set()
        self._cur_program: int | None = None
        self._cur_vao: int | None = None
        # geometry id -> local-space centroid, for depth-sorting transparency
        self._centroid_cache: dict[int, np.ndarray] = {}
        # Whether GL_CLIP_DISTANCE0 is currently enabled on the context.
        self._clip_distance_on: bool = False

        # Scene mode: when set, this 4x4 matrix is multiplied into the
        # model-view for meshes with ``mesh.scene_affected == True``.
        # This applies the supine rotation + table positioning at render
        # time, bypassing the scene graph (which stays in clinical frame).
        self.scene_transform: Mat4 | None = None
        self._scene_transform_logged: bool = False

        # Clip plane: world-space half-plane for cutaway views.
        # Plane equation: dot(pos, normal) + offset < 0 → discard.
        self.clip_plane_enabled: bool = False
        self.clip_plane: tuple = (1.0, 0.0, 0.0, 0.0)  # (nx, ny, nz, offset)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init_gl(self) -> None:
        """Set up GL state and compile all shader programs.

        Must be called with a current OpenGL context.
        """
        glClearColor(*self.CLEAR_COLOR)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glEnable(GL_MULTISAMPLE)
        glEnable(GL_PROGRAM_POINT_SIZE)

        self._compile_shaders()
        self._initialised = True
        logger.info("GLRenderer initialised.")

    def resize(self, width: int, height: int) -> None:
        """Update the viewport dimensions."""
        self._width = max(width, 1)
        self._height = max(height, 1)

    def destroy(self) -> None:
        """Free all GL resources."""
        for gl_mesh in self._gl_meshes.values():
            gl_mesh.destroy()
        self._gl_meshes.clear()
        self._centroid_cache.clear()
        self._begin_frame()

        for shader in self._shaders.values():
            shader.destroy()
        self._shaders.clear()

        self._initialised = False
        logger.info("GLRenderer destroyed.")

    # ------------------------------------------------------------------
    # Frame rendering
    # ------------------------------------------------------------------

    def render(self, scene: Scene, camera: Camera, lights: LightSetup) -> None:
        """Render one frame: clear, traverse scene, draw all visible meshes."""
        if not self._initialised:
            return

        if self._bg_color_dirty:
            glClearColor(*self.CLEAR_COLOR)
            self._bg_color_dirty = False

        glViewport(0, 0, self._width, self._height)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Update scene graph world matrices
        scene.update()

        view = camera.get_view_matrix()
        proj = camera.get_projection_matrix()

        # Free the GL resources of meshes that have left the scene graph.
        self._drain_orphans(scene)

        # Collect visible meshes with world transforms
        mesh_list: list[tuple[MeshInstance, Mat4]] = scene.collect_meshes()

        # Partition into opaque and blended.  The decision is now per mesh: it
        # used to be derived from mesh_list[0] alone, so a single mesh whose
        # mode happened to be OPAQUE turned off back-to-front sorting for every
        # genuinely transparent mesh in the scene (and vice versa).
        cam_pos = camera.position
        opaque: list[tuple[MeshInstance, Mat4]] = []
        transparent: list[tuple[MeshInstance, Mat4, float]] = []

        for mesh, world in mesh_list:
            if needs_blending(mesh.material):
                # Sort key is the transformed geometry centroid, not the node
                # origin: BodyParts3D structures sit far from their node origins
                # (many share the body root), so world[:3, 3] gave every mesh in
                # a group the same depth and the sort did nothing.
                centre = self._world_centroid(mesh, world)
                delta = centre - cam_pos
                transparent.append((mesh, world, float(delta @ delta)))
            else:
                opaque.append((mesh, world))

        # Group opaque draws by shader program and material state so the state
        # mirror and uniform cache below actually hit.  Stable, so a homogeneous
        # scene keeps its traversal order exactly.
        opaque.sort(key=self._sort_key)
        # Squared distance, so the ordering is identical to sorting on distance.
        transparent.sort(key=lambda t: t[2], reverse=True)

        total = len(opaque) + len(transparent)
        self._frame_count += 1
        if self._frame_count <= 3 or (self._frame_count % 300 == 0 and total > 0):
            logger.debug(
                "Frame %d: %d meshes (%d opaque, %d transparent), viewport %dx%d",
                self._frame_count, total, len(opaque), len(transparent),
                self._width, self._height,
            )

        # Per-frame cache reset.  Anything outside this class that touches GL
        # state (the selection overlay, an exporter) would otherwise leave the
        # mirror believing stale values, so it is rebuilt from scratch each
        # frame rather than trusted across one.
        self._begin_frame()
        self._sync_clip_distance()

        # Draw opaque
        for mesh, world in opaque:
            self._draw_mesh(mesh, world, view, proj, lights)

        # Draw transparent
        for mesh, world, _dist in transparent:
            self._draw_mesh(mesh, world, view, proj, lights)

        # One unbind for the whole frame instead of one per mesh.
        if self._cur_vao:
            glBindVertexArray(0)
            self._cur_vao = 0

        restore_material_defaults()
        self._state.invalidate()

    # ------------------------------------------------------------------
    # Clip plane
    # ------------------------------------------------------------------

    def invalidate_state_cache(self) -> None:
        """Call after any code outside the renderer changes GL state.

        gl_widget's selection-point and lasso overlays toggle depth test and
        blending directly; without this the mirror would believe stale values.
        ``render()`` already does this at both ends of a frame, so this is only
        needed by callers that interleave their own GL work with ``_draw_mesh``.
        """
        self._begin_frame()

    def _begin_frame(self) -> None:
        self._frame_programs.clear()
        self._uniform_state.clear()
        self._state.invalidate()
        self._cur_program = None
        self._cur_vao = None

    def _drain_orphans(self, scene: Scene) -> None:
        """Release GL resources for meshes detached from *scene*.

        ``remove_mesh`` is the only eviction path and used to be called from
        nowhere, while ~10 sites across the tree drop nodes from the graph.
        ``SceneNode.remove`` now queues the detached subtree's meshes and this
        drains the queue, so eviction happens wherever a detach happens without
        every one of those sites needing a renderer reference.
        """
        take = getattr(scene, "take_orphaned_meshes", None)
        if take is None:
            return
        for mesh in take():
            self.remove_mesh(mesh)

    def _sync_clip_distance(self) -> None:
        """Match ``GL_CLIP_DISTANCE0`` to :attr:`clip_plane_enabled`.

        Clipping moved from a fragment-shader ``discard`` to hardware clipping
        via ``gl_ClipDistance[0]`` in the vertex shader: the mere *presence* of
        ``discard`` in a fragment shader statically disables early depth
        rejection on most drivers, whether or not any fragment is discarded and
        whether or not the cutaway is even switched on.

        Done here rather than in ``set_clip_plane`` because that is called from
        UI callbacks that have no guarantee of a current GL context.  Mirrored,
        so it costs one GL call when the user toggles the cutaway and none
        otherwise.
        """
        want = self.clip_plane_enabled
        if want != self._clip_distance_on:
            if want:
                glEnable(GL_CLIP_DISTANCE0)
            else:
                glDisable(GL_CLIP_DISTANCE0)
            self._clip_distance_on = want

    def set_clip_plane(self, normal: tuple, offset: float) -> None:
        """Enable a world-space clip plane.  Fragments with
        ``dot(pos, normal) + offset < 0`` are clipped."""
        self.clip_plane_enabled = True
        self.clip_plane = (float(normal[0]), float(normal[1]), float(normal[2]), float(offset))

    def clear_clip_plane(self) -> None:
        """Disable the clip plane."""
        self.clip_plane_enabled = False

    # ------------------------------------------------------------------
    # Shader access
    # ------------------------------------------------------------------

    def get_shader(self, mode: RenderMode) -> ShaderProgram:
        """Return the compiled shader program for a given render mode."""
        return self._shaders[mode]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compile_shaders(self) -> None:
        """Compile all shader variants."""
        vert_src = load_shader_source("default.vert")
        points_vert_src = load_shader_source("points.vert")

        # Use point-light-aware phong shader (backward compatible when point
        # light is disabled — the uniform branch adds zero cost).
        phong_frag = load_shader_source("phong_pointlight.frag")
        shader_configs = {
            # Standard clinical
            RenderMode.SOLID: (vert_src, phong_frag),
            RenderMode.WIREFRAME: (vert_src, load_shader_source("wireframe.frag")),
            RenderMode.XRAY: (vert_src, load_shader_source("xray.frag")),
            RenderMode.POINTS: (points_vert_src, load_shader_source("points.frag")),
            RenderMode.OPAQUE: (vert_src, phong_frag),
            # Textbook / illustration
            RenderMode.ILLUSTRATION: (vert_src, load_shader_source("illustration.frag")),
            RenderMode.SEPIA: (vert_src, load_shader_source("sepia.frag")),
            RenderMode.COLOR_ATLAS: (vert_src, load_shader_source("color_atlas.frag")),
            RenderMode.PEN_INK: (vert_src, load_shader_source("pen_ink.frag")),
            RenderMode.MEDICAL: (vert_src, load_shader_source("medical.frag")),
            # Creative / stylised
            RenderMode.HOLOGRAM: (vert_src, load_shader_source("hologram.frag")),
            RenderMode.CARTOON: (vert_src, load_shader_source("cartoon.frag")),
            RenderMode.PORCELAIN: (vert_src, load_shader_source("porcelain.frag")),
            RenderMode.BLUEPRINT: (vert_src, load_shader_source("blueprint.frag")),
            RenderMode.THERMAL: (vert_src, load_shader_source("thermal.frag")),
            RenderMode.ETHEREAL: (vert_src, load_shader_source("ethereal.frag")),
        }

        for mode, (v_src, f_src) in shader_configs.items():
            sp = ShaderProgram(v_src, f_src)
            sp.compile()
            self._shaders[mode] = sp
            logger.debug("Compiled shader for %s", mode.name)

    @staticmethod
    def _sort_key(pair) -> tuple:
        """Group draws by shader program, then by the material state that costs
        a GL call to change."""
        mat = pair[0].material
        return (mat.render_mode.value, mat.double_sided, mat.shininess,
                mat.opacity, mat.vertex_colors_active)

    #: Attribute the local centroid is memoised under, on the geometry itself.
    _CENTROID_ATTR = "_ff_local_centroid"

    def _world_centroid(self, mesh: MeshInstance, world: Mat4) -> np.ndarray:
        """Geometry centroid in world space, with the local centroid cached.

        The cache lives ON the geometry object rather than in a dict keyed by
        ``id(geometry)``.  That is the same defect this class already fixed for
        ``_gl_meshes`` (see ``__init__``): CPython recycles ``id()`` values
        after collection, and a dict that keys on ``id(x)`` while storing only
        a value derived from ``x`` holds nothing alive -- so once a geometry is
        freed and its address reused, the next geometry silently inherits the
        previous one's centroid.  Here that would mis-sort transparent draws.

        Memoising on the object ties the cache's lifetime to the thing it
        describes, needs no eviction, and cannot go stale: a new geometry is a
        new object with no such attribute.  ``BufferGeometry`` is not hashable
        (it is an ``eq=True`` dataclass), so a ``WeakKeyDictionary`` is not
        available as an alternative.
        """
        geom = mesh.geometry
        local = getattr(geom, self._CENTROID_ATTR, None)
        if local is None:
            pos = np.asarray(geom.positions, dtype=np.float64).reshape(-1, 3)
            local = pos.mean(axis=0)
            try:
                setattr(geom, self._CENTROID_ATTR, local)
            except (AttributeError, TypeError):
                # A slotted or frozen geometry cannot memoise; recomputing is
                # correct, just slower.  Never fall back to id()-keying.
                pass
        return world[:3, :3] @ local + world[:3, 3]

    # -- uniform helpers with redundancy elimination ----------------------

    def _u_vec3(self, shader: ShaderProgram, name: str, value) -> None:
        if shader.get_uniform_location(name) < 0:
            return
        val = (float(value[0]), float(value[1]), float(value[2]))
        cache = self._uniform_state.setdefault(shader.program_id, {})
        if cache.get(name) == val:
            return
        cache[name] = val
        shader.set_uniform_vec3(name, val)

    def _u_vec4(self, shader: ShaderProgram, name: str, value) -> None:
        if shader.get_uniform_location(name) < 0:
            return
        val = tuple(float(v) for v in value[:4])
        cache = self._uniform_state.setdefault(shader.program_id, {})
        if cache.get(name) == val:
            return
        cache[name] = val
        shader.set_uniform_vec4(name, val)

    def _u_float(self, shader: ShaderProgram, name: str, value: float) -> None:
        if shader.get_uniform_location(name) < 0:
            return
        val = float(value)
        cache = self._uniform_state.setdefault(shader.program_id, {})
        if cache.get(name) == val:
            return
        cache[name] = val
        shader.set_uniform_float(name, val)

    def _u_int(self, shader: ShaderProgram, name: str, value: int) -> None:
        if shader.get_uniform_location(name) < 0:
            return
        val = int(value)
        cache = self._uniform_state.setdefault(shader.program_id, {})
        if cache.get(name) == val:
            return
        cache[name] = val
        shader.set_uniform_int(name, val)

    def _use_program(self, shader: ShaderProgram, proj: Mat4, view: Mat4,
                     lights: LightSetup) -> None:
        """Bind *shader* and, on its first use this frame, upload everything
        that cannot change between meshes within the frame.

        uProjection, the three directional-light uniforms, the point-light
        block and the clip-plane state were previously re-uploaded for every
        mesh: 6 of the 13 per-mesh uniform uploads.
        """
        pid = shader.program_id
        if pid != self._cur_program:
            shader.use()
            self._cur_program = pid
        if pid in self._frame_programs:
            return
        self._frame_programs.add(pid)

        shader.set_uniform_mat4("uProjection", proj)
        self._u_vec3(shader, "uAmbientColor", lights.ambient_color)
        self._u_vec3(shader, "uLightDir", lights.light_dir)
        self._u_vec3(shader, "uLightColor", lights.light_color)
        lights.upload_point_light(shader, view)
        self._u_int(shader, "uClipEnabled", 1 if self.clip_plane_enabled else 0)
        if self.clip_plane_enabled:
            self._u_vec4(shader, "uClipPlane", self.clip_plane)
        self._u_float(shader, "uPointSize", 4.0)

    def _apply_material(self, shader: ShaderProgram, material) -> None:
        """Material uniforms + GL state, both redundancy-filtered."""
        is_opaque_mode = material.render_mode == RenderMode.OPAQUE
        self._u_vec3(shader, "uColor", material.color)
        self._u_float(shader, "uOpacity",
                      1.0 if is_opaque_mode else material.opacity)
        self._u_float(shader, "uShininess", material.shininess)
        self._u_int(shader, "uUseVertexColor",
                    1 if material.vertex_colors_active else 0)

        state = self._state
        if needs_blending(material):
            state.set_blend(True)
            state.set_depth_mask(False)
        else:
            state.set_blend(False)
            state.set_depth_mask(True)
        state.set_cull(not material.double_sided)
        state.set_poly(
            GL_LINE if material.render_mode == RenderMode.WIREFRAME
            else GL_POINT if material.render_mode == RenderMode.POINTS
            else GL_FILL)

    def _ensure_gl_mesh(self, mesh: MeshInstance) -> GLMesh:
        """Upload or update the GPU-side mesh for *mesh*."""
        gl_mesh = mesh.gl_handle

        if not isinstance(gl_mesh, GLMesh) or not gl_mesh.uploaded:
            # First time -- create and upload.  Keyed on the object itself, so a
            # recycled id() can never hand this mesh someone else's buffers.
            gl_mesh = GLMesh(mesh.geometry, dynamic=True)
            gl_mesh.upload()
            mesh.gl_handle = gl_mesh
            mesh.needs_update = False
            self._gl_meshes[id(gl_mesh)] = gl_mesh
        elif mesh.needs_update:
            # Geometry changed -- re-stream vertex data
            gl_mesh.update_positions(mesh.geometry.positions)
            gl_mesh.update_normals(mesh.geometry.normals)
            mesh.needs_update = False

        # Stream vertex colors if dirty
        geom = mesh.geometry
        if geom.vertex_colors is not None and geom.colors_dirty:
            if gl_mesh.has_colors:
                gl_mesh.update_colors(geom.vertex_colors)
            else:
                gl_mesh.upload_colors(geom.vertex_colors)
            geom.colors_dirty = False

        return gl_mesh

    def _draw_mesh(
        self,
        mesh: MeshInstance,
        world: Mat4,
        view: Mat4,
        proj: Mat4,
        lights: LightSetup,
    ) -> None:
        """Draw a single mesh with appropriate shader and uniforms."""
        gl_mesh = self._ensure_gl_mesh(mesh)
        mode = mesh.material.render_mode
        shader = self._shaders.get(mode, self._shaders[RenderMode.SOLID])

        self._use_program(shader, proj, view, lights)

        # Apply scene transform for body meshes when in scene mode.
        # model_view = view @ scene_transform @ world  (body meshes)
        # model_view = view @ world                    (environment meshes)
        if self.scene_transform is not None and mesh.scene_affected:
            effective_world = self.scene_transform @ world
        else:
            effective_world = world
        model_view = view @ effective_world

        shader.set_uniform_mat4("uModelView", model_view)
        # uModelMatrix only feeds vWorldPos, which is read solely by the
        # clip-plane test except in HOLOGRAM and BLUEPRINT (world-space grid /
        # interference bands).
        if self.clip_plane_enabled or mode in _NEEDS_WORLD_POS:
            shader.set_uniform_mat4("uModelMatrix", effective_world)

        # No uNormalMatrix upload: default.vert / points.vert derive the normal
        # matrix as mat3(uModelView).  Valid because FaceForge applies no
        # non-uniform node scale; see the comment in default.vert.

        self._apply_material(shader, mesh.material)

        # Draw without the trailing glBindVertexArray(0) -- the frame does one.
        self._cur_vao = gl_mesh.vao
        gl_mesh.draw(mode, unbind=False)

    def render_split(
        self,
        scene: Scene,
        camera: Camera,
        lights: LightSetup,
        left_config: dict,
        right_config: dict,
    ) -> None:
        """Render split viewport: left half with left_config, right half with right_config.

        Each config dict has 'visibility' (set of toggle names to show) and
        optionally 'render_mode' (RenderMode).
        """
        if not self._initialised:
            return

        if self._bg_color_dirty:
            glClearColor(*self.CLEAR_COLOR)
            self._bg_color_dirty = False

        glViewport(0, 0, self._width, self._height)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        scene.update()
        view = camera.get_view_matrix()
        proj = camera.get_projection_matrix()

        hw = self._width // 2

        self._drain_orphans(scene)
        # collect_meshes() once per frame, not once per half
        mesh_list = scene.collect_meshes()

        # render_mode is overwritten per mesh below and must be put back even if
        # a draw raises: without the try/finally, one mid-frame GL failure left
        # every mesh in the comparison mode permanently.
        saved_modes: dict[int, tuple] = {}
        try:
            for half_idx, config in enumerate((left_config, right_config)):
                x_offset = 0 if half_idx == 0 else hw
                w = hw if half_idx == 0 else (self._width - hw)
                glViewport(x_offset, 0, w, self._height)

                # Adjust projection for half-width aspect ratio
                half_proj = camera.get_projection_matrix_for_size(w, self._height)

                visible_toggles = config.get("visibility", set())
                override_mode = config.get("render_mode")

                # each half is a fresh set of frame constants (different projection)
                self._begin_frame()
                self._sync_clip_distance()

                for mesh, world in mesh_list:
                    # Check if mesh should be visible in this config
                    if visible_toggles and mesh.name not in visible_toggles:
                        continue

                    if override_mode is not None:
                        mat = mesh.material
                        saved_modes.setdefault(id(mat), (mat, mat.render_mode))
                        mat.render_mode = override_mode

                    self._draw_mesh(mesh, world, view, half_proj, lights)

                    if override_mode is not None:
                        mat, original = saved_modes.pop(id(mesh.material))
                        mat.render_mode = original
        finally:
            for mat, original in saved_modes.values():
                mat.render_mode = original
            if self._cur_vao:
                glBindVertexArray(0)
                self._cur_vao = 0
            # Restore full viewport
            glViewport(0, 0, self._width, self._height)
            restore_material_defaults()
            self._state.invalidate()
            self._frame_count += 1

    def remove_mesh(self, mesh: MeshInstance) -> None:
        """Remove a mesh's GL resources (e.g. when it's removed from the scene).

        Called for you by ``_drain_orphans`` whenever a node leaves the scene
        graph; also safe to call directly.
        """
        gl_mesh = mesh.gl_handle
        if isinstance(gl_mesh, GLMesh):
            self._gl_meshes.pop(id(gl_mesh), None)
            gl_mesh.destroy()
            mesh.gl_handle = None
