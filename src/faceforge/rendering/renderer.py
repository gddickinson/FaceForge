"""Main OpenGL renderer -- traverses the scene graph and issues draw calls.

Uses OpenGL 3.3 core profile with Phong lighting.
"""

import logging

import numpy as np
from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_LESS,
    GL_MULTISAMPLE,
    GL_PROGRAM_POINT_SIZE,
    glClear,
    glClearColor,
    glDepthFunc,
    glEnable,
    glViewport,
)

from faceforge.core.material import RenderMode
from faceforge.core.math_utils import Mat4, mat3_normal
from faceforge.core.mesh import MeshInstance
from faceforge.core.scene_graph import Scene
from faceforge.rendering.camera import Camera
from faceforge.rendering.gl_material import apply_material, restore_material_defaults
from faceforge.rendering.gl_mesh import GLMesh
from faceforge.rendering.lights import LightSetup
from faceforge.rendering.shader_program import ShaderProgram, load_shader_source

logger = logging.getLogger(__name__)


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
        self._gl_meshes: dict[int, GLMesh] = {}  # keyed by id(MeshInstance)
        self._initialised: bool = False
        self._width: int = 1
        self._height: int = 1
        self._frame_count: int = 0
        self._bg_color_dirty: bool = False

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

        # Collect visible meshes with world transforms
        mesh_list: list[tuple[MeshInstance, Mat4]] = scene.collect_meshes()

        # Multi-frame diagnostic: log body mesh world matrices for the
        # first 10 frames after a non-identity world matrix is detected.
        if not self._scene_transform_logged and len(mesh_list) > 0:
            has_non_identity = False
            for mesh, world in mesh_list[:5]:
                if not np.allclose(np.diag(world), [1, 1, 1, 1], atol=0.01):
                    has_non_identity = True
                    break
            if has_non_identity:
                self._scene_transform_logged = True
                self._scene_diag_start = self._frame_count

        if (hasattr(self, '_scene_diag_start')
                and self._frame_count - self._scene_diag_start < 10
                and len(mesh_list) > 0):
            body_count = 0
            identity_count = 0
            sample_name = ""
            sample_diag = None
            for mesh, world in mesh_list[:20]:
                if getattr(mesh, 'scene_affected', False):
                    body_count += 1
                    if np.allclose(np.diag(world), [1, 1, 1, 1], atol=0.01):
                        identity_count += 1
                    elif not sample_name:
                        sample_name = mesh.name
                        sample_diag = np.diag(world).round(3)
            if body_count > 0:
                logger.info(
                    "Frame %d: %d body meshes, %d identity, sample='%s' diag=%s",
                    self._frame_count, body_count, identity_count,
                    sample_name, sample_diag,
                )

        # Check if we're in OPAQUE mode (all meshes rendered as fully opaque)
        global_opaque = (
            len(mesh_list) > 0
            and mesh_list[0][0].material.render_mode == RenderMode.OPAQUE
        )

        # Sort: opaque first, then transparent (back-to-front by distance)
        cam_pos = camera.position
        opaque: list[tuple[MeshInstance, Mat4]] = []
        transparent: list[tuple[MeshInstance, Mat4, float]] = []

        for mesh, world in mesh_list:
            if global_opaque:
                # In OPAQUE mode, treat everything as opaque
                opaque.append((mesh, world))
            else:
                is_trans = mesh.material.transparent or mesh.material.opacity < 1.0
                if is_trans:
                    center = world[:3, 3]
                    dist = float(np.linalg.norm(center - cam_pos))
                    transparent.append((mesh, world, dist))
                else:
                    opaque.append((mesh, world))

        transparent.sort(key=lambda t: t[2], reverse=True)

        total = len(opaque) + len(transparent)
        self._frame_count += 1
        if self._frame_count <= 3 or (self._frame_count % 300 == 0 and total > 0):
            logger.debug(
                "Frame %d: %d meshes (%d opaque, %d transparent), viewport %dx%d",
                self._frame_count, total, len(opaque), len(transparent),
                self._width, self._height,
            )

        # Draw opaque
        for mesh, world in opaque:
            self._draw_mesh(mesh, world, view, proj, lights)

        # Draw transparent
        for mesh, world, _dist in transparent:
            self._draw_mesh(mesh, world, view, proj, lights)

        restore_material_defaults()

    # ------------------------------------------------------------------
    # Clip plane
    # ------------------------------------------------------------------

    def set_clip_plane(self, normal: tuple, offset: float) -> None:
        """Enable a world-space clip plane.  Fragments with
        ``dot(pos, normal) + offset < 0`` are discarded."""
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
            RenderMode.SOLID: (vert_src, phong_frag),
            RenderMode.WIREFRAME: (vert_src, load_shader_source("wireframe.frag")),
            RenderMode.XRAY: (vert_src, load_shader_source("xray.frag")),
            RenderMode.POINTS: (points_vert_src, load_shader_source("points.frag")),
            RenderMode.OPAQUE: (vert_src, phong_frag),
            RenderMode.ILLUSTRATION: (vert_src, load_shader_source("illustration.frag")),
        }

        for mode, (v_src, f_src) in shader_configs.items():
            sp = ShaderProgram(v_src, f_src)
            sp.compile()
            self._shaders[mode] = sp
            logger.debug("Compiled shader for %s", mode.name)

    def _ensure_gl_mesh(self, mesh: MeshInstance) -> GLMesh:
        """Upload or update the GPU-side mesh for *mesh*."""
        key = id(mesh)
        gl_mesh = self._gl_meshes.get(key)

        if gl_mesh is None:
            # First time -- create and upload
            gl_mesh = GLMesh(mesh.geometry, dynamic=True)
            gl_mesh.upload()
            mesh.gl_handle = gl_mesh
            mesh.needs_update = False
            self._gl_meshes[key] = gl_mesh
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

        shader.use()

        # Apply scene transform for body meshes when in scene mode.
        # model_view = view @ scene_transform @ world  (body meshes)
        # model_view = view @ world                    (environment meshes)
        if self.scene_transform is not None and mesh.scene_affected:
            effective_world = self.scene_transform @ world
            # One-shot diagnostic: log first body mesh drawn with scene_transform
            if not self._scene_transform_logged:
                self._scene_transform_logged = True
                logger.info(
                    "Scene transform applied to '%s': "
                    "scene_diag=%s, world_diag=%s, eff_diag=%s, mv_diag=%s",
                    mesh.name,
                    np.diag(self.scene_transform).round(3),
                    np.diag(world).round(3),
                    np.diag(effective_world).round(3),
                    np.diag(view @ effective_world).round(3),
                )
        else:
            effective_world = world
        model_view = view @ effective_world

        shader.set_uniform_mat4("uModelView", model_view)
        shader.set_uniform_mat4("uProjection", proj)
        shader.set_uniform_mat4("uModelMatrix", effective_world)

        # Clip plane
        shader.set_uniform_int("uClipEnabled", 1 if self.clip_plane_enabled else 0)
        if self.clip_plane_enabled:
            shader.set_uniform_vec4("uClipPlane", self.clip_plane)

        # Normal matrix (inverse transpose of upper-left 3x3 of model-view)
        try:
            normal_mat = mat3_normal(model_view)
        except np.linalg.LinAlgError:
            normal_mat = np.eye(3, dtype=np.float64)
        shader.set_uniform_mat3("uNormalMatrix", normal_mat)

        # Point size for points mode
        if mode == RenderMode.POINTS:
            shader.set_uniform_float("uPointSize", 4.0)

        # Light and material uniforms
        lights.apply(shader)
        lights.upload_point_light(shader, view)
        apply_material(shader, mesh.material)

        # Draw
        gl_mesh.draw(mode)

    def remove_mesh(self, mesh: MeshInstance) -> None:
        """Remove a mesh's GL resources (e.g. when it's removed from the scene)."""
        key = id(mesh)
        gl_mesh = self._gl_meshes.pop(key, None)
        if gl_mesh is not None:
            gl_mesh.destroy()
            mesh.gl_handle = None
