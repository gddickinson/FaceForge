# rendering/ -- OpenGL 3.3 Rendering Engine

This package implements the OpenGL rendering pipeline using OpenGL 3.3 core profile, integrated with PySide6's QOpenGLWidget for the desktop GUI. It provides Phong-shaded rendering with multiple render modes, mouse-driven orbit controls, and efficient per-frame vertex streaming for animated meshes.

## Modules

### `renderer.py` -- Main Renderer

Traverses the scene graph and issues draw calls.

**Key class:** `GLRenderer`
- `init_gl()` -- Compile shaders, set up GL state (depth test, multisampling)
- `resize(width, height)` -- Update viewport and camera aspect ratio
- `render(scene, camera)` -- Full render pass: clear, traverse visible nodes, upload/update meshes, draw
- `destroy()` -- Clean up GL resources

Manages a cache of `GLMesh` objects keyed by `MeshInstance` identity. Meshes are uploaded on first encounter and updated via `glBufferSubData` for animated geometry. Supports SOLID, WIREFRAME, XRAY, POINTS, and OPAQUE render modes.

Background color: dark blue-grey (0.12, 0.12, 0.15).

### `gl_widget.py` -- PySide6 OpenGL Widget

Bridges Qt and OpenGL rendering.

**Key class:** `GLViewport` (extends QOpenGLWidget)
- Creates OpenGL 3.3 core-profile context with 4x multisampling
- Drives rendering at ~60 fps via QTimer
- Forwards mouse events to OrbitControls
- Manages camera, lights, and renderer lifecycle

**Helper:** `create_gl_format()` -- Configures QSurfaceFormat for OpenGL 3.3 core profile.

### `camera.py` -- Perspective Camera

Produces view and projection matrices.

**Key class:** `Camera`
- Parameters: FOV (50 deg), near (0.1), far (1000)
- Default position: (0, -40, 120), target: (0, -30, 0) for full-body view
- Lazy view/projection matrix computation with dirty flags
- Methods: `get_view_matrix()`, `get_projection_matrix()`, `look_at()`

### `gl_mesh.py` -- VAO/VBO Management

GPU-side representation of mesh geometry.

**Key class:** `GLMesh`
- One VAO per mesh with:
  - VBO slot 0: positions (vec3, location 0)
  - VBO slot 1: normals (vec3, location 1)
  - Optional EBO for indexed geometry
- `upload()` -- Initial GPU upload (GL_STATIC_DRAW or GL_DYNAMIC_DRAW)
- `update_positions(data)` / `update_normals(data)` -- Per-frame vertex streaming via `glBufferSubData`
- `draw(mode)` -- Issues `glDrawElements` (indexed) or `glDrawArrays` (non-indexed)
- Supports GL_TRIANGLES, GL_POINTS, GL_LINES render modes

### `shader_program.py` -- GLSL Shader Management

Compiles and links vertex + fragment shader programs.

**Key class:** `ShaderProgram`
- `compile(vert_source, frag_source)` -- Compile and link with error reporting
- Uniform setters: `set_uniform_float()`, `set_uniform_vec3()`, `set_uniform_mat4()`, `set_uniform_mat3()`
- Attribute location queries
- `use()` / `unuse()` for program binding

**Helper:** `load_shader_source(filename)` -- Loads GLSL from the `shaders/` directory.

### `gl_material.py` -- Material Application

Translates Material properties to GL state and shader uniforms.

**Key functions:**
- `apply_material(shader, material)` -- Sets color, opacity, shininess uniforms; configures blending, culling, polygon mode, depth write based on RenderMode
- `restore_material_defaults()` -- Resets GL state to default after draw

### `orbit_controls.py` -- Mouse Orbit Controls

Mouse-driven camera orbit, pan, and zoom using spherical coordinates.

**Key class:** `OrbitControls`
- Left mouse: orbit (rotate theta/phi around target)
- Middle mouse: pan (translate target and camera)
- Right mouse / scroll: zoom (adjust radius)
- Configurable limits: min/max radius, phi bounds (avoid gimbal lock)
- Smooth damping support

### `lights.py` -- Scene Lighting

Ambient and directional light setup for Phong shading.

**Key class:** `LightSetup`
- Ambient: warm grey (0.4, 0.4, 0.45)
- Directional: normalized (1, 1, 1) direction, intensity (0.8, 0.8, 0.75)
- `apply(shader)` -- Uploads light uniforms

### `shaders/` -- GLSL Source Files

| File | Purpose |
|------|---------|
| `default.vert` | Standard vertex shader: MVP transform, normal transform |
| `phong.frag` | Phong fragment shader: ambient + diffuse + specular |
| `wireframe.frag` | Solid color for wireframe mode |
| `xray.frag` | View-angle-dependent transparency for X-ray mode |
| `points.vert` | Point cloud vertex shader with GL_PROGRAM_POINT_SIZE |
| `points.frag` | Circular point rendering with smooth edges |

## External Dependencies

- `PyOpenGL` -- OpenGL 3.3 core profile bindings
- `PySide6` -- QOpenGLWidget, QSurfaceFormat, QTimer, mouse events
- `numpy` -- Matrix and vertex array operations

## Internal Dependencies

- `core.scene_graph` -- Scene traversal
- `core.mesh` -- BufferGeometry, MeshInstance
- `core.material` -- Material, RenderMode
- `core.math_utils` -- Mat4, Vec3, matrix operations
- `constants` -- Default camera position/target
