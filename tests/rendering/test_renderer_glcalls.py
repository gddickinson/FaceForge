"""Renderer regression tests driven through the headless GL recorder.

These lock in the *GL call budget* of a frame.  The per-mesh cost is the
quantity that decides whether the app can hit 60 fps: PyOpenGL marshals every
call through ctypes, so 22 calls/mesh x 900 meshes is ~20k calls/frame and
consumes the whole 16.7 ms budget in Python before the driver is reached.
If a change to :mod:`faceforge.rendering.renderer` alters the per-mesh call
count, these tests say so and by how much.
"""

# ----------------------------------------------------------------------
# Frame cost
# ----------------------------------------------------------------------

def test_init_gl_compiles_a_shader_for_every_render_mode(gl_env):
    """Every RenderMode must have a compiled program, or get_shader() KeyErrors."""
    rec, mods = gl_env
    from faceforge.core.material import RenderMode

    r = mods["renderer"].GLRenderer()
    r.init_gl()

    assert r._initialised is True
    missing = [m.name for m in RenderMode if m not in r._shaders]
    assert missing == [], f"RenderModes with no compiled shader: {missing}"


def test_steady_state_frame_costs_exactly_3_gl_calls_per_mesh(gl_env, scene_with):
    """Per-mesh GL cost is a fixed 3 calls; frame overhead is 23 calls.

    Derived from the slope between a 1-mesh and a 50-mesh scene, so the
    constant frame overhead cancels out.

    Was 22 per mesh and 6 per frame.  The 19 eliminated per-mesh calls were,
    by group (this fixture, all-identical materials):

      12 of the 13 uniform uploads --
         6 frame constants re-uploaded per mesh (uProjection, the three
           directional-light uniforms, uHasPointLight, uClipEnabled),
         uNormalMatrix (now derived on the GPU as mat3(uModelView)),
         uModelMatrix (only needed when clipping is on or the mode reads
           vWorldPos),
         4 material uniforms re-uploaded with unchanged values
           (uColor, uOpacity, uShininess, uUseVertexColor);
       5 GL state calls (blend, depth mask, cull, cull face, polygon mode);
       1 glUseProgram, re-bound for every mesh;
       1 glBindVertexArray(0) after every draw.

    12 + 5 + 1 + 1 = 19, leaving uModelView + VAO bind + draw = 3.  Most of the
    eliminated calls moved into the constant frame overhead rather than
    vanishing, which is why that number went from 6 to 23.

    This fixture gives every mesh an identical Material, as the app does, so
    the material uniforms are filtered entirely: 3 = uModelView + VAO bind +
    draw.  A scene with a distinct colour per mesh costs 4 (uColor cannot be
    filtered) -- see tests/rendering/bench_glrec.py.
    """
    rec, mods = gl_env
    cam = mods["camera"].Camera()
    lights = mods["lights"].LightSetup()

    def frame_cost(n_meshes: int) -> int:
        scene, _ = scene_with(n_meshes)
        r = mods["renderer"].GLRenderer()
        r.init_gl()
        r.resize(800, 600)
        r.render(scene, cam, lights)   # warm-up: uploads VAOs/VBOs
        rec.reset()
        r.render(scene, cam, lights)   # steady state
        return rec.total

    c1, c50 = frame_cost(1), frame_cost(50)
    per_mesh = (c50 - c1) / 49

    assert per_mesh == 3.0, f"per-mesh GL cost changed: {per_mesh} (was 3)"
    overhead = c1 - per_mesh
    assert overhead == 23.0, f"frame overhead changed: {overhead} (was 23)"


def test_per_mesh_calls_are_1_uniform_1_bind_1_draw(gl_env, scene_with):
    """Break the 3 per-mesh calls down, so a regression names its own cause."""
    rec, mods = gl_env
    scene, _ = scene_with(10)
    r = mods["renderer"].GLRenderer()
    r.init_gl()
    r.resize(800, 600)
    r.render(scene, mods["camera"].Camera(), mods["lights"].LightSetup())
    rec.reset()
    r.render(scene, mods["camera"].Camera(), mods["lights"].LightSetup())

    c = rec.counts
    # 1 matrix upload per mesh (uModelView) + 1 for the frame (uProjection).
    # uModelMatrix is only needed when the clip plane is on or the mode reads
    # vWorldPos; uNormalMatrix is gone entirely (default.vert derives it as
    # mat3(uModelView)).
    assert c["glUniformMatrix4fv"] == 11
    assert c["glUniformMatrix3fv"] == 0, "uNormalMatrix is uploaded again"
    # Frame constants only: uAmbientColor, uLightDir, uLightColor + uColor once.
    assert c["glUniform3f"] == 4
    assert c["glUniform1i"] == 3            # uHasPointLight, uClipEnabled, uUseVertexColor
    assert c["glUniform1f"] == 2            # uOpacity, uShininess
    # 1 VAO bind + 1 draw per mesh; one glUseProgram and one unbind per FRAME.
    assert c["glUseProgram"] == 1
    assert c["glBindVertexArray"] == 11      # 10 binds + 1 unbind for the frame
    assert c["glDrawArrays"] == 10

    uniforms = sum(n for k, n in c.items() if k.startswith("glUniform"))
    assert uniforms == 20, f"{uniforms} uniform uploads for 10 meshes (was 130)"


def test_uniform_uploads_no_longer_dominate_the_frame(gl_env, scene_with):
    """Uniform churn was the bottleneck at 13:1 uniforms per draw; hold the line.

    Was ``g["uniform"] == 650`` for 50 meshes.
    """
    rec, mods = gl_env
    scene, _ = scene_with(50)
    r = mods["renderer"].GLRenderer()
    r.init_gl()
    r.resize(800, 600)
    r.render(scene, mods["camera"].Camera(), mods["lights"].LightSetup())
    rec.reset()
    r.render(scene, mods["camera"].Camera(), mods["lights"].LightSetup())

    g = rec.group()
    assert g["draw"] == 50
    assert g["uniform"] == 60               # was 650
    assert g["uniform"] < 2 * g["draw"], (
        f"uniform:draw ratio {g['uniform'] / g['draw']:.1f}:1 (was 13.0:1)"
    )


def test_frame_state_calls_do_not_scale_with_mesh_count(gl_env, scene_with):
    """The five per-mesh GL state calls are now per-frame, not per-mesh.

    apply_material() used to issue glEnable/glDisable(GL_BLEND), glDepthMask,
    glEnable/glDisable(GL_CULL_FACE) (+glCullFace) and glPolygonMode for every
    mesh regardless of whether anything had changed: 2,506 state calls per frame
    at 500 meshes.
    """
    rec, mods = gl_env
    cam = mods["camera"].Camera()
    lights = mods["lights"].LightSetup()

    def state_calls(n: int) -> int:
        scene, _ = scene_with(n)
        r = mods["renderer"].GLRenderer()
        r.init_gl()
        r.resize(800, 600)
        r.render(scene, cam, lights)
        rec.reset()
        r.render(scene, cam, lights)
        return rec.group()["state"]

    s10, s100 = state_calls(10), state_calls(100)
    assert s10 == s100, (
        f"state calls scale with mesh count: {s10} at 10 meshes, {s100} at 100"
    )


# ----------------------------------------------------------------------
# Visibility / culling behaviour
# ----------------------------------------------------------------------

def test_invisible_mesh_issues_no_draw_call(gl_env, scene_with):
    rec, mods = gl_env
    scene, meshes = scene_with(3)
    r = mods["renderer"].GLRenderer()
    r.init_gl()
    r.resize(800, 600)
    r.render(scene, mods["camera"].Camera(), mods["lights"].LightSetup())

    meshes[1].visible = False
    rec.reset()
    r.render(scene, mods["camera"].Camera(), mods["lights"].LightSetup())
    assert rec.counts["glDrawArrays"] == 2


def test_invisible_node_prunes_whole_subtree(gl_env, make_mesh):
    """traverse_visible must skip descendants of an invisible node."""
    rec, mods = gl_env
    from faceforge.core.scene_graph import Scene, SceneNode

    scene = Scene()
    parent = SceneNode(name="parent")
    parent.mesh = make_mesh("parent")
    child = SceneNode(name="child")
    child.mesh = make_mesh("child")
    parent.add(child)
    scene.add(parent)
    scene.update()

    r = mods["renderer"].GLRenderer()
    r.init_gl()
    r.resize(800, 600)
    r.render(scene, mods["camera"].Camera(), mods["lights"].LightSetup())

    parent.visible = False
    rec.reset()
    r.render(scene, mods["camera"].Camera(), mods["lights"].LightSetup())
    assert rec.counts["glDrawArrays"] == 0, "child of invisible node was drawn"


def test_render_is_a_noop_before_init_gl(gl_env, scene_with):
    rec, mods = gl_env
    scene, _ = scene_with(4)
    r = mods["renderer"].GLRenderer()
    rec.reset()
    r.render(scene, mods["camera"].Camera(), mods["lights"].LightSetup())
    assert rec.total == 0


# ----------------------------------------------------------------------
# Geometry streaming
# ----------------------------------------------------------------------

def test_dirty_geometry_restreams_once_then_stops(gl_env, scene_with):
    """needs_update must trigger exactly one re-stream, then clear."""
    rec, mods = gl_env
    scene, meshes = scene_with(1)
    r = mods["renderer"].GLRenderer()
    r.init_gl()
    r.resize(800, 600)
    r.render(scene, mods["camera"].Camera(), mods["lights"].LightSetup())

    meshes[0].positions = meshes[0].positions * 2.0   # setter sets needs_update
    assert meshes[0].needs_update is True

    rec.reset()
    r.render(scene, mods["camera"].Camera(), mods["lights"].LightSetup())
    restream = rec.counts["glBufferSubData"]
    assert restream > 0, "dirty geometry was not re-streamed"
    assert meshes[0].needs_update is False

    rec.reset()
    r.render(scene, mods["camera"].Camera(), mods["lights"].LightSetup())
    assert rec.counts["glBufferSubData"] == 0, "clean geometry re-streamed anyway"


def test_destroy_releases_every_gl_object_it_created(gl_env, scene_with):
    """No leaked VAOs/VBOs/programs after destroy()."""
    rec, mods = gl_env
    scene, _ = scene_with(5)
    r = mods["renderer"].GLRenderer()
    r.init_gl()
    r.resize(800, 600)
    r.render(scene, mods["camera"].Camera(), mods["lights"].LightSetup())

    created = (rec.counts["glGenVertexArrays"] + rec.counts["glGenBuffers"])
    rec.reset()
    r.destroy()
    freed = (rec.counts["glDeleteVertexArrays"] + rec.counts["glDeleteBuffers"])

    assert created > 0
    assert freed == created, f"created {created} GL objects, freed {freed}"
    assert r._gl_meshes == {}
    assert r._shaders == {}
