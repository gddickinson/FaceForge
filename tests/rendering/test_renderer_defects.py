"""Tests that pin down confirmed defects in :mod:`faceforge.rendering.renderer`.

Each test asserts the *correct* behaviour.  They were written during the audit
as ``xfail(strict=True)``, so the suite stayed green while the defects stood;
every marker in this module has since been removed because the defect it
encoded is fixed.  Each test's docstring names the defect it used to pin, so a
regression reintroduces a named failure rather than an anonymous one.

Historical note on the original arrangement: each test was marked
``xfail(strict=True)``,
so the suite stays green while the defect exists and fails loudly the moment
someone fixes it without removing the marker.  Every marker names the defect
it tracks; see ``defects.md``.
"""

import gc

import numpy as np
import pytest


def _record_draw_order(renderer) -> list:
    """Replace ``_draw_mesh`` with a recorder and return the list it fills."""
    order = []

    def _fake(mesh, world, view, proj, lights):
        order.append(mesh.name)

    renderer._draw_mesh = _fake
    return order


# ----------------------------------------------------------------------
# DEFECT gl-mesh-cache-id-reuse
# ----------------------------------------------------------------------

# `slow`: 3.15 s, and it is 200 explicit `gc.collect()` calls -- a measurement
# of CPython allocator behaviour, not of this project's code.  It establishes a
# premise for the test below rather than guarding a regression, so the fast
# tier does not need it.
@pytest.mark.slow
def test_meshinstance_addresses_are_heavily_recycled(make_mesh):
    """Establishes the premise: id(MeshInstance) is not a stable unique key.

    Measured, not assumed -- CPython hands the same address back over and over
    for same-shaped objects, which is what makes an ``id()``-keyed cache with
    no eviction a correctness hazard rather than merely a leak.
    """
    ids = []
    for _ in range(200):
        m = make_mesh("throwaway")
        ids.append(id(m))
        del m
        gc.collect()

    distinct = len(set(ids))
    assert distinct < 50, (
        f"expected heavy address reuse, saw {distinct} distinct ids in 200 "
        "create-and-free cycles"
    )


def test_gl_mesh_cache_rejects_a_foreign_handle(gl_env, make_mesh):
    """A cache hit must be validated against the mesh's own geometry.

    FIXED (was DEFECT gl-mesh-cache-id-reuse).  ``_gl_meshes`` used to be keyed
    by ``id(MeshInstance)`` with no identity validation, and ``remove_mesh()``
    was called from nowhere in the codebase, so entries were never evicted.
    When a freed mesh's address was recycled (see
    test_meshinstance_addresses_are_heavily_recycled) the new mesh was handed
    the old mesh's VAO and rendered the wrong geometry with no error.

    Ownership now lives on ``mesh.gl_handle``, so the GLMesh's lifetime tracks
    the mesh object rather than an address the allocator is free to reuse.  The
    collision below is injected deterministically rather than waiting on the
    allocator.
    """
    rec, mods = gl_env
    r = mods["renderer"].GLRenderer()
    r.init_gl()

    victim = make_mesh("victim")
    gl_victim = r._ensure_gl_mesh(victim)
    victim_geom = victim.geometry

    # A different mesh, never uploaded, whose address collides with the freed one.
    fresh = make_mesh("fresh")
    assert fresh.geometry is not victim_geom
    r._gl_meshes.clear()
    r._gl_meshes[id(fresh)] = gl_victim      # the recycled-address situation

    handed_out = r._ensure_gl_mesh(fresh)
    assert handed_out is not gl_victim, (
        "a never-uploaded mesh was handed another mesh's GLMesh; the renderer "
        "will draw the wrong geometry with no error"
    )


def test_detaching_a_node_evicts_its_gl_resources(gl_env, scene_with):
    """Detaching a node must release its VAO/VBOs on the next frame.

    Replaces test_remove_mesh_is_the_only_eviction_path_and_is_never_called,
    which pinned the old behaviour (cache grows forever) and explicitly asked to
    be updated once eviction was added.  ``remove_mesh`` is still the only
    eviction path, but ``SceneNode.remove`` now queues the detached subtree's
    meshes and the renderer drains that queue at the top of each frame -- so
    every one of the ~10 detach sites across the tree evicts, without any of
    them needing a renderer reference.
    """
    rec, mods = gl_env
    scene, meshes = scene_with(3)
    r = mods["renderer"].GLRenderer()
    r.init_gl()
    r.resize(800, 600)
    r.render(scene, mods["camera"].Camera(), mods["lights"].LightSetup())
    assert len(r._gl_meshes) == 3
    assert all(m.gl_handle is not None for m in meshes)

    for child in list(scene.children):
        scene.remove(child)
    scene.update()
    rec.reset()
    r.render(scene, mods["camera"].Camera(), mods["lights"].LightSetup())

    assert rec.counts["glDrawArrays"] == 0, "detached meshes were still drawn"
    assert len(r._gl_meshes) == 0, "detached meshes were not evicted"
    assert all(m.gl_handle is None for m in meshes), "gl_handle not cleared"
    assert rec.counts["glDeleteVertexArrays"] == 3, "VAOs were not deleted"


def test_reparenting_does_not_evict(gl_env, make_mesh):
    """A reparent is remove-then-add and must NOT free the GPU buffers.

    ``SceneNode.add`` detaches from the old parent first, which queues the
    subtree for eviction; the queue is filtered against the live graph before
    anything is destroyed.
    """
    rec, mods = gl_env
    from faceforge.core.scene_graph import Scene, SceneNode

    scene = Scene()
    group_a = SceneNode(name="a")
    group_b = SceneNode(name="b")
    node = SceneNode(name="n")
    node.mesh = make_mesh("m")
    group_a.add(node)
    scene.add(group_a)
    scene.add(group_b)
    scene.update()

    r = mods["renderer"].GLRenderer()
    r.init_gl()
    r.resize(800, 600)
    r.render(scene, mods["camera"].Camera(), mods["lights"].LightSetup())
    handle = node.mesh.gl_handle
    assert handle is not None

    group_b.add(node)           # reparent
    scene.update()
    rec.reset()
    r.render(scene, mods["camera"].Camera(), mods["lights"].LightSetup())

    assert node.mesh.gl_handle is handle, "reparenting destroyed the GL mesh"
    assert rec.counts["glDeleteVertexArrays"] == 0
    assert rec.counts["glDrawArrays"] == 1, "reparented mesh was not drawn"


# ----------------------------------------------------------------------
# DEFECT global-opaque-from-first-mesh
# ----------------------------------------------------------------------

def test_one_opaque_mesh_does_not_disable_transparency_sorting(gl_env, make_mesh):
    """Transparent meshes must still be drawn last, back-to-front.

    FIXED (was DEFECT global-opaque-from-first-mesh).  ``render()`` decided
    ``global_opaque`` from ``mesh_list[0][0]`` alone, so one mesh in OPAQUE mode
    disabled back-to-front sorting for every genuinely transparent mesh in the
    scene.  The decision is per mesh now, via
    ``gl_material.needs_blending(material)`` -- the same predicate that drives
    the blend state, so the two cannot disagree.
    """
    rec, mods = gl_env
    from faceforge.core.material import RenderMode
    from faceforge.core.scene_graph import Scene, SceneNode

    scene = Scene()
    # First in traversal order: a mesh whose mode happens to be OPAQUE.
    first = make_mesh("opaque_first")
    first.material.render_mode = RenderMode.OPAQUE
    # Then two genuinely transparent meshes at different depths.
    near = make_mesh("near_transparent", opacity=0.4, transparent=True)
    far = make_mesh("far_transparent", opacity=0.4, transparent=True)

    for name, mesh, z in (("a", first, 0.0), ("b", near, -10.0), ("c", far, -200.0)):
        node = SceneNode(name=name)
        node.mesh = mesh
        node.set_position(0.0, 0.0, z)
        scene.add(node)
    scene.update()

    r = mods["renderer"].GLRenderer()
    r.init_gl()
    r.resize(800, 600)
    order = _record_draw_order(r)
    r.render(scene, mods["camera"].Camera(), mods["lights"].LightSetup())

    assert order[0] == "opaque_first", "opaque mesh should be drawn first"
    assert order[1:] == ["far_transparent", "near_transparent"], (
        f"transparent meshes not sorted back-to-front: {order}"
    )


# ----------------------------------------------------------------------
# DEFECT render-split-mode-not-restored
# ----------------------------------------------------------------------

def test_render_split_restores_render_mode_when_draw_raises(gl_env, scene_with):
    """FIXED (was DEFECT render-split-mode-not-restored).

    ``render_split()`` overwrote ``mesh.material.render_mode`` and restored it
    after ``_draw_mesh`` returned, with no try/finally -- one raise mid-frame
    left every mesh stuck in the comparison mode permanently.
    """
    rec, mods = gl_env
    from faceforge.core.material import RenderMode

    scene, meshes = scene_with(2)
    for m in meshes:
        m.material.render_mode = RenderMode.SOLID

    r = mods["renderer"].GLRenderer()
    r.init_gl()
    r.resize(800, 600)

    def _boom(*a, **kw):
        raise RuntimeError("simulated GL failure mid-frame")

    r._draw_mesh = _boom
    with pytest.raises(RuntimeError):
        r.render_split(
            scene, mods["camera"].Camera(), mods["lights"].LightSetup(),
            {"render_mode": RenderMode.XRAY}, {"render_mode": RenderMode.XRAY},
        )

    assert [m.material.render_mode for m in meshes] == [RenderMode.SOLID] * 2, (
        "render_mode left overwritten after an exception"
    )


# ----------------------------------------------------------------------
# DEFECT scene-diag-attr-unguarded
# ----------------------------------------------------------------------

def test_scene_diag_start_is_not_declared_in_init(gl_env):
    """``_scene_diag_start`` is created mid-render and probed with hasattr().

    Records current behaviour: the attribute is absent on a fresh renderer and
    is conjured inside ``render()``.  Any state that only exists after a
    particular frame path has run is invisible to ``__init__`` readers and to
    static analysis.
    """
    rec, mods = gl_env
    r = mods["renderer"].GLRenderer()
    assert not hasattr(r, "_scene_diag_start"), (
        "_scene_diag_start is now initialised in __init__ -- good; update this "
        "test and the scene-diag-attr-unguarded defect note"
    )


def test_render_does_no_per_frame_diagnostic_work(gl_env, scene_with):
    """The scene-transform diagnostic block is gone from the hot path.

    Replaces test_diagnostic_logging_runs_on_every_frame_forever, which pinned
    the broken behaviour: ``_scene_transform_logged`` served two purposes at
    once (arming a 10-frame body-mesh dump, and a one-shot scene_transform log),
    so setting it in one path silently disabled the other, and the dump then ran
    ``np.allclose`` + ``np.diag`` over up to 20 meshes on every frame with no
    way to disarm.

    The whole block is removed, so the flag must now stay untouched by
    ``render()`` and ``_scene_diag_start`` must never come into existence.
    """
    rec, mods = gl_env
    scene, meshes = scene_with(1)
    scene.children[0].set_scale(2.0, 2.0, 2.0)   # non-identity world diag
    scene.update()

    r = mods["renderer"].GLRenderer()
    r.init_gl()
    r.resize(800, 600)
    r.render(scene, mods["camera"].Camera(), mods["lights"].LightSetup())

    assert r._scene_transform_logged is False, (
        "render() still consumes the one-shot diagnostic flag"
    )
    assert not hasattr(r, "_scene_diag_start"), (
        "the multi-frame diagnostic dump is still armed inside render()"
    )

    # A real scene_transform must not resurrect it either.
    r.scene_transform = np.eye(4, dtype=np.float64) * 2.0
    r.scene_transform[3, 3] = 1.0
    r.render(scene, mods["camera"].Camera(), mods["lights"].LightSetup())
    assert r._scene_transform_logged is False
    assert not hasattr(r, "_scene_diag_start")
