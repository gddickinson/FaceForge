"""Numerical-equivalence evidence for the renderer performance changes.

Each optimisation deliberately changes the GL *call stream*, so an A/B on call
counts proves nothing about correctness.  These tests pin the quantities that
must not change:

* the shaded normal, after dropping the CPU-side ``uNormalMatrix``;
* every world matrix in the scene graph, after the dirty-flag rewrite;
* the output of ``collect_meshes``, after the traversal cache;
* the transparency sort order, after the switch to squared distance;

plus the two *intended* behaviour changes, asserted directly rather than
inferred: the render-mode blending policy and the clip-plane state.

A reference implementation of each replaced algorithm is written out longhand
here and compared against the live one under randomised mutation sequences.
"""

import numpy as np
import pytest

from faceforge.core.material import Material, RenderMode
from faceforge.core.math_utils import (
    mat3_normal, mat4_compose, mat4_look_at, quat_from_euler,
)
from faceforge.core.mesh import BufferGeometry, MeshInstance
from faceforge.core.scene_graph import Scene, SceneNode
from faceforge.rendering.gl_material import (
    _MODE_NEEDS_BLENDING, mode_needs_blending, needs_blending,
)

RNG_SEED = 20260829


def _mesh(name: str, n_tris: int = 4, rng=None) -> MeshInstance:
    rng = rng or np.random.default_rng(0)
    n = n_tris * 3
    return MeshInstance(
        name=name,
        geometry=BufferGeometry(
            positions=rng.normal(0, 25, n * 3).astype(np.float32),
            normals=np.tile([0.0, 0.0, 1.0], n).astype(np.float32),
            vertex_count=n,
        ),
        material=Material(),
    )


# ======================================================================
# 1. Normal matrix:  mat3_normal(MV) @ n  ==  mat3(MV) @ n  after normalise
# ======================================================================

def _random_similarity(rng) -> np.ndarray:
    """A transform from the space FaceForge actually produces: rotation,
    translation and *uniform* scale, viewed through a look-at matrix."""
    q = quat_from_euler(*rng.uniform(-np.pi, np.pi, 3))
    s = float(rng.uniform(0.2, 5.0))
    world = mat4_compose(rng.uniform(-300, 300, 3), q, np.array([s, s, s]))
    view = mat4_look_at(rng.uniform(-400, 400, 3),
                        rng.uniform(-50, 50, 3),
                        np.array([0.0, 0.0, 1.0]))
    return view @ world


def test_gpu_normal_matrix_matches_the_cpu_inverse_transpose():
    """Dropping uNormalMatrix must not change a single shaded normal.

    default.vert now computes ``mat3(uModelView) * aNormal`` instead of
    receiving ``inverse(transpose(mat3(uModelView)))`` from the CPU.  For a
    similarity transform (rotation + translation + uniform scale) the two differ
    only by the positive scalar s^2, which the fragment shader's normalize()
    removes -- so the *shaded* normal is identical.  This measures that on the
    transform space the app actually produces.

    Verified separately (grep for set_scale in src/faceforge): the tree has
    exactly one set_scale call site, anatomy/face.py:77, and it is uniform;
    renderer.scene_transform is None in the current app.

    Tolerance: the worst case measured over 400 randomised similarity
    transforms is 2.6e-8 rad (1.5e-6 degrees).  That residual is the
    conditioning of the explicit ``np.linalg.inv`` inside ``mat3_normal``, not a
    difference between the two formulas -- i.e. it is error the change *removes*
    rather than introduces.  For scale, the shader works in float32, where one
    ULP near 1.0 is already ~6e-8; the bound below is 1e-6 rad, still two orders
    of magnitude tighter than anything a float32 normal can represent.
    """
    rng = np.random.default_rng(RNG_SEED)
    worst = 0.0
    for _ in range(400):
        mv = _random_similarity(rng)
        n = rng.normal(size=3)
        n /= np.linalg.norm(n)

        old = mat3_normal(mv) @ n
        new = mv[:3, :3] @ n
        old /= np.linalg.norm(old)
        new /= np.linalg.norm(new)

        # angle between the two unit normals, in radians
        worst = max(worst, float(np.arccos(np.clip(old @ new, -1.0, 1.0))))

    assert worst < 1e-6, f"shaded normal deviates by up to {worst:.3e} rad"
    # Regression guard: if this ever tightens to exact agreement, mat3_normal
    # was changed; if it loosens past 1e-7 something real broke.
    assert worst < 1e-7


def test_gpu_normal_matrix_would_be_wrong_under_non_uniform_scale():
    """The counter-example that makes the guard comment in default.vert real.

    If anyone ever introduces a non-uniform node scale, this is the error they
    will see, and uNormalMatrix must come back.
    """
    rng = np.random.default_rng(RNG_SEED + 1)
    mv = mat4_compose(np.zeros(3), quat_from_euler(0.3, 0.4, 0.5),
                      np.array([1.0, 1.0, 8.0]))
    n = np.array([0.0, 1.0, 1.0]) / np.sqrt(2.0)
    old = mat3_normal(mv) @ n
    new = mv[:3, :3] @ n
    old /= np.linalg.norm(old)
    new /= np.linalg.norm(new)
    angle = float(np.arccos(np.clip(old @ new, -1.0, 1.0)))
    assert angle > 0.5, (
        "non-uniform scale no longer breaks the mat3(uModelView) shortcut -- "
        "the warning in default.vert can be relaxed"
    )


# ======================================================================
# 2. Scene-graph dirty flags:  world matrices identical to the old algorithm
# ======================================================================

def _reference_world_matrices(root: SceneNode) -> dict[str, np.ndarray]:
    """The pre-fix update_world_matrix, written out longhand.

    Unconditional: recompose the local matrix if _matrix_dirty, multiply the
    parent's world matrix, recurse into every child, every frame.
    """
    out: dict[str, np.ndarray] = {}

    def walk(node: SceneNode, parent_world) -> None:
        local = mat4_compose(node.position, node.quaternion, node.scale)
        world = local.copy() if parent_world is None else parent_world @ local
        out[node.name] = world
        for child in node.children:
            walk(child, world)

    walk(root, None)
    return out


def _build_graph(rng, depth: int = 4, breadth: int = 3):
    """A randomly shaped graph with unique node names."""
    scene = Scene()
    nodes: list[SceneNode] = []
    frontier = [scene]
    counter = 0
    for _ in range(depth):
        new_frontier = []
        for parent in frontier:
            for _ in range(rng.integers(1, breadth + 1)):
                node = SceneNode(name=f"n{counter}")
                counter += 1
                node.set_position(*rng.uniform(-50, 50, 3))
                node.set_quaternion(quat_from_euler(*rng.uniform(-1, 1, 3)))
                s = float(rng.uniform(0.5, 2.0))
                node.set_scale(s, s, s)
                parent.add(node)
                nodes.append(node)
                new_frontier.append(node)
        frontier = new_frontier
    return scene, nodes


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_dirty_flag_update_matches_the_unconditional_reference(seed):
    """Every world matrix must equal what the old unconditional walk produced,
    through a randomised sequence of transform edits and reparents."""
    rng = np.random.default_rng(RNG_SEED + seed)
    scene, nodes = _build_graph(rng)
    scene.update()

    for step in range(40):
        node = nodes[int(rng.integers(len(nodes)))]
        kind = int(rng.integers(4))
        if kind == 0:
            node.set_position(*rng.uniform(-80, 80, 3))
        elif kind == 1:
            node.set_quaternion(quat_from_euler(*rng.uniform(-2, 2, 3)))
        elif kind == 2:
            s = float(rng.uniform(0.3, 3.0))
            node.set_scale(s, s, s)
        else:
            # reparent, avoiding cycles
            target = nodes[int(rng.integers(len(nodes)))]
            anc = target
            while anc is not None and anc is not node:
                anc = anc.parent
            if anc is None:
                target.add(node)

        scene.update()
        ref = _reference_world_matrices(scene)
        for n in [scene] + nodes:
            np.testing.assert_allclose(
                n.world_matrix, ref[n.name], atol=1e-12,
                err_msg=f"seed={seed} step={step} node={n.name}",
            )


def test_clean_scene_update_touches_no_matrices():
    """The point of the rewrite: a static scene must do no work at all.

    ``_matrix_dirty`` previously gated only the LOCAL matrix, so the world
    matrix product and the recursion into every child ran unconditionally for
    all ~900 mesh nodes on every frame.
    """
    rng = np.random.default_rng(RNG_SEED)
    scene, nodes = _build_graph(rng)
    scene.update()

    before = [n.world_matrix.copy() for n in nodes]
    ids_before = [id(n.world_matrix) for n in nodes]

    assert not any(n._world_dirty or n._subtree_dirty for n in nodes), (
        "nodes are still flagged dirty after update()"
    )
    scene.update()

    for n, w, i in zip(nodes, before, ids_before):
        np.testing.assert_array_equal(n.world_matrix, w)
        # written in place, so cached (mesh, world_matrix) tuples stay valid
        assert id(n.world_matrix) == i, "world_matrix object was replaced"


def test_world_matrix_is_written_in_place_when_it_changes():
    """collect_meshes caches (mesh, world_matrix) tuples, so the array object
    must be reused, not reassigned, when a transform moves."""
    scene = Scene()
    node = SceneNode(name="n")
    scene.add(node)
    scene.update()
    arr = node.world_matrix
    node.set_position(9.0, 8.0, 7.0)
    scene.update()
    assert node.world_matrix is arr
    np.testing.assert_allclose(node.get_world_position(), [9, 8, 7])


# ======================================================================
# 3. collect_meshes cache:  identical to the old traverse_visible closure
# ======================================================================

def _reference_collect(scene: Scene):
    """The pre-fix collect_meshes, written out longhand."""
    result = []

    def _collect(node):
        if node.mesh is not None and node.mesh.visible:
            result.append((node.mesh, node.world_matrix))

    scene.traverse_visible(_collect)
    return result


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_collect_meshes_cache_matches_the_uncached_traversal(seed):
    """Order and content must match through visibility and topology churn.

    Both flavours of visibility matter: ``node.visible`` (prunes a subtree, and
    invalidates the cache) and ``mesh.visible`` (re-tested on every call,
    because the anatomy toggles flip it constantly).
    """
    rng = np.random.default_rng(RNG_SEED + 100 + seed)
    scene, nodes = _build_graph(rng, depth=3, breadth=3)
    for i, n in enumerate(nodes):
        if i % 2 == 0:
            n.mesh = _mesh(f"m{i}", rng=rng)
    scene.update()

    for step in range(60):
        kind = int(rng.integers(4))
        node = nodes[int(rng.integers(len(nodes)))]
        if kind == 0:
            node.visible = not node.visible
        elif kind == 1 and node.mesh is not None:
            node.mesh.visible = not node.mesh.visible
        elif kind == 2 and node.parent is not None:
            node.parent.remove(node)
        else:
            target = nodes[int(rng.integers(len(nodes)))]
            anc = target
            while anc is not None and anc is not node:
                anc = anc.parent
            if anc is None:
                target.add(node)
        scene.update()

        got = scene.collect_meshes()
        want = _reference_collect(scene)
        assert [m.name for m, _ in got] == [m.name for m, _ in want], (
            f"seed={seed} step={step} kind={kind}"
        )
        for (gm, gw), (wm, ww) in zip(got, want):
            assert gm is wm
            assert gw is ww, "cached world matrix is not the node's own array"


def test_collect_cache_is_reused_when_nothing_changes():
    scene = Scene()
    for i in range(5):
        node = SceneNode(name=f"n{i}")
        node.mesh = _mesh(f"m{i}")
        scene.add(node)
    scene.update()
    first = scene.collect_meshes()
    assert scene.collect_meshes() is first, "traversal was rebuilt for nothing"

    scene.children[0].visible = False
    assert scene.collect_meshes() is not first, "visibility change was missed"


# ======================================================================
# 4. Transparency sort:  squared distance orders identically to distance
# ======================================================================

def test_squared_distance_sort_matches_distance_sort():
    """The sort key became ``delta @ delta`` instead of ``norm(delta)``.

    x -> x^2 is monotonic on non-negative reals, so the ordering is identical;
    this measures it on the actual magnitudes (BodyParts3D coordinates reach the
    hundreds, so the squares reach ~10^5 -- well inside float64).
    """
    rng = np.random.default_rng(RNG_SEED + 7)
    for _ in range(200):
        deltas = rng.normal(0, 250, size=(40, 3))
        by_dist = np.argsort(-np.linalg.norm(deltas, axis=1), kind="stable")
        by_sq = np.argsort(-(deltas * deltas).sum(axis=1), kind="stable")
        np.testing.assert_array_equal(by_dist, by_sq)


def test_centroid_sort_separates_meshes_that_share_a_node_origin():
    """Why the sort key moved from the node origin to the geometry centroid.

    BodyParts3D structures sit far from their node origins and many share the
    body root, so ``world[:3, 3]`` gave every mesh in a group the same depth and
    the back-to-front sort did nothing at all.
    """
    from faceforge.rendering.renderer import GLRenderer

    r = GLRenderer()
    world = np.eye(4, dtype=np.float64)
    near = MeshInstance(
        name="near",
        geometry=BufferGeometry(
            positions=(np.tile([0.0, 0.0, 10.0], 3)).astype(np.float32),
            normals=np.tile([0.0, 0.0, 1.0], 3).astype(np.float32),
            vertex_count=3),
        material=Material())
    far = MeshInstance(
        name="far",
        geometry=BufferGeometry(
            positions=(np.tile([0.0, 0.0, -400.0], 3)).astype(np.float32),
            normals=np.tile([0.0, 0.0, 1.0], 3).astype(np.float32),
            vertex_count=3),
        material=Material())

    # Identical node origins -- the old key could not tell these apart.
    assert tuple(world[:3, 3]) == (0.0, 0.0, 0.0)
    c_near = r._world_centroid(near, world)
    c_far = r._world_centroid(far, world)
    assert c_near[2] == pytest.approx(10.0)
    assert c_far[2] == pytest.approx(-400.0)


# ======================================================================
# 5. Blending policy (intended behaviour change)
# ======================================================================

def test_mode_blending_table_is_exactly_the_five_alpha_computing_modes():
    assert _MODE_NEEDS_BLENDING == {
        RenderMode.XRAY, RenderMode.HOLOGRAM, RenderMode.BLUEPRINT,
        RenderMode.ETHEREAL, RenderMode.POINTS,
    }


@pytest.mark.parametrize("mode", list(RenderMode))
def test_opaque_material_still_blends_in_the_alpha_computing_modes(mode):
    """The half of the opacity fix that must land with it.

    With the loader default corrected to opacity 1.0 / transparent False, these
    five modes would render as dark solids if blending were decided by the
    material alone -- their whole visual identity is a fractional alpha.
    """
    mat = Material(render_mode=mode, opacity=1.0, transparent=False)
    assert needs_blending(mat) is mode_needs_blending(mode), (
        f"{mode.name}: blend decision does not follow the mode table"
    )


@pytest.mark.parametrize("mode", list(RenderMode))
def test_transparent_material_always_blends_except_in_opaque_mode(mode):
    mat = Material(render_mode=mode, opacity=0.4, transparent=True)
    expected = mode is not RenderMode.OPAQUE
    assert needs_blending(mat) is expected


def test_opaque_mode_overrides_everything():
    mat = Material(render_mode=RenderMode.OPAQUE, opacity=0.1, transparent=True)
    assert needs_blending(mat) is False


def test_default_material_is_not_blended():
    """The anatomy is opaque now; that is the point of the opacity change."""
    assert needs_blending(Material()) is False


def test_stl_batch_loader_defaults_to_opaque():
    """941 of ~950 structures inherit this default."""
    import inspect

    from faceforge.loaders import stl_batch_loader

    src = inspect.getsource(stl_batch_loader)
    assert 'defn.get("opacity", 1.0)' in src, (
        "the STL loader no longer defaults opacity to 1.0"
    )
    assert 'defn.get("opacity", 0.7)' not in src


# ======================================================================
# 6. Clip plane (intended behaviour change)
# ======================================================================

def test_clip_distance_state_follows_the_clip_plane(gl_env, scene_with):
    """GL_CLIP_DISTANCE0 must be enabled exactly while the cutaway is on, and
    toggled once -- not per mesh, and not per frame."""
    rec, mods = gl_env
    scene, _ = scene_with(6)
    r = mods["renderer"].GLRenderer()
    r.init_gl()
    r.resize(800, 600)
    cam, lights = mods["camera"].Camera(), mods["lights"].LightSetup()
    r.render(scene, cam, lights)
    assert r._clip_distance_on is False

    r.set_clip_plane((1.0, 0.0, 0.0), -5.0)
    rec.reset()
    r.render(scene, cam, lights)
    assert r._clip_distance_on is True
    enable_calls = rec.counts["glEnable"]

    rec.reset()
    r.render(scene, cam, lights)          # steady state, clip still on
    assert r._clip_distance_on is True
    assert rec.counts["glEnable"] < enable_calls, (
        "GL_CLIP_DISTANCE0 is re-enabled every frame"
    )

    r.clear_clip_plane()
    rec.reset()
    r.render(scene, cam, lights)
    assert r._clip_distance_on is False


def test_umodelmatrix_is_only_uploaded_when_something_reads_vworldpos(
        gl_env, scene_with):
    """uModelMatrix feeds vWorldPos, read only by the clip test and by
    HOLOGRAM / BLUEPRINT."""
    rec, mods = gl_env
    cam, lights = mods["camera"].Camera(), mods["lights"].LightSetup()

    def mat4_uploads(mode, clip: bool) -> int:
        scene, meshes = scene_with(10)
        for m in meshes:
            m.material.render_mode = mode
        r = mods["renderer"].GLRenderer()
        r.init_gl()
        r.resize(800, 600)
        if clip:
            r.set_clip_plane((1.0, 0.0, 0.0), 0.0)
        r.render(scene, cam, lights)
        rec.reset()
        r.render(scene, cam, lights)
        return rec.counts["glUniformMatrix4fv"]

    from faceforge.core.material import RenderMode as RM

    # 10 uModelView + 1 uProjection
    assert mat4_uploads(RM.SOLID, clip=False) == 11
    # + 10 uModelMatrix
    assert mat4_uploads(RM.SOLID, clip=True) == 21
    assert mat4_uploads(RM.HOLOGRAM, clip=False) == 21
    assert mat4_uploads(RM.BLUEPRINT, clip=False) == 21
