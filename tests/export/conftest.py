"""Fixtures for the export tests.

Most of this module's tests run in the *fast* tier, which means they cannot
touch ``assets/stl`` (see ``tests/conftest.py``: the fast tier must pass on a
checkout with no BodyParts3D dataset).  So the geometry here is procedural --
two axis-aligned boxes carrying the same ``source_id`` / ``ontology_id`` /
``preferred_label`` fields a real BodyParts3D mesh carries.

Procedural geometry is not a weaker test for what these modules do.  Every
claim being checked is about the *file*: vertex and triangle counts surviving a
round-trip, provenance strings arriving intact, voxel positions matching the
tags that describe them.  A box makes those claims sharper, because the
expected answer is known in closed form.  The two tests that genuinely need
real anatomy -- the ones asserting that BodyParts3D coordinates are LPS -- are
marked ``slow`` and read the dataset.
"""

from __future__ import annotations

import numpy as np
import pytest


def box_mesh(
    name: str,
    *,
    source_id: str = "",
    ontology_id: str = "",
    preferred_label: str = "",
    centre: tuple[float, float, float] = (0.0, 0.0, 0.0),
    size: tuple[float, float, float] = (40.0, 40.0, 40.0),
    color: tuple[float, float, float] = (0.8, 0.7, 0.6),
):
    """An indexed axis-aligned box: 24 vertices, 12 triangles, outward normals.

    24 vertices rather than 8 so each face has its own normals, which is what
    every real mesh in the dataset looks like and what makes the normal-baking
    path meaningful.
    """
    from faceforge.core.material import Material
    from faceforge.core.mesh import BufferGeometry, MeshInstance

    cx, cy, cz = centre
    hx, hy, hz = (s / 2.0 for s in size)
    # (axis, sign) -> the four corners of that face, wound counter-clockwise
    # seen from outside.
    faces = [
        ((1, 0, 0), [(hx, -hy, -hz), (hx, hy, -hz), (hx, hy, hz), (hx, -hy, hz)]),
        ((-1, 0, 0), [(-hx, -hy, hz), (-hx, hy, hz), (-hx, hy, -hz), (-hx, -hy, -hz)]),
        ((0, 1, 0), [(-hx, hy, -hz), (-hx, hy, hz), (hx, hy, hz), (hx, hy, -hz)]),
        ((0, -1, 0), [(-hx, -hy, hz), (-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz)]),
        ((0, 0, 1), [(-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)]),
        ((0, 0, -1), [(hx, -hy, -hz), (-hx, -hy, -hz), (-hx, hy, -hz), (hx, hy, -hz)]),
    ]
    positions, normals, indices = [], [], []
    for normal, corners in faces:
        base = len(positions)
        for x, y, z in corners:
            positions.append((x + cx, y + cy, z + cz))
            normals.append(normal)
        indices += [base, base + 1, base + 2, base, base + 2, base + 3]

    geom = BufferGeometry(
        positions=np.asarray(positions, dtype=np.float32).ravel(),
        normals=np.asarray(normals, dtype=np.float32).ravel(),
        indices=np.asarray(indices, dtype=np.uint32),
    )
    return MeshInstance(
        name=name,
        geometry=geom,
        material=Material(color=color, opacity=1.0),
        source_id=source_id,
        ontology_id=ontology_id,
        preferred_label=preferred_label,
    )


#: The two structures every synthetic-scene test shares.  The names are chosen
#: so that ``TissueMapper.classify`` returns a *known* tissue: "Mandible"
#: matches the bone keyword list and "Heart" the organ list, which is what
#: makes the expected CT tissue-table values (0.95 and 0.40) predictable.
SYNTHETIC_STRUCTURES = (
    {
        "name": "Mandible",
        "source_id": "FMA52748",
        "ontology_id": "FMA:52748",
        "preferred_label": "Mandible",
        "centre": (-50.0, 0.0, 0.0),
        "expected_tissue": "bone",
        "expected_ct_value": 0.95,
    },
    {
        "name": "Heart",
        "source_id": "FMA7088",
        "ontology_id": "FMA:7088",
        "preferred_label": "Heart",
        "centre": (50.0, 0.0, 0.0),
        "expected_tissue": "organ",
        "expected_ct_value": 0.40,
    },
)


@pytest.fixture
def synthetic_scene():
    """A two-box scene with BodyParts3D-shaped provenance on both meshes.

    The boxes span z in [-20, 20] and sit at x = -50 and x = +50, so an axial
    volume centred on the origin with a few 5 mm slices intersects both, and
    the two tissue values are separated in x.
    """
    from faceforge.core.scene_graph import Scene, SceneNode

    scene = Scene()
    meshes = []
    for spec in SYNTHETIC_STRUCTURES:
        mesh = box_mesh(
            spec["name"], source_id=spec["source_id"],
            ontology_id=spec["ontology_id"],
            preferred_label=spec["preferred_label"],
            centre=spec["centre"], size=(40.0, 40.0, 40.0),
        )
        node = SceneNode(name=spec["source_id"])
        node.mesh = mesh
        scene.add(node)
        meshes.append(mesh)
    scene.update()
    return scene


@pytest.fixture
def mixed_scene(synthetic_scene):
    """The synthetic scene plus one mesh with *no* provenance at all.

    Procedural geometry (the scan-plane quad, a test primitive) is not
    BodyParts3D and must not be labelled as though it were; several tests check
    that the exporters report it as unattributed rather than quietly claiming
    it.
    """
    from faceforge.core.scene_graph import SceneNode

    mesh = box_mesh("procedural_marker", centre=(0.0, 0.0, 60.0),
                    size=(10.0, 10.0, 10.0))
    node = SceneNode(name="marker")
    node.mesh = mesh
    synthetic_scene.add(node)
    synthetic_scene.update()
    return synthetic_scene


@pytest.fixture
def ct_volume(synthetic_scene):
    """A small axial CT volume over the synthetic scene, reduction='max'.

    ``max`` because that is the reduction under which pixel values are exactly
    tissue-table entries, which is the precondition for ``hu_mode='class'``.

    ``slab_depth`` is 50 mm against a 5 mm slice spacing, which is deliberate
    and is worth understanding, because it is the clearest demonstration of the
    limitation documented in :mod:`faceforge.export.hounsfield`: the scanner
    samples *surfaces*.  A 5 mm slab sitting entirely inside a 40 mm solid box
    crosses no triangle at all and comes back empty -- there is no volumetric
    material model for it to sample.  A 50 mm slab reaches one of the box's
    faces, so the ray registers a crossing and the pixel takes that tissue's
    value.  Real BodyParts3D anatomy hides this most of the time because thin
    shells and many nested structures put a surface in almost every slab; a
    single convex box does not.
    """
    from faceforge.export.volume import scan_volume

    return scan_volume(
        synthetic_scene, orientation="axial", centre=(0.0, 0.0, 0.0),
        field_width=200.0, field_height=200.0, resolution=32,
        slices=4, slice_spacing=5.0, slab_depth=50.0,
        mode="ct", reduction="max",
    )


@pytest.fixture(autouse=True)
def _no_leaked_session():
    """Fail the test that leaked a Session, not the next one.

    Same guard as ``tests/session/conftest.py``: a process holds at most one
    Session, so a leak is otherwise reported against an innocent test.
    """
    from faceforge import session as fs

    before = fs.Session.active()
    assert before is None, (
        f"a Session was already live before this test started: {before!r}"
    )
    yield
    leaked = fs.Session.active()
    if leaked is not None:
        leaked.close()
        pytest.fail(f"this test left a Session open: {leaked!r}")
