"""Image-equivalence tests for the triangle-centric scanner engine.

``ScannerEngine.scan`` was rewritten from tile-centric (every triangle in a
16x16 tile tested against all 256 of its rays) to triangle-centric (each
triangle tested only against the rays inside its own projected pixel box).
The maths is unchanged, but float32 hit sums accumulate in a different
order, so images agree to float32 rounding rather than bit-exactly.

These tests pin the rewrite against the frozen tile-centric implementation
in ``_reference_scanner`` across orientations, resolutions, reductions and
imaging modes.
"""

from __future__ import annotations

import numpy as np
import pytest

from faceforge.core.material import Material
from faceforge.core.mesh import BufferGeometry, MeshInstance
from faceforge.scanner.engine import ScannerEngine
from faceforge.scanner.tissue_map import TissueMapper

from tests.scanner._reference_scanner import ref_scan


ORIENTATIONS = {
    "axial": dict(
        normal=np.array([0, 0, -1], np.float32),
        right=np.array([1, 0, 0], np.float32),
        up=np.array([0, -1, 0], np.float32),
    ),
    "coronal": dict(
        normal=np.array([0, -1, 0], np.float32),
        right=np.array([1, 0, 0], np.float32),
        up=np.array([0, 0, -1], np.float32),
    ),
    "sagittal": dict(
        normal=np.array([1, 0, 0], np.float32),
        right=np.array([0, -1, 0], np.float32),
        up=np.array([0, 0, -1], np.float32),
    ),
}

# Names chosen so TissueMapper classifies them as different tissues, giving
# distinct intensities in every imaging mode.
_MESH_SPECS = [
    ("femur bone", (0.9, 0.9, 0.85), (0.0, 0.0, 0.0), 9.0),
    ("biceps muscle", (0.7, 0.2, 0.2), (3.0, 1.0, -2.0), 6.0),
    ("liver", (0.5, 0.3, 0.25), (-3.0, -1.0, 2.0), 7.0),
    ("brain cortex", (0.85, 0.8, 0.8), (0.0, 3.0, 4.0), 5.0),
]


def _blob(center, radius, n_tri, seed) -> BufferGeometry:
    """A closed-ish triangle soup on a jittered sphere: irregular projected
    footprints, so triangles land in many different pixel-box shapes."""
    rng = np.random.default_rng(seed)
    pts = rng.normal(0, 1, (n_tri * 3, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    pts *= radius * (1.0 + 0.25 * rng.random((len(pts), 1)))
    pts += np.asarray(center, dtype=np.float64)
    pos = pts.astype(np.float32)
    tri = pos.reshape(-1, 3, 3)
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = nrm / np.maximum(ln, 1e-9)
    normals = np.repeat(nrm, 3, axis=0).astype(np.float32)
    return BufferGeometry(
        positions=pos.ravel(), normals=normals.ravel(), vertex_count=len(pos)
    )


@pytest.fixture(scope="module")
def engine() -> ScannerEngine:
    meshes = []
    for i, (name, color, center, radius) in enumerate(_MESH_SPECS):
        geom = _blob(center, radius, 700, seed=i)
        mesh = MeshInstance(
            name=name, geometry=geom, material=Material(color=color)
        )
        meshes.append((mesh, np.eye(4, dtype=np.float32)))
    eng = ScannerEngine(TissueMapper())
    eng.cache_meshes(meshes)
    assert len(eng._cache) == len(_MESH_SPECS)
    return eng


def _kwargs(orient, res, mode, reduction):
    return dict(
        origin=np.array([0.0, 0.0, 0.0], np.float32),
        **ORIENTATIONS[orient],
        width=30.0, height=30.0, depth=12.0,
        resolution=res, mode=mode, reduction=reduction,
    )


CASES = [
    ("axial", 64, "ct", "mean"),
    ("coronal", 64, "ct", "mean"),
    ("sagittal", 64, "ct", "mean"),
    ("axial", 128, "ct", "mean"),
    ("axial", 96, "mri_t1", "mean"),
    ("axial", 96, "mri_t2", "max"),
    ("coronal", 96, "xray", "sum"),
    ("sagittal", 96, "ct", "min"),
    ("axial", 96, "anatomical", "mean"),
]


@pytest.mark.parametrize("orient,res,mode,reduction", CASES,
                         ids=[f"{o}-{r}-{m}-{d}" for o, r, m, d in CASES])
def test_scan_matches_tile_reference(engine, orient, res, mode, reduction):
    kw = _kwargs(orient, res, mode, reduction)
    ref = ref_scan(engine, **kw)
    got = engine.scan(**kw)

    assert got.shape == ref.shape
    assert got.dtype == ref.dtype
    # The image must be non-trivial, or the comparison proves nothing.
    assert np.count_nonzero(ref) > 0.02 * ref.size

    diff = np.abs(got.astype(np.float64) - ref.astype(np.float64))
    scale = max(float(np.abs(ref).max()), 1e-30)
    # Bound: float32 eps (1.2e-7) times the number of hits accumulated per
    # ray, which for these overlapping blobs reaches a few dozen.  The
    # second assertion is the visually meaningful one — nothing differs at
    # 8-bit display precision (1/255 ~ 3.9e-3).
    assert diff.max() <= 1e-5 * scale, f"max deviation {diff.max():.3e}"
    assert int((diff > 1e-4).sum()) == 0


def test_empty_scene_returns_zeros():
    eng = ScannerEngine(TissueMapper())
    eng.cache_meshes([])
    got = eng.scan(**_kwargs("axial", 32, "ct", "mean"))
    assert got.shape == (32, 32)
    assert not got.any()
    rgb = eng.scan(**_kwargs("axial", 32, "anatomical", "mean"))
    assert rgb.shape == (32, 32, 3)
    assert not rgb.any()


def test_scan_misses_when_slab_is_off_target(engine):
    kw = _kwargs("axial", 48, "ct", "mean")
    kw["origin"] = np.array([500.0, 500.0, 500.0], np.float32)
    got = engine.scan(**kw)
    assert not got.any()


def test_progress_callback_is_monotonic_and_completes(engine):
    seen: list[float] = []
    engine.scan(**_kwargs("axial", 64, "ct", "mean"),
                progress_callback=seen.append)
    assert seen, "progress callback never fired"
    assert seen == sorted(seen)
    assert seen[-1] == pytest.approx(1.0)
    assert 0.0 <= seen[0] <= 1.0
