# tests/ -- Test Suite

Unit and integration tests for FaceForge, using pytest.

## Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test module
pytest tests/core/test_scene_graph.py

# Run a specific test
pytest tests/core/test_scene_graph.py::test_node_hierarchy
```

## Configuration

Test configuration is in `pyproject.toml`:
- `testpaths = ["tests"]`
- `pythonpath = ["src"]`

Dev dependencies: `pytest >= 7.4`, `pytest-qt >= 4.2` (for Qt widget tests).

## Test Structure

```
tests/
  core/                  # Core infrastructure tests
    test_scene_graph.py  # SceneNode hierarchy, matrix propagation, add/remove
    test_state.py        # FaceState, BodyState, StateManager
    test_events.py       # EventBus subscribe/publish/unsubscribe
    test_math_utils.py   # Vec3, Quat, Mat4 operations, quaternion math
  animation/             # Animation subsystem tests
    test_interpolation.py  # StateInterpolator lerp behavior
    test_auto_blink.py    # AutoBlink timing and blink cycle
  body/                  # Body system tests
    test_skinning_diagnostics.py  # SkinningDiagnostic analysis, cross-region binding prevention
  loaders/               # Loader tests
    test_stl_parser.py   # Binary STL parsing, indexed geometry dedup
  tools/                 # Tool tests
    test_headless_diagnostic.py  # Headless loader and diagnostic integration
  anatomy/               # (placeholder, no tests yet)
  integration/           # (placeholder, no tests yet)
  fixtures/              # Shared test fixtures directory
```

## Test Coverage

### `core/test_scene_graph.py`
Tests SceneNode parent-child relationships, matrix composition, world matrix propagation, visibility, mesh attachment, and Scene traversal.

### `core/test_state.py`
Tests FaceState AU getter/setter, BodyState DOF management, TargetAU/TargetHead, StateManager aggregation.

### `core/test_events.py`
Tests EventBus subscribe, publish with data, unsubscribe, and clear.

### `core/test_math_utils.py`
Tests vector operations, quaternion identity/composition/slerp, matrix inverse, look-at, perspective, euler-to-quaternion conversions.

### `animation/test_interpolation.py`
Tests StateInterpolator convergence behavior for AUs, head rotation, and body state.

### `animation/test_auto_blink.py`
Tests AutoBlink cycle timing, blink amount ramp up/down, random interval generation.

### `body/test_skinning_diagnostics.py`
Tests with realistic joint coordinates matching the actual BodyParts3D model. Verifies:
- Hand/foot chains cannot grab distant vertices (proportional Z margin)
- Arm chains do not bind to leg-region vertices (spatial limit guard)
- Cross-binding anomalies are detected by the diagnostic tool
- Arm raise does not displace leg/torso skin

### `loaders/test_stl_parser.py`
Tests binary STL parsing with synthetic triangle data, indexed geometry deduplication, and error handling for malformed files.

### `tools/test_headless_diagnostic.py`
Integration test for the headless loader: loads scene without Qt/GL, runs diagnostic analysis, verifies binding reports.

## Writing New Tests

- Place tests in the subdirectory matching the source package (e.g., `tests/body/` for `src/faceforge/body/`)
- Use NumPy for constructing test geometry (positions, normals, indices)
- Tests should not require OpenGL or Qt unless specifically testing those integrations
- Use `pytest-qt` for tests that need a QApplication context
