"""Tier 3-5 on-demand loaders must not swallow load failures silently.

``OrganManager``, ``BrainManager``, ``VasculatureManager``, ``BodyMuscleManager``
and ``SkeletonBuilder.load_all`` each used to wrap their asset call in a bare
``except Exception: pass`` and then set ``self.loaded = True`` regardless of
outcome.  A missing or unreadable STL therefore produced:

  * no log record of any kind,
  * an empty layer with ``loaded is True``, so it could never be retried, and
  * a UI toggle that appeared to work but showed nothing.

This is the mechanism that let a broken ``assets/stl`` symlink go unnoticed:
the app started cleanly and 29 tests skipped instead of failing.

FIXED (defect ``silent-load-failure``): each handler now calls
``logger.exception()``, catches only ``(OSError, ValueError, KeyError)`` so that
programming errors propagate, and records the failure on a separate flag
(``load_failed`` / ``failed_regions`` / ``failed_batches``) while leaving
``loaded`` False so the layer can be retried.

All five managers are exercised below, via a per-manager adapter -- the earlier
version of this module named five managers in its docstring but only
parametrised three.
"""

import logging
import pathlib

import pytest

from faceforge.body.body_muscles import MUSCLE_CONFIGS, BodyMuscleManager
from faceforge.body.brain import BrainManager
from faceforge.body.organs import OrganManager
from faceforge.body.skeleton import SkeletonBuilder
from faceforge.body.vasculature import VasculatureManager
from faceforge.coordination import loading_pipeline
from faceforge.coordination.loading_pipeline import ASSET_ERRORS, LoadReport
from faceforge.core.scene_graph import Scene, SceneNode

MISSING_STL = "assets/stl/FMA7202.stl missing"


class _ExplodingAssets:
    """Stands in for AssetManager; every loader raises as a missing STL would."""

    def __init__(self, exc: Exception | None = None):
        self.exc = exc or FileNotFoundError(MISSING_STL)
        self.calls: list[str] = []
        # SkeletonBuilder reads these off the asset manager directly.
        self.transform = None
        self.stl_dir = "/nonexistent/stl"

    def _boom(self, name):
        self.calls.append(name)
        raise self.exc

    def load_organs(self):
        return self._boom("load_organs")

    def load_brain(self):
        return self._boom("load_brain")

    def load_vasculature(self):
        return self._boom("load_vasculature")

    def load_body_muscles(self, config_name):
        return self._boom("load_body_muscles")

    def load_skeleton_batch(self, config_name, label=None):
        return self._boom("load_skeleton_batch")


# ----------------------------------------------------------------------
# Per-manager adapters. The five managers have genuinely different shapes
# (list vs dict of groups, bool vs dict `loaded`), so the shared behavioural
# assertions are expressed through these small accessors rather than by
# pretending the interfaces are identical.
# ----------------------------------------------------------------------

def _new_parent():
    scene = Scene()
    parent = SceneNode(name="bodyRoot")
    scene.add(parent)
    return parent


class _SimpleCase:
    """OrganManager / VasculatureManager / BrainManager."""

    def __init__(self, cls, loader_name):
        self.cls = cls
        self.loader_name = loader_name

    def build(self, assets):
        return self.cls(assets)

    def load(self, mgr):
        mgr.load(_new_parent())

    def expected_calls(self):
        return [self.loader_name]

    def is_empty(self, mgr):
        return mgr.group is None and mgr.meshes == []

    def claims_loaded(self, mgr):
        return mgr.loaded

    def failed_flag(self, mgr):
        return mgr.load_failed


class _MuscleCase:
    cls = BodyMuscleManager
    loader_name = "load_body_muscles"

    def build(self, assets):
        return BodyMuscleManager(assets)

    def load(self, mgr):
        mgr.load_all(_new_parent())

    def expected_calls(self):
        # One attempt per region config; all six fail.
        return [self.loader_name] * len(MUSCLE_CONFIGS)

    def is_empty(self, mgr):
        return mgr.groups == {} and mgr.meshes == {}

    def claims_loaded(self, mgr):
        return mgr.loaded

    def failed_flag(self, mgr):
        return mgr.load_failed


class _SkeletonCase:
    """SkeletonBuilder.load_all.

    Its eight batch loaders reach past the asset manager (``load_stl_batch``
    with ``assets.stl_dir``), so the failure is injected at the loader methods
    themselves. That is exactly the surface ``load_all``'s handler guards.
    """

    cls = SkeletonBuilder
    loader_name = "skeleton_batch"
    BATCH_METHODS = (
        "load_thoracic_spine", "load_lumbar_spine", "load_rib_cage",
        "load_pelvis", "load_upper_limbs", "load_hands",
        "load_lower_limbs", "load_feet",
    )

    def build(self, assets):
        mgr = SkeletonBuilder(assets)

        def make(name):
            def _boom():
                assets.calls.append(self.loader_name)
                raise assets.exc
            _boom.__name__ = name
            return _boom

        for name in self.BATCH_METHODS:
            setattr(mgr, name, make(name))
        return mgr

    def load(self, mgr):
        mgr.load_all(_new_parent())

    def expected_calls(self):
        return [self.loader_name] * len(self.BATCH_METHODS)

    def is_empty(self, mgr):
        return mgr.groups == {}

    def claims_loaded(self, mgr):
        return any(mgr.loaded.values())

    def failed_flag(self, mgr):
        return mgr.load_failed


CASES = [
    pytest.param(_SimpleCase(OrganManager, "load_organs"), id="organs"),
    pytest.param(_SimpleCase(VasculatureManager, "load_vasculature"), id="vasculature"),
    pytest.param(_SimpleCase(BrainManager, "load_brain"), id="brain"),
    pytest.param(_MuscleCase(), id="body_muscles"),
    pytest.param(_SkeletonCase(), id="skeleton"),
]


def _load(case, assets):
    mgr = case.build(assets)
    case.load(mgr)
    return mgr


# ----------------------------------------------------------------------
# A missing asset is still not fatal...
# ----------------------------------------------------------------------

@pytest.mark.parametrize("case", CASES)
def test_failed_load_is_swallowed_without_raising(case):
    assets = _ExplodingAssets()
    mgr = _load(case, assets)

    assert assets.calls == case.expected_calls(), "asset loader was not even called"
    assert case.is_empty(mgr)


# ----------------------------------------------------------------------
# ...but it must be visible, recoverable, and never mistaken for a bug.
# ----------------------------------------------------------------------

@pytest.mark.parametrize("case", CASES)
def test_failed_load_is_logged(case, caplog):
    assets = _ExplodingAssets()
    with caplog.at_level(logging.DEBUG):
        _load(case, assets)

    assert caplog.records, (
        f"{case.cls.__name__} logged nothing when the asset load raised"
    )
    # logger.exception(), not logger.info() -- the traceback must be there.
    assert any(r.levelno >= logging.ERROR for r in caplog.records), (
        f"{case.cls.__name__} logged only "
        f"{sorted({r.levelname for r in caplog.records})}"
    )
    assert any(r.exc_info for r in caplog.records), (
        f"{case.cls.__name__} logged no traceback"
    )
    assert any(MISSING_STL in r.getMessage() or
               (r.exc_info and MISSING_STL in str(r.exc_info[1]))
               for r in caplog.records), "the failing path is not in the log"


@pytest.mark.parametrize("case", CASES)
def test_failed_load_does_not_claim_success(case):
    mgr = _load(case, _ExplodingAssets())
    assert not case.claims_loaded(mgr), (
        "manager reports loaded=True after the asset load raised"
    )
    assert case.failed_flag(mgr) is True, "load_failed was not set"


@pytest.mark.parametrize("case", CASES)
def test_failed_load_can_be_retried(case):
    """The whole point of not setting ``loaded``: a retry re-attempts."""
    assets = _ExplodingAssets()
    mgr = _load(case, assets)

    assets.calls.clear()
    case.load(mgr)
    assert assets.calls == case.expected_calls(), (
        "retry short-circuited; the layer is stuck empty for the session"
    )


@pytest.mark.parametrize("case", CASES)
def test_programming_errors_are_not_swallowed(case):
    """An AttributeError is a bug, not a missing asset -- it must propagate."""
    assets = _ExplodingAssets(AttributeError("typo in loader implementation"))
    with pytest.raises(AttributeError):
        _load(case, assets)


# ----------------------------------------------------------------------
# LoadReport: a caller must be able to tell a complete scene from a
# degraded one without scraping the log.
# ----------------------------------------------------------------------

@pytest.mark.parametrize("case", CASES)
def test_load_report_absorbs_every_manager_shape(case):
    """One aggregation API over the three different failure-state shapes."""
    mgr = _load(case, _ExplodingAssets())
    report = LoadReport()
    assert report.ok, "a fresh report must start clean"

    report.absorb("layer", mgr)
    assert report.degraded, f"LoadReport did not see {case.cls.__name__}'s failure"
    assert report.failed_names == ["layer"]
    assert "FileNotFoundError" in report.summary()
    assert "DEGRADED" in report.summary()


def test_load_report_absorbs_a_healthy_manager_as_ok():
    class _Healthy:
        loaded = True
        load_failed = False
        load_error = None

    report = LoadReport()
    report.absorb("layer", _Healthy())
    assert report.ok
    assert report.summary() == "Scene loaded completely."


def test_load_report_flags_a_missing_manager():
    report = LoadReport()
    report.absorb("organs", None)
    assert report.degraded
    assert report.failures == {"organs": "not constructed"}


def test_load_report_records_phase_exceptions():
    report = LoadReport()
    report.record("face_features", FileNotFoundError("face_features.json missing"))
    report.record("vertebrae", KeyError("C7"))
    assert report.failed_names == ["face_features", "vertebrae"]
    assert "face_features.json missing" in report.summary()


def test_load_report_reports_partial_group_failures_separately():
    class _Partial:
        failed_regions = {"leg": "OSError: leg_muscles.json unreadable"}

    report = LoadReport()
    report.absorb("body_muscles", _Partial())
    assert report.degraded
    assert report.partial == {"body_muscles": _Partial.failed_regions}
    assert "1 of its groups failed (leg)" in report.summary()


def test_asset_errors_excludes_programming_errors():
    """The narrowing that makes an API removal loud instead of silent.

    A NumPy 2.0 removal (``ndarray.ptp``) raises AttributeError. Under the old
    ``except Exception`` it became one WARNING line and an empty subsystem.
    """
    assert ASSET_ERRORS == (OSError, ValueError, KeyError)
    for exc in (AttributeError, TypeError, ImportError, NameError):
        assert not issubclass(exc, ASSET_ERRORS), (
            f"{exc.__name__} would still be swallowed by a phase handler"
        )
    # And the errors that DO mean "missing asset" are still handled.
    for exc in (FileNotFoundError, PermissionError, IsADirectoryError):
        assert issubclass(exc, ASSET_ERRORS)


def test_no_blind_except_remains_in_the_loading_pipeline():
    """Guard: a re-added `except Exception` reintroduces the silent-failure class."""
    source = pathlib.Path(loading_pipeline.__file__).read_text()
    code = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )
    assert "except Exception" not in code
    assert code.count("except ASSET_ERRORS") == 9


@pytest.mark.parametrize("case", CASES)
def test_recorded_error_names_the_exception(case):
    mgr = _load(case, _ExplodingAssets())
    recorded = getattr(mgr, "load_error", None)
    if recorded is None:
        # Per-region / per-batch managers record a dict instead.
        recorded = " ".join(
            getattr(mgr, "failed_regions", {}) or getattr(mgr, "failed_batches", {})
        )
        detail = " ".join(
            (getattr(mgr, "failed_regions", None) or mgr.failed_batches).values()
        )
        assert "FileNotFoundError" in detail
    else:
        assert "FileNotFoundError" in recorded
    assert recorded
