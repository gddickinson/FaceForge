"""What the scanner's numbers are, enforced.

The claim :mod:`faceforge.export.hounsfield` makes is narrow on purpose:
``index`` mode never pretends to be Hounsfield units, and ``class`` mode is
only reachable in the one configuration where the model's values invert
exactly to a tissue class.  These tests are what stop that narrowness from
being quietly widened later.
"""

from __future__ import annotations

import numpy as np
import pytest

from faceforge.export import hounsfield as hu
from faceforge.scanner.tissue_map import TissueMapper


# ---------------------------------------------------------------------------
# The mapping table itself
# ---------------------------------------------------------------------------


def test_tissue_class_list_matches_tissue_map():
    """A tissue added to the scanner must not be silently missing an HU.

    This is the only place that reads ``tissue_map``'s private table, and it
    reads it precisely so the *public* module never has to: if a class appears
    there and not in ``TISSUE_CLASSES``, this fails.
    """
    from faceforge.scanner.tissue_map import _CT_TABLE

    assert set(hu.TISSUE_CLASSES) == set(_CT_TABLE), (
        "faceforge.export.hounsfield.TISSUE_CLASSES has drifted from "
        "faceforge.scanner.tissue_map._CT_TABLE"
    )


def test_every_tissue_class_has_a_nominal_hu():
    assert set(hu.TISSUE_NOMINAL_HU) == set(hu.TISSUE_CLASSES)


def test_the_value_to_hu_inverse_map_is_well_defined():
    """Colliding tissue-table values must agree on their HU, or refuse."""
    report = hu.check_table_consistency()
    assert report["tissue_classes"] == 15
    # nerve and skin share 0.35.  Both are 30 HU, so the value still inverts.
    assert 0.35 in report["colliding_values"]
    assert report["colliding_values"][0.35]["tissues"] == ["nerve", "skin"]
    assert report["colliding_values"][0.35]["nominal_hu"] == [30]


def test_nominal_hu_ordering_is_physically_sensible():
    """Air < fat < water < soft tissue < bone -- the one ordering HU must have."""
    table = hu.TISSUE_NOMINAL_HU
    assert table["air"] == -1000
    assert table["fluid"] == 0, "fluid stands in for water, which defines 0 HU"
    assert table["air"] < table["fat"] < table["fluid"]
    assert table["fluid"] < table["muscle"] < table["cartilage"] < table["bone"]


def test_a_disagreeing_collision_is_refused(monkeypatch):
    """The guard has to actually fire, not just exist."""
    patched = dict(hu.TISSUE_NOMINAL_HU)
    patched["skin"] = 999          # nerve stays at 30; both are table value 0.35
    monkeypatch.setattr(hu, "TISSUE_NOMINAL_HU", patched)
    with pytest.raises(hu.HUMappingError, match="cannot be inverted"):
        hu.check_table_consistency()


def test_a_tissue_with_no_nominal_hu_is_refused(monkeypatch):
    patched = dict(hu.TISSUE_NOMINAL_HU)
    del patched["bone"]
    monkeypatch.setattr(hu, "TISSUE_NOMINAL_HU", patched)
    with pytest.raises(hu.HUMappingError, match="no nominal HU assigned"):
        hu.check_table_consistency()


# ---------------------------------------------------------------------------
# index mode: the default, and exactly invertible
# ---------------------------------------------------------------------------


def test_index_mode_does_not_claim_hounsfield_units():
    spec = hu.to_hounsfield(np.zeros((2, 2), dtype=np.float32))
    assert spec.rescale_type == "US", (
        "RescaleType must be US (unspecified), not HU: these are not CT numbers"
    )
    assert "NOT Hounsfield units" in spec.description
    assert "dimensionless" in spec.unit


def test_index_mode_rescale_returns_the_model_value():
    """A DICOM reader applying slope/intercept must get the scanner's number back."""
    values = np.array([[0.0, 0.10, 0.40], [0.45, 0.95, 1.0]], dtype=np.float32)
    spec = hu.to_hounsfield(values, hu_mode="index")

    assert spec.stored.dtype == np.uint16
    assert spec.slope == 1.0 / 1000
    assert spec.intercept == 0.0
    # The scale is 1000 and the table values have at most two decimals, so the
    # round-trip is exact, not approximate.
    assert np.array_equal(spec.stored,
                          np.rint(values.astype(np.float64) * 1000).astype(np.uint16))
    assert np.allclose(spec.real_values(), values, atol=1e-9)


def test_index_mode_accepts_any_reduction_and_any_mode():
    """index mode is true of every scan, so nothing about it is conditional."""
    values = np.array([[0.123, 0.777]], dtype=np.float32)
    for mode in ("ct", "mri_t1", "mri_t2"):
        for reduction in ("mean", "max", "min", "sum"):
            spec = hu.to_hounsfield(values, hu_mode="index", mode=mode,
                                    reduction=reduction)
            assert spec.rescale_type == "US"


# ---------------------------------------------------------------------------
# class mode: opt-in, and refused where it would not be defensible
# ---------------------------------------------------------------------------


def _ct_value(tissue: str) -> float:
    return float(TissueMapper.get_value(tissue, "ct"))


def test_class_mode_maps_table_values_to_their_nominal_hu():
    image = np.array([
        [_ct_value("bone"), _ct_value("fat")],
        [_ct_value("muscle"), 0.0],
    ], dtype=np.float32)
    spec = hu.to_hounsfield(image, hu_mode="class", mode="ct", reduction="max")

    assert spec.rescale_type == "HU"
    assert spec.intercept == -1024.0 and spec.slope == 1.0
    real = spec.real_values()
    assert real[0, 0] == hu.TISSUE_NOMINAL_HU["bone"]
    assert real[0, 1] == hu.TISSUE_NOMINAL_HU["fat"]
    assert real[1, 0] == hu.TISSUE_NOMINAL_HU["muscle"]
    assert real[1, 1] == hu.TISSUE_NOMINAL_HU["air"] == -1000, (
        "a pixel no ray crossed is air, and air is -1000 HU"
    )


def test_class_mode_reports_which_classes_it_found():
    image = np.full((4, 4), _ct_value("bone"), dtype=np.float32)
    image[0, :] = 0.0
    spec = hu.to_hounsfield(image, hu_mode="class", mode="ct", reduction="max")

    present = spec.class_report["classes_present"]
    assert present["bone"]["voxels"] == 12
    assert present["bone"]["nominal_hu"] == 700
    assert present["air"]["voxels"] == 4
    assert spec.class_report["worst_residual"] < hu.CLASS_TOLERANCE


def test_class_mode_states_its_limitations_in_the_description():
    spec = hu.to_hounsfield(
        np.full((2, 2), _ct_value("bone"), dtype=np.float32),
        hu_mode="class", mode="ct", reduction="max",
    )
    text = spec.description
    assert text.startswith("SIMULATED, NOT AN ACQUISITION")
    for phrase in ("nominal", "no partial-volume", "no noise",
                   "Do not use for dosimetry"):
        assert phrase in text, phrase
    assert any("Surface sampling, not volume sampling" in n for n in spec.notes)


def test_class_mode_refuses_a_mean_reduction():
    """With 'mean' a pixel can sit between two tissues; there is no class for that."""
    image = np.full((2, 2), 0.675, dtype=np.float32)     # midway bone/cartilage
    with pytest.raises(hu.HUMappingError, match="reduction='max'"):
        hu.to_hounsfield(image, hu_mode="class", mode="ct", reduction="mean")


def test_class_mode_refuses_values_that_are_not_table_entries():
    """Even labelled reduction='max', off-table values are refused, not rounded."""
    image = np.full((2, 2), 0.675, dtype=np.float32)
    with pytest.raises(hu.HUMappingError,
                       match="not equal to any CT tissue table entry"):
        hu.to_hounsfield(image, hu_mode="class", mode="ct", reduction="max")


@pytest.mark.parametrize("mode", ["mri_t1", "mri_t2"])
def test_class_mode_refuses_mri(mode):
    """HU is defined by X-ray attenuation; an MRI intensity has no HU."""
    image = np.full((2, 2), 0.7, dtype=np.float32)
    with pytest.raises(hu.HUMappingError, match="no HU equivalent"):
        hu.to_hounsfield(image, hu_mode="class", mode=mode, reduction="max")


def test_an_unknown_hu_mode_is_refused():
    with pytest.raises(hu.HUMappingError, match="unknown hu_mode"):
        hu.to_hounsfield(np.zeros((2, 2)), hu_mode="calibrated")


def test_class_mode_stored_values_fit_an_unsigned_16_bit_pixel():
    """Every nominal HU plus 1024 must be representable as written."""
    for tissue, value in hu.TISSUE_NOMINAL_HU.items():
        stored = value - hu.HU_INTERCEPT
        assert 0 <= stored <= 65535, tissue
