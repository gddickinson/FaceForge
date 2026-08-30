"""Tests for ``tools.fetch_assets`` — the BodyParts3D asset checker.

Every dataset state is built in a ``tmp_path``; the real 1.2 GB dataset is
never written to and only read by the single asset-gated test at the bottom,
which exists because the state it checks (a symlink that resolves) is the one
whose failure silently disabled 33 tests before the 2026-08 review.
"""

from __future__ import annotations

import json
import struct

import pytest

from tools.fetch_assets import (
    STATE_CORRUPT,
    STATE_DANGLING_SYMLINK,
    STATE_EMPTY,
    STATE_MISSING_DIR,
    STATE_OK,
    STATE_PARTIAL,
    _assert_safe_cache_dir,
    _collect_mesh_ids,
    _stl_defect,
    build_cache,
    main,
    render,
    verify,
)


# ── helpers ─────────────────────────────────────────────────────────

def _stl_bytes(n_tri: int = 1) -> bytes:
    """A valid binary STL with *n_tri* degenerate-but-well-formed triangles."""
    out = b"\x00" * 80 + struct.pack("<I", n_tri)
    for _ in range(n_tri):
        out += struct.pack("<3f", 0.0, 0.0, 1.0)
        out += struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0)
        out += struct.pack("<H", 0)
    return out


def _make_configs(tmp_path, mapping: dict[str, list[str]]):
    """Write a config tree: {config filename: [mesh ids]} -> config dir."""
    config = tmp_path / "assets" / "config"
    (config / "muscles").mkdir(parents=True)
    for name, ids in mapping.items():
        target = config / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps([{"name": i, "stl": i} for i in ids]), encoding="utf-8"
        )
    return config


def _make_assets(tmp_path, ids, *, name="stl"):
    stl = tmp_path / name
    stl.mkdir(parents=True, exist_ok=True)
    for i in ids:
        (stl / f"{i}.stl").write_bytes(_stl_bytes())
    return stl


# ── manifest extraction ─────────────────────────────────────────────

def test_manifest_collects_stl_keys_at_any_depth(tmp_path):
    config = tmp_path / "assets" / "config"
    config.mkdir(parents=True)
    (config / "flat.json").write_text(json.dumps([{"stl": "FMA1"}]))
    (config / "nested.json").write_text(json.dumps(
        {"groups": [{"members": [{"stl": "FMA2"}, {"stl": "FMA3"}]}]}
    ))
    required, errors = _collect_mesh_ids(config)
    assert set(required) == {"FMA1", "FMA2", "FMA3"}
    assert errors == []


def test_manifest_ignores_stl_base_directory_prefix(tmp_path):
    """``stl_base`` holds a directory, not a mesh id — the one such key."""
    config = tmp_path / "assets" / "config"
    config.mkdir(parents=True)
    (config / "transform.json").write_text(
        json.dumps({"stl_base": "bodyparts3D/stl/", "stl": "FMA1"})
    )
    required, _ = _collect_mesh_ids(config)
    assert set(required) == {"FMA1"}


def test_manifest_records_which_config_requires_each_mesh(tmp_path):
    config = _make_configs(tmp_path, {
        "muscles/arm.json": ["FMA1", "FMA2"],
        "skull.json": ["FMA2"],
    })
    required, _ = _collect_mesh_ids(config)
    assert len(required["FMA1"]) == 1
    assert len(required["FMA2"]) == 2, "a mesh named by two configs lists both"


def test_manifest_reports_unreadable_config(tmp_path):
    config = tmp_path / "assets" / "config"
    config.mkdir(parents=True)
    (config / "broken.json").write_text("{not json", encoding="utf-8")
    required, errors = _collect_mesh_ids(config)
    assert required == {}
    assert len(errors) == 1 and "broken.json" in errors[0]


def test_manifest_reports_absent_config_dir(tmp_path):
    _, errors = _collect_mesh_ids(tmp_path / "nope")
    assert errors and "does not exist" in errors[0]


# ── per-file integrity ──────────────────────────────────────────────

def test_valid_stl_has_no_defect(tmp_path):
    path = tmp_path / "a.stl"
    path.write_bytes(_stl_bytes(3))
    assert _stl_defect(path) is None


def test_zero_byte_stl_is_a_defect(tmp_path):
    path = tmp_path / "a.stl"
    path.write_bytes(b"")
    assert _stl_defect(path) == "zero bytes"


def test_truncated_stl_is_a_defect(tmp_path):
    """The failure mode an interrupted 1.2 GB copy actually produces."""
    path = tmp_path / "a.stl"
    path.write_bytes(_stl_bytes(50)[:200])
    defect = _stl_defect(path)
    assert defect is not None and "declares 50 triangles" in defect


def test_header_shorter_than_84_bytes_is_a_defect(tmp_path):
    path = tmp_path / "a.stl"
    path.write_bytes(b"\x00" * 40)
    defect = _stl_defect(path)
    assert defect is not None and "shorter than an STL header" in defect


# ── verify: the dataset states ──────────────────────────────────────

def test_verify_complete_dataset(tmp_path):
    config = _make_configs(tmp_path, {"muscles/arm.json": ["FMA1", "FMA2"]})
    stl = _make_assets(tmp_path, ["FMA1", "FMA2"])
    report = verify(stl, config)
    assert report.state == STATE_OK and report.ok
    assert (report.required, report.present, report.missing) == (2, 2, [])
    assert report.bytes_on_disk > 0


def test_verify_missing_directory(tmp_path):
    config = _make_configs(tmp_path, {"muscles/arm.json": ["FMA1"]})
    report = verify(tmp_path / "absent", config)
    assert report.state == STATE_MISSING_DIR and not report.ok
    assert report.missing == ["FMA1"], "everything required counts as missing"
    assert report.symlink_target is None


def test_verify_dangling_symlink_is_its_own_state(tmp_path):
    """The pre-review failure: the link exists, the target does not."""
    config = _make_configs(tmp_path, {"muscles/arm.json": ["FMA1"]})
    link = tmp_path / "stl"
    link.symlink_to(tmp_path / "nowhere")
    report = verify(link, config)
    assert report.state == STATE_DANGLING_SYMLINK
    assert report.symlink_target == str(tmp_path / "nowhere")
    assert report.resolved is None
    assert "symlink whose target does not exist" in render(report)


def test_verify_symlink_that_resolves_is_followed(tmp_path):
    config = _make_configs(tmp_path, {"muscles/arm.json": ["FMA1"]})
    real = _make_assets(tmp_path, ["FMA1"], name="real")
    link = tmp_path / "stl"
    link.symlink_to(real)
    report = verify(link, config)
    assert report.state == STATE_OK
    assert report.symlink_target == str(real)


def test_verify_empty_directory(tmp_path):
    config = _make_configs(tmp_path, {"muscles/arm.json": ["FMA1"]})
    stl = tmp_path / "stl"
    stl.mkdir()
    report = verify(stl, config)
    assert report.state == STATE_EMPTY and report.files_on_disk == 0


def test_verify_partial_names_the_missing_meshes_and_their_configs(tmp_path):
    config = _make_configs(tmp_path, {
        "muscles/arm.json": ["FMA1", "FMA2"],
        "skull.json": ["FMA3"],
    })
    stl = _make_assets(tmp_path, ["FMA1"])
    report = verify(stl, config)
    assert report.state == STATE_PARTIAL
    assert report.missing == ["FMA2", "FMA3"]
    assert report.present == 1
    text = render(report)
    assert "FMA2.stl" in text and "muscles/arm.json" in text
    assert "FMA3.stl" in text and "skull.json" in text


def test_verify_flags_present_but_corrupt_files(tmp_path):
    config = _make_configs(tmp_path, {"muscles/arm.json": ["FMA1", "FMA2"]})
    stl = _make_assets(tmp_path, ["FMA1"])
    (stl / "FMA2.stl").write_bytes(b"")
    report = verify(stl, config)
    assert report.state == STATE_CORRUPT, "complete but unreadable is not OK"
    assert report.missing == []
    assert report.defective == [("FMA2.stl", "zero bytes")]


def test_verify_integrity_check_is_optional(tmp_path):
    config = _make_configs(tmp_path, {"muscles/arm.json": ["FMA1"]})
    stl = tmp_path / "stl"
    stl.mkdir()
    (stl / "FMA1.stl").write_bytes(b"")
    assert verify(stl, config, integrity=True).state == STATE_CORRUPT
    assert verify(stl, config, integrity=False).state == STATE_OK


def test_verify_reports_unreferenced_files_without_failing(tmp_path):
    config = _make_configs(tmp_path, {"muscles/arm.json": ["FMA1"]})
    stl = _make_assets(tmp_path, ["FMA1", "FMA_extra"])
    report = verify(stl, config)
    assert report.state == STATE_OK, "spare files on disk are not an error"
    assert report.unreferenced == ["FMA_extra"]


def test_report_json_is_serialisable(tmp_path):
    config = _make_configs(tmp_path, {"muscles/arm.json": ["FMA1", "FMA2"]})
    stl = _make_assets(tmp_path, ["FMA1"])
    payload = json.loads(json.dumps(verify(stl, config).as_dict()))
    assert payload["state"] == STATE_PARTIAL and payload["missing"] == ["FMA2"]


# ── cache building ──────────────────────────────────────────────────

def test_cache_dry_run_writes_nothing(tmp_path):
    stl = _make_assets(tmp_path, ["FMA1", "FMA2"])
    cache = tmp_path / "cache"
    stats = build_cache(stl, cache, dry_run=True, log=lambda *a: None)
    assert stats["meshes"] == 2 and stats["built"] == 0
    assert not cache.exists(), "dry-run must not even create the directory"


def test_cache_builds_npz_entries_and_leaves_assets_untouched(tmp_path):
    stl = _make_assets(tmp_path, ["FMA1", "FMA2"])
    before = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in stl.iterdir()}
    cache = tmp_path / "cache"

    stats = build_cache(stl, cache, log=lambda *a: None)

    assert stats["built"] == 2 and stats["failed"] == []
    assert len(list(cache.glob("*.npz"))) == 2
    assert stats["cache_bytes"] > 0
    after = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in stl.iterdir()}
    assert after == before, "the asset directory must be read-only to this tool"


def test_cache_second_call_is_idempotent(tmp_path):
    stl = _make_assets(tmp_path, ["FMA1"])
    cache = tmp_path / "cache"
    build_cache(stl, cache, log=lambda *a: None)
    entries = {p.name for p in cache.glob("*.npz")}
    stats = build_cache(stl, cache, log=lambda *a: None)
    assert stats["built"] == 1 and stats["failed"] == []
    assert {p.name for p in cache.glob("*.npz")} == entries


def test_cache_reports_a_bad_mesh_without_aborting_the_rest(tmp_path):
    stl = _make_assets(tmp_path, ["FMA1", "FMA3"])
    (stl / "FMA2.stl").write_bytes(b"garbage")
    stats = build_cache(stl, tmp_path / "cache", log=lambda *a: None)
    assert stats["built"] == 2, "one malformed file must not stop a 934-file build"
    assert [f["file"] for f in stats["failed"]] == ["FMA2.stl"]


def test_cache_can_be_restricted_to_required_meshes(tmp_path):
    stl = _make_assets(tmp_path, ["FMA1", "FMA_extra"])
    stats = build_cache(stl, tmp_path / "cache", names=["FMA1"], log=lambda *a: None)
    assert stats["meshes"] == 1 and stats["built"] == 1


# ── safety ──────────────────────────────────────────────────────────

def test_cache_dir_inside_repo_assets_is_refused(tmp_path):
    repo = tmp_path / "repo"
    (repo / "assets").mkdir(parents=True)
    with pytest.raises(ValueError, match="inside the asset tree"):
        _assert_safe_cache_dir(repo / "assets" / "cache", repo / "assets" / "stl", repo)


def test_cache_dir_inside_the_real_asset_dir_is_refused_through_a_symlink(tmp_path):
    """A cache dir reached via the assets/stl symlink still lands in the data."""
    repo = tmp_path / "repo"
    (repo / "assets").mkdir(parents=True)
    real = tmp_path / "outside" / "stl"
    real.mkdir(parents=True)
    link = repo / "assets" / "stl"
    link.symlink_to(real)
    with pytest.raises(ValueError, match="inside the asset tree"):
        _assert_safe_cache_dir(real / "cache", link, repo)


def test_cache_dir_outside_the_asset_tree_is_allowed(tmp_path):
    repo = tmp_path / "repo"
    (repo / "assets").mkdir(parents=True)
    _assert_safe_cache_dir(tmp_path / "elsewhere", repo / "assets" / "stl", repo)


# ── CLI exit codes ──────────────────────────────────────────────────

def test_cli_exit_zero_on_complete_dataset(tmp_path, capsys):
    config = _make_configs(tmp_path, {"muscles/arm.json": ["FMA1"]})
    stl = _make_assets(tmp_path, ["FMA1"])
    code = main(["verify", "--stl-dir", str(stl), "--config-dir", str(config)])
    assert code == 0
    assert "COMPLETE" in capsys.readouterr().out


def test_cli_exit_one_on_missing_assets(tmp_path, capsys):
    config = _make_configs(tmp_path, {"muscles/arm.json": ["FMA1"]})
    code = main([
        "verify", "--stl-dir", str(tmp_path / "absent"), "--config-dir", str(config),
    ])
    assert code == 1
    out = capsys.readouterr().out
    assert "MISSING-DIRECTORY" in out
    assert "Database Center for Life Science" in out, "must say how to get the data"


def test_cli_allow_missing_tolerates_a_known_gap(tmp_path):
    config = _make_configs(tmp_path, {"muscles/arm.json": ["FMA1", "FMA2"]})
    stl = _make_assets(tmp_path, ["FMA1"])
    args = ["verify", "--stl-dir", str(stl), "--config-dir", str(config)]
    assert main(args) == 1
    assert main([*args, "--allow-missing", "1"]) == 0
    assert main([*args, "--allow-missing", "0"]) == 1


def test_cli_allow_missing_does_not_excuse_corrupt_files(tmp_path):
    config = _make_configs(tmp_path, {"muscles/arm.json": ["FMA1"]})
    stl = tmp_path / "stl"
    stl.mkdir()
    (stl / "FMA1.stl").write_bytes(b"")
    code = main([
        "verify", "--stl-dir", str(stl), "--config-dir", str(config),
        "--allow-missing", "99",
    ])
    assert code == 1, "a threshold on absence must not tolerate corruption"


def test_cli_json_output_parses(tmp_path, capsys):
    config = _make_configs(tmp_path, {"muscles/arm.json": ["FMA1", "FMA2"]})
    stl = _make_assets(tmp_path, ["FMA1"])
    main(["verify", "--stl-dir", str(stl), "--config-dir", str(config), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["missing"] == ["FMA2"] and payload["ok"] is False


def test_cli_fetch_is_honestly_unimplemented(capsys):
    code = main(["fetch"])
    out = capsys.readouterr().out
    assert code == 3, "a distinct code, so a caller can tell it apart from failure"
    assert "not implemented" in out
    assert "http" not in out.lower(), "no guessed download URL may be printed"


def test_cli_manifest_lists_required_meshes(tmp_path, capsys):
    config = _make_configs(tmp_path, {"muscles/arm.json": ["FMA2", "FMA1"]})
    code = main(["manifest", "--config-dir", str(config)])
    assert code == 0
    assert capsys.readouterr().out.split() == ["FMA1", "FMA2"]


def test_cli_cache_refuses_an_unsafe_cache_dir(tmp_path, capsys):
    config = _make_configs(tmp_path, {"muscles/arm.json": ["FMA1"]})
    stl = _make_assets(tmp_path, ["FMA1"])
    code = main([
        "cache", "--stl-dir", str(stl), "--config-dir", str(config),
        "--cache-dir", str(stl / "inside"),
    ])
    assert code == 2
    assert "refusing" in capsys.readouterr().err


def test_cli_cache_with_no_assets_present_does_not_pretend_to_work(tmp_path, capsys):
    config = _make_configs(tmp_path, {"muscles/arm.json": ["FMA1"]})
    code = main([
        "cache", "--stl-dir", str(tmp_path / "absent"), "--config-dir", str(config),
        "--cache-dir", str(tmp_path / "cache"),
    ])
    assert code == 1
    assert "nothing to cache" in capsys.readouterr().err


def test_cli_cache_dry_run_reports_and_writes_nothing(tmp_path, capsys):
    config = _make_configs(tmp_path, {"muscles/arm.json": ["FMA1"]})
    stl = _make_assets(tmp_path, ["FMA1"])
    cache = tmp_path / "cache"
    code = main([
        "cache", "--stl-dir", str(stl), "--config-dir", str(config),
        "--cache-dir", str(cache), "--dry-run",
    ])
    assert code == 0
    assert "[dry-run]" in capsys.readouterr().out
    assert not cache.exists()


# ── the real repository ─────────────────────────────────────────────

@pytest.mark.slow
def test_real_repo_asset_path_is_not_a_dangling_symlink():
    """Regression guard for the state that silently disabled 33 tests.

    Marked `slow` because it reads the real 1.2 GB dataset's directory listing
    and 930 file headers; it is also the one test here that depends on a local
    checkout, so it is skipped rather than failed when the data is absent.
    """
    from faceforge.constants import CONFIG_DIR, STL_DIR

    report = verify(STL_DIR, CONFIG_DIR)
    if report.state in (STATE_MISSING_DIR, STATE_EMPTY):
        pytest.skip(f"BodyParts3D dataset not installed here ({report.state})")
    assert report.state != STATE_DANGLING_SYMLINK, (
        "assets/stl points at a path that does not exist; the asset-gated test "
        "modules will silently skip. Run `python -m tools.fetch_assets verify`."
    )
    assert report.defective == [], f"corrupt STL files present: {report.defective}"
    assert report.present >= report.required - 2, (
        f"only {report.present}/{report.required} required meshes present"
    )
