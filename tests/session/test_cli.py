"""``faceforge-cli`` argument handling, help text and failure modes.

Fast tier: nothing here acquires a GL context or reads the BodyParts3D
dataset.  Every subcommand's ``--help`` is exercised, because a CLI whose help
crashes is a CLI nobody can use, and argparse only builds a subparser's
choices when that subparser is reached.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from faceforge import cli


ALL_COMMANDS = ["render", "batch", "scan", "export", "verify-assets",
                "list-layers", "gui"]


# ---------------------------------------------------------------------------
# Import purity
# ---------------------------------------------------------------------------


def test_importing_the_cli_does_not_import_qt():
    """The CLI exists for machines that have no Qt; check it stays that way.

    ``faceforge.app`` imports PySide6 at module scope, so the ``gui``
    subcommand imports it inside its handler.  If that ever moves to the top of
    a module in the import path, this fails.
    """
    code = (
        "import sys; import faceforge.cli; "
        "faceforge.cli.build_parser(); "
        "print('PySide6' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, check=True,
        env={"PYTHONPATH": ":".join(sys.path), "PATH": "/usr/bin:/bin"},
    )
    assert result.stdout.strip() == "False", (
        "importing faceforge.cli (or building its parser) pulled in Qt"
    )


# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------


def test_bare_invocation_prints_help_and_signals_usage(capsys):
    assert cli.main([]) == cli.EXIT_USAGE
    out = capsys.readouterr().out
    for command in ALL_COMMANDS:
        assert command in out, f"{command} is missing from the top-level help"


@pytest.mark.parametrize("command", ALL_COMMANDS)
def test_every_subcommand_has_working_help(command, capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main([command, "--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert out.strip(), f"{command} --help printed nothing"
    assert f"faceforge-cli {command}" in out


def test_top_level_help_explains_reproducibility_and_the_gui_split(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = capsys.readouterr().out
    assert "byte-identical" in out
    assert "video export" in out, "the help should say what is NOT available"


def test_render_help_lists_every_render_mode(capsys):
    from faceforge.core.scene_state import render_mode_names

    with pytest.raises(SystemExit):
        cli.main(["render", "--help"])
    out = capsys.readouterr().out
    for mode in render_mode_names():
        assert mode in out, f"render --help does not mention {mode}"


def test_scan_help_offers_exactly_the_tissue_map_modes(capsys):
    from faceforge.scanner.tissue_map import MODES

    with pytest.raises(SystemExit):
        cli.main(["scan", "--help"])
    out = capsys.readouterr().out
    for mode in MODES:
        assert mode in out


# ---------------------------------------------------------------------------
# Value parsing
# ---------------------------------------------------------------------------


def test_parse_size_accepts_and_rejects():
    assert cli.parse_size("512x512") == (512, 512)
    assert cli.parse_size("1920X1080") == (1920, 1080)
    for bad in ("0x0", "8x64", "9000x100", "abc", "512", "512x", "-1x-1", "1x2x3"):
        with pytest.raises(ValueError):
            cli.parse_size(bad)


def test_as_vec3_validates():
    assert cli.as_vec3([1, -2.5, 3]) == (1.0, -2.5, 3.0)
    for bad in ([1, 2], [float("nan"), 0, 0], [float("inf"), 0, 0]):
        with pytest.raises(ValueError):
            cli.as_vec3(bad)


def test_parse_list_drops_empties():
    assert cli.parse_list("a, b ,,c") == ("a", "b", "c")
    assert cli.parse_list("") == ()
    assert cli.parse_list(None) == ()


def test_negative_scan_coordinates_parse():
    """Anatomical coordinates are routinely negative; argparse must take them."""
    args = cli.build_parser().parse_args(
        ["scan", "--out", "x.png", "--position", "-0.825", "-91.095", "1556.629"],
    )
    assert cli.as_vec3(args.position) == (-0.825, -91.095, 1556.629)


# ---------------------------------------------------------------------------
# Failure modes that must not need GL
# ---------------------------------------------------------------------------


def test_a_missing_state_file_fails_before_any_gl_work(tmp_path, capsys):
    code = cli.main(["render", "--state", str(tmp_path / "nope.state.json"),
                     "--out", str(tmp_path / "out.png")])
    assert code == cli.EXIT_FAILED
    assert not (tmp_path / "out.png").exists(), "a failed render wrote a file"
    assert "no such state file" in capsys.readouterr().err.lower()


def test_an_unknown_scan_mode_is_an_argument_error():
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["scan", "--out", "x.png", "--mode", "pet"])
    assert excinfo.value.code == cli.EXIT_USAGE


def test_batch_reports_a_missing_directory(tmp_path, capsys):
    assert cli.main(["batch", "--states", str(tmp_path / "gone"),
                     "--out", str(tmp_path / "o")]) == cli.EXIT_FAILED
    assert "not a directory" in capsys.readouterr().err


def test_batch_reports_an_empty_directory(tmp_path, capsys):
    (tmp_path / "states").mkdir()
    assert cli.main(["batch", "--states", str(tmp_path / "states"),
                     "--out", str(tmp_path / "o")]) == cli.EXIT_FAILED
    assert "no files match" in capsys.readouterr().err


def test_list_layers_prints_the_tiers_and_layers(capsys):
    assert cli.main(["list-layers"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    from faceforge.session import LAYER_LOADERS, TIER_LOADS

    for tier in TIER_LOADS:
        assert f"  {tier}  " in out
    for layer in LAYER_LOADERS:
        assert layer in out


def test_verify_assets_delegates_to_the_existing_tool(monkeypatch):
    """It must not reimplement asset verification; it must call the tool."""
    from faceforge import session as fs

    seen = {}

    class _Stub:
        @staticmethod
        def main(argv):
            seen["argv"] = argv
            return 0

    monkeypatch.setattr(fs, "repo_tool_module",
                        lambda name: _Stub if name == "fetch_assets" else None)
    assert cli.main(["verify-assets"]) == 0
    assert seen["argv"] == ["verify"]
    assert cli.main(["verify-assets", "--", "cache", "--dry-run"]) == 0
    assert seen["argv"] == ["cache", "--dry-run"]


# ---------------------------------------------------------------------------
# The config-fingerprint mechanism, surfaced on the command line
# ---------------------------------------------------------------------------


def _empty_state():
    """A structure-free SceneState, captured from empty collaborators.

    Built by capture rather than by hand: ``SceneState()``'s face/body blocks
    default to ``{}``, which the codec rightly refuses to serialise, and a
    hand-written dict would be a second definition of those blocks.
    """
    from faceforge.core.scene_graph import Scene
    from faceforge.core.scene_state import capture_scene_state
    from faceforge.rendering.camera import Camera
    from faceforge.rendering.lights import LightSetup

    return capture_scene_state(
        scene=Scene(), camera=Camera(), lights=LightSetup(), viewport=(64, 64),
    )


def _state_file(tmp_path, *, digest="deliberately-wrong"):
    """A state file whose config fingerprint does not match the configs on disk."""
    import dataclasses

    from faceforge.core.scene_state import codec

    state = _empty_state()
    state = dataclasses.replace(
        state, config=dataclasses.replace(state.config, digest=digest),
    )
    path = tmp_path / "drifted.state.json"
    codec.save(state, path)
    return path


def test_config_drift_warns_by_default(tmp_path, capsys):
    """Re-rendering an old state against changed configs is legitimate...

    ...doing it unknowingly is not.  The default is a loud warning, not a
    refusal.
    """
    import argparse

    path = _state_file(tmp_path)
    args = argparse.Namespace(state=path, no_config_check=False,
                              require_config_match=False)
    state, drift = cli._load_state(args)
    assert state is not None
    assert drift and "configs have changed" in drift
    assert "WARNING" in capsys.readouterr().err


def test_require_config_match_refuses_and_writes_nothing(tmp_path, capsys):
    path = _state_file(tmp_path)
    out = tmp_path / "figure.png"
    code = cli.main(["render", "--state", str(path), "--out", str(out),
                     "--require-config-match"])
    assert code == cli.EXIT_CONFIG_DRIFT
    assert not out.exists(), "a refused render still wrote an image"
    assert "REFUSING TO RENDER" in capsys.readouterr().err


def test_no_config_check_silences_the_comparison(tmp_path, capsys):
    import argparse

    path = _state_file(tmp_path)
    args = argparse.Namespace(state=path, no_config_check=True,
                              require_config_match=True)
    state, drift = cli._load_state(args)
    assert state is not None and drift is None
    assert capsys.readouterr().err == ""


def test_a_matching_fingerprint_does_not_warn(tmp_path, capsys):
    """The drift check must be capable of passing, or the test above is vacuous."""
    import argparse

    from faceforge.core.scene_state import codec

    path = tmp_path / "current.state.json"
    codec.save(_empty_state(), path)
    args = argparse.Namespace(state=path, no_config_check=False,
                              require_config_match=True)
    _state, drift = cli._load_state(args)
    assert drift is None, f"an unmodified fingerprint reported drift: {drift}"
    assert capsys.readouterr().err == ""


# ---------------------------------------------------------------------------
# Console-script wiring
# ---------------------------------------------------------------------------


def test_pyproject_wires_both_entry_points():
    """The GUI entry point must survive the addition of the CLI one."""
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    data = tomllib.loads((root / "pyproject.toml").read_text())
    scripts = data["project"]["scripts"]
    assert scripts["faceforge"] == "faceforge.app:main", (
        "the GUI entry point changed; `faceforge` must keep launching the GUI"
    )
    assert scripts["faceforge-cli"] == "faceforge.cli:main"


def test_the_module_is_runnable_with_dash_m(tmp_path):
    """`python -m faceforge.cli` is the no-install path; keep it working."""
    result = subprocess.run(
        [sys.executable, "-m", "faceforge.cli", "list-layers"],
        capture_output=True, text=True, check=False,
        env={"PYTHONPATH": ":".join(sys.path), "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, result.stderr[-500:]
    assert "tiers" in result.stdout
