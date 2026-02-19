"""JSON persistence for vertex chain reassignment overrides."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from faceforge.body.soft_tissue import SoftTissueSkinning

logger = logging.getLogger(__name__)

# Default override file location
_ASSETS_DIR = Path(__file__).parent.parent.parent.parent / "assets" / "config"
DEFAULT_OVERRIDES_PATH = _ASSETS_DIR / "skinning_overrides.json"


def save_overrides(
    skinning: SoftTissueSkinning,
    overrides: dict[str, dict[int, dict]],
    path: Path | str | None = None,
) -> Path:
    """Save vertex override data to JSON.

    Parameters
    ----------
    skinning : SoftTissueSkinning
        The skinning system (used for chain ID lookup).
    overrides : dict
        ``{mesh_name: {vertex_idx: {chain_id, joint, secondary, weight}}}``
    path : Path or str, optional
        Output file. Defaults to ``assets/config/skinning_overrides.json``.

    Returns
    -------
    Path
        The file that was written.
    """
    if path is None:
        path = DEFAULT_OVERRIDES_PATH
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types for JSON serialization
    data: dict = {"version": 1, "mesh_overrides": {}}
    for mesh_name, verts in overrides.items():
        mesh_data = {}
        for vi, info in verts.items():
            mesh_data[str(vi)] = {
                "chain_id": int(info["chain_id"]),
                "joint": int(info["joint"]),
                "secondary": int(info["secondary"]),
                "weight": float(info["weight"]),
            }
        data["mesh_overrides"][mesh_name] = mesh_data

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info("Saved %d mesh overrides to %s",
                sum(len(v) for v in overrides.values()), path)
    return path


def load_overrides(path: Path | str | None = None) -> dict[str, dict[int, dict]]:
    """Load vertex overrides from JSON.

    Returns
    -------
    dict
        ``{mesh_name: {vertex_idx: {chain_id, joint, secondary, weight}}}``
        Empty dict if file doesn't exist.
    """
    if path is None:
        path = DEFAULT_OVERRIDES_PATH
    path = Path(path)

    if not path.exists():
        return {}

    with open(path) as f:
        data = json.load(f)

    if data.get("version") != 1:
        logger.warning("Unknown overrides version: %s", data.get("version"))
        return {}

    result: dict[str, dict[int, dict]] = {}
    for mesh_name, verts in data.get("mesh_overrides", {}).items():
        mesh_data: dict[int, dict] = {}
        for vi_str, info in verts.items():
            mesh_data[int(vi_str)] = {
                "chain_id": info["chain_id"],
                "joint": info["joint"],
                "secondary": info["secondary"],
                "weight": info["weight"],
            }
        result[mesh_name] = mesh_data

    logger.info("Loaded %d vertex overrides from %s",
                sum(len(v) for v in result.values()), path)
    return result


def apply_overrides(
    skinning: SoftTissueSkinning,
    overrides: dict[str, dict[int, dict]],
) -> int:
    """Apply loaded overrides to skinning bindings.

    Parameters
    ----------
    skinning : SoftTissueSkinning
        The skinning system with registered bindings.
    overrides : dict
        As returned by :func:`load_overrides`.

    Returns
    -------
    int
        Number of vertices overridden.
    """
    count = 0
    for binding in skinning.bindings:
        mesh_name = binding.mesh.name
        if mesh_name not in overrides:
            continue

        verts = overrides[mesh_name]
        V = binding.mesh.geometry.vertex_count
        for vi, info in verts.items():
            if vi >= V:
                continue
            binding.joint_indices[vi] = info["joint"]
            binding.secondary_indices[vi] = info["secondary"]
            binding.weights[vi] = info["weight"]
            count += 1

    logger.info("Applied %d vertex overrides", count)
    return count


def collect_overrides(skinning: SoftTissueSkinning) -> dict[str, dict[int, dict]]:
    """Collect current binding state as overrides dict.

    This captures ALL vertex assignments, not just modified ones.
    For differential overrides, compare with the original registration.
    """
    result: dict[str, dict[int, dict]] = {}
    for binding in skinning.bindings:
        if binding.is_muscle:
            continue
        mesh_data: dict[int, dict] = {}
        for vi in range(binding.mesh.geometry.vertex_count):
            ji = int(binding.joint_indices[vi])
            mesh_data[vi] = {
                "chain_id": int(skinning.joints[ji].chain_id),
                "joint": ji,
                "secondary": int(binding.secondary_indices[vi]),
                "weight": float(binding.weights[vi]),
            }
        result[binding.mesh.name] = mesh_data
    return result


def collect_modified_overrides(
    skinning: SoftTissueSkinning,
    modified_vertices: dict[int, set[int]],
) -> dict[str, dict[int, dict]]:
    """Collect overrides for only the modified vertices.

    Parameters
    ----------
    modified_vertices : dict
        ``{binding_idx: set of vertex indices}`` — vertices that were reassigned.
    """
    result: dict[str, dict[int, dict]] = {}
    for bi, vis in modified_vertices.items():
        if bi >= len(skinning.bindings):
            continue
        binding = skinning.bindings[bi]
        mesh_data: dict[int, dict] = {}
        for vi in vis:
            vi = int(vi)
            if vi >= binding.mesh.geometry.vertex_count:
                continue
            ji = int(binding.joint_indices[vi])
            mesh_data[vi] = {
                "chain_id": int(skinning.joints[ji].chain_id),
                "joint": ji,
                "secondary": int(binding.secondary_indices[vi]),
                "weight": float(binding.weights[vi]),
            }
        if mesh_data:
            result[binding.mesh.name] = mesh_data
    return result
