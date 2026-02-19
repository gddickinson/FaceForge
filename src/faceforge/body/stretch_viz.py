"""Stretch heatmap and chain assignment visualization for soft tissue skinning."""

from __future__ import annotations

import numpy as np

from faceforge.body.soft_tissue import SkinBinding, SoftTissueSkinning


# Skin base color (float 0..1)
_SKIN_R, _SKIN_G, _SKIN_B = 218 / 255, 190 / 255, 160 / 255

# Chain color palette (up to 32 chains)
CHAIN_COLORS = np.array([
    [0.2, 0.4, 0.9],   # 0 spine — blue
    [0.9, 0.2, 0.2],   # 1 arm_R — red
    [0.2, 0.8, 0.3],   # 2 leg_R — green
    [0.9, 0.5, 0.1],   # 3 arm_L — orange
    [0.6, 0.2, 0.8],   # 4 leg_L — purple
    [0.2, 0.8, 0.8],   # 5 ribs — cyan
    [0.9, 0.9, 0.2],   # 6 — yellow
    [0.9, 0.3, 0.6],   # 7 — pink
    [0.4, 0.7, 0.4],   # 8 — sage
    [0.7, 0.4, 0.2],   # 9 — brown
    [0.3, 0.6, 0.9],   # 10 — sky
    [0.8, 0.6, 0.3],   # 11 — tan
    [0.5, 0.3, 0.7],   # 12 — violet
    [0.3, 0.8, 0.6],   # 13 — teal
    [0.8, 0.3, 0.3],   # 14 — salmon
    [0.4, 0.5, 0.8],   # 15 — periwinkle
], dtype=np.float32)


def _stretch_to_colors(ratios: np.ndarray) -> np.ndarray:
    """Map per-vertex stretch ratios to RGB float colors.

    Same ramp as tools/mesh_renderer.py but in float 0..1.
    """
    V = len(ratios)
    colors = np.full((V, 3), [_SKIN_R, _SKIN_G, _SKIN_B], dtype=np.float32)

    # Compressed (<0.8): lerp skin → blue
    comp = ratios < 0.8
    if np.any(comp):
        t = np.clip((0.8 - ratios[comp]) / 0.4, 0, 1)
        colors[comp, 0] = _SKIN_R * (1 - t) + 0.314 * t
        colors[comp, 1] = _SKIN_G * (1 - t) + 0.471 * t
        colors[comp, 2] = _SKIN_B * (1 - t) + 0.863 * t

    # Normal (0.8–1.2): skin tone (already set)

    # Mild (1.2–1.5): lerp skin → yellow
    mild = (ratios >= 1.2) & (ratios < 1.5)
    if np.any(mild):
        t = (ratios[mild] - 1.2) / 0.3
        colors[mild, 0] = np.clip(_SKIN_R + (1.0 - _SKIN_R) * t, 0, 1)
        colors[mild, 1] = np.clip(_SKIN_G + (1.0 - _SKIN_G) * t * 0.9, 0, 1)
        colors[mild, 2] = np.clip(_SKIN_B * (1 - t * 0.85), 0, 1)

    # Moderate (1.5–2.0): lerp yellow → orange-red
    mod = (ratios >= 1.5) & (ratios < 2.0)
    if np.any(mod):
        t = (ratios[mod] - 1.5) / 0.5
        colors[mod, 0] = 1.0
        colors[mod, 1] = np.clip(0.824 * (1 - t * 0.8), 0, 1)
        colors[mod, 2] = 0.0

    # High (2.0–3.0): red
    high = (ratios >= 2.0) & (ratios < 3.0)
    if np.any(high):
        colors[high] = [1.0, 0.118, 0.0]

    # Extreme (3.0+): magenta
    ext = ratios >= 3.0
    if np.any(ext):
        colors[ext] = [1.0, 0.0, 0.784]

    return colors


def _compute_vertex_stretch(binding: SkinBinding) -> np.ndarray:
    """Compute per-vertex average edge stretch ratio.

    Returns (V,) float array of stretch ratios (1.0 = no stretch).
    Rest edge lengths are cached on first call since they never change.
    """
    if binding.edge_pairs is None:
        V = binding.mesh.geometry.vertex_count
        return np.ones(V, dtype=np.float32)

    edges = binding.edge_pairs  # (E, 2)
    v0, v1 = edges[:, 0], edges[:, 1]

    # Cache rest edge lengths (computed once, never changes)
    if not hasattr(binding, '_rest_edge_len') or binding._rest_edge_len is None:
        rest = binding.mesh.rest_positions.reshape(-1, 3)
        rest_len = np.linalg.norm(rest[v0] - rest[v1], axis=1)
        binding._rest_edge_len = np.maximum(rest_len, 1e-6)

    # Current edge lengths (changes every frame)
    pos = binding.mesh.geometry.positions.reshape(-1, 3)
    cur_len = np.linalg.norm(pos[v0] - pos[v1], axis=1)
    edge_ratio = cur_len / binding._rest_edge_len

    # Per-vertex: average stretch of incident edges
    V = len(pos)
    vertex_stretch = np.ones(V, dtype=np.float32)
    vertex_count = np.zeros(V, dtype=np.int32)

    np.add.at(vertex_stretch, v0, edge_ratio)
    np.add.at(vertex_count, v0, 1)
    np.add.at(vertex_stretch, v1, edge_ratio)
    np.add.at(vertex_count, v1, 1)

    # vertex_stretch started at 1.0 so subtract 1 for the ones with edges
    has_edges = vertex_count > 0
    vertex_stretch[has_edges] = (vertex_stretch[has_edges] - 1.0) / vertex_count[has_edges]
    # vertices with no edges stay at 1.0

    return vertex_stretch


def compute_chain_colors(skinning: SoftTissueSkinning, binding: SkinBinding) -> np.ndarray:
    """Compute per-vertex colors based on chain assignment.

    Returns (V*3,) float32 flat array for GL upload.
    """
    V = binding.mesh.geometry.vertex_count
    chain_ids = np.array([skinning.joints[ji].chain_id for ji in binding.joint_indices],
                         dtype=np.int32)

    # Map chain IDs to palette, wrapping around
    palette = CHAIN_COLORS
    color_idx = chain_ids % len(palette)
    colors = palette[color_idx]  # (V, 3)

    return colors.ravel().astype(np.float32)


class StretchVisualizer:
    """Per-vertex stretch heatmap visualization for skin bindings."""

    def __init__(self, skinning: SoftTissueSkinning) -> None:
        self.skinning = skinning
        self._stretch_enabled = False
        self._chain_enabled = False
        # Cache chain colors (only recompute on reassignment)
        self._chain_color_cache: dict[int, np.ndarray] = {}

    @property
    def stretch_enabled(self) -> bool:
        return self._stretch_enabled

    @stretch_enabled.setter
    def stretch_enabled(self, value: bool) -> None:
        self._stretch_enabled = value
        if not value and not self._chain_enabled:
            self._disable_all()

    @property
    def chain_enabled(self) -> bool:
        return self._chain_enabled

    @chain_enabled.setter
    def chain_enabled(self, value: bool) -> None:
        self._chain_enabled = value
        if value:
            # Rebuild cache
            self._chain_color_cache.clear()
        elif not self._stretch_enabled:
            self._disable_all()

    def update(self) -> None:
        """Recompute per-vertex colors for active visualization mode.

        Call once per frame when enabled.
        """
        if self._stretch_enabled:
            self._update_stretch()
        elif self._chain_enabled:
            self._update_chain()

    def invalidate_chain_cache(self) -> None:
        """Clear cached chain colors (call after vertex reassignment)."""
        self._chain_color_cache.clear()

    def _update_stretch(self) -> None:
        for binding in self.skinning.bindings:
            if binding.is_muscle:
                continue
            ratios = _compute_vertex_stretch(binding)
            colors = _stretch_to_colors(ratios)
            binding.mesh.geometry.vertex_colors = colors.ravel().astype(np.float32)
            binding.mesh.geometry.colors_dirty = True
            binding.mesh.material.vertex_colors_active = True

    def _update_chain(self) -> None:
        for binding in self.skinning.bindings:
            if binding.is_muscle:
                continue
            key = id(binding)
            if key not in self._chain_color_cache:
                self._chain_color_cache[key] = compute_chain_colors(
                    self.skinning, binding
                )
            binding.mesh.geometry.vertex_colors = self._chain_color_cache[key]
            binding.mesh.geometry.colors_dirty = True
            binding.mesh.material.vertex_colors_active = True

    def _disable_all(self) -> None:
        for binding in self.skinning.bindings:
            binding.mesh.material.vertex_colors_active = False
