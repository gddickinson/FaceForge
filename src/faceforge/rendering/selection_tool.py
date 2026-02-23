"""Interactive vertex selection tool using screen-space nearest-vertex projection."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import Qt

from faceforge.body.soft_tissue import SkinBinding, SoftTissueSkinning
from faceforge.core.mesh import MeshInstance
from faceforge.core.math_utils import Mat4


# Pixel radius for click selection
CLICK_THRESHOLD = 20.0


@dataclass
class SelectionState:
    """Tracks selected vertices across bindings."""
    # Map from binding index → set of vertex indices within that binding
    selected: dict[int, set[int]] = field(default_factory=dict)

    @property
    def total_count(self) -> int:
        return sum(len(s) for s in self.selected.values())

    def clear(self) -> None:
        self.selected.clear()

    def add(self, binding_idx: int, vertex_indices: set[int]) -> None:
        if binding_idx not in self.selected:
            self.selected[binding_idx] = set()
        self.selected[binding_idx].update(vertex_indices)

    def remove(self, binding_idx: int, vertex_indices: set[int]) -> None:
        if binding_idx in self.selected:
            self.selected[binding_idx] -= vertex_indices
            if not self.selected[binding_idx]:
                del self.selected[binding_idx]

    def get_flat_indices(self) -> list[tuple[int, list[int]]]:
        """Return list of (binding_idx, sorted vertex indices)."""
        return [(bi, sorted(vis)) for bi, vis in self.selected.items()]


class SelectionTool:
    """Vertex selection tool using screen-space projection.

    Supports click-to-select and lasso polygon selection.
    """

    def __init__(self, skinning: SoftTissueSkinning) -> None:
        self.skinning = skinning
        self.active = False
        self.selection = SelectionState()
        self.lasso_points: list[tuple[float, float]] = []
        self._dragging = False
        self._drag_started = False
        self._press_x = 0.0
        self._press_y = 0.0
        # Callback for selection changes
        self.on_selection_changed: callable | None = None
        # Body mesh for region label selection
        self.body_mesh: MeshInstance | None = None
        self.body_selection: set[int] = set()

    def on_mouse_press(self, x: float, y: float, button: int, modifiers=None) -> bool:
        """Handle mouse press. Returns True if consumed."""
        if not self.active or button != 1:  # Left button only
            return False
        self._press_x = x
        self._press_y = y
        self._dragging = True
        self._drag_started = False
        self.lasso_points = [(x, y)]
        return True

    def on_mouse_move(self, x: float, y: float) -> bool:
        """Handle mouse move. Returns True if in lasso drag."""
        if not self.active or not self._dragging:
            return False

        # Start lasso only after 5px drag distance
        dx = x - self._press_x
        dy = y - self._press_y
        if not self._drag_started and (dx * dx + dy * dy) > 25:
            self._drag_started = True

        if self._drag_started:
            self.lasso_points.append((x, y))
            return True
        return False

    def on_mouse_release(
        self,
        x: float,
        y: float,
        vp_matrix: Mat4,
        viewport_w: int,
        viewport_h: int,
        modifiers=None,
    ) -> bool:
        """Handle mouse release. Finalize click or lasso selection.

        Returns True if the event was consumed.
        """
        if not self.active or not self._dragging:
            return False

        self._dragging = False

        # Modifier flags — accept Qt.KeyboardModifier enum or int
        if modifiers is None:
            shift = ctrl = False
        else:
            shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
            ctrl = bool(modifiers & Qt.KeyboardModifier.ControlModifier)

        if self._drag_started and len(self.lasso_points) >= 3:
            # Lasso selection
            self._select_lasso(vp_matrix, viewport_w, viewport_h, shift, ctrl)
        else:
            # Click selection
            self._select_click(x, y, vp_matrix, viewport_w, viewport_h, shift, ctrl)

        self.lasso_points.clear()
        self._drag_started = False

        if self.on_selection_changed:
            self.on_selection_changed()

        return True

    def get_selected_positions(self) -> np.ndarray | None:
        """Get world positions of all selected vertices for rendering.

        Returns (N, 3) array or None if nothing selected.
        """
        total = self.selection.total_count + len(self.body_selection)
        if total == 0:
            return None

        parts = []
        for bi, vis in self.selection.selected.items():
            if bi < len(self.skinning.bindings):
                binding = self.skinning.bindings[bi]
                pos = binding.mesh.geometry.positions.reshape(-1, 3)
                idx = np.array(sorted(vis), dtype=np.int64)
                idx = idx[idx < len(pos)]
                if len(idx) > 0:
                    parts.append(pos[idx])

        # Include body mesh selections
        if self.body_mesh is not None and self.body_selection:
            pos = self.body_mesh.geometry.positions.reshape(-1, 3)
            idx = np.array(sorted(self.body_selection), dtype=np.int64)
            idx = idx[idx < len(pos)]
            if len(idx) > 0:
                parts.append(pos[idx])

        if not parts:
            return None
        return np.concatenate(parts, axis=0)

    def _project_vertices(
        self,
        binding: SkinBinding,
        vp_matrix: Mat4,
        viewport_w: int,
        viewport_h: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project binding vertices to screen coords.

        Returns (screen_xy (V,2), front_mask (V,), vertex_indices (V,)).
        """
        pos = binding.mesh.geometry.positions.reshape(-1, 3)
        V = len(pos)

        # Homogeneous coords
        ones = np.ones((V, 1), dtype=np.float32)
        pos4 = np.hstack([pos, ones])  # (V, 4)

        # VP transform
        clip = (vp_matrix @ pos4.T).T  # (V, 4)
        w = clip[:, 3]

        # Front-facing check (w > 0 means in front of camera)
        front = w > 0.01

        # NDC
        ndc = np.zeros((V, 3), dtype=np.float32)
        ndc[front] = clip[front, :3] / w[front, np.newaxis]

        # Screen coords
        sx = (ndc[:, 0] + 1.0) * viewport_w * 0.5
        sy = (1.0 - ndc[:, 1]) * viewport_h * 0.5

        screen = np.column_stack([sx, sy])
        return screen, front, np.arange(V)

    def _project_body_mesh(
        self,
        vp_matrix: Mat4,
        viewport_w: int,
        viewport_h: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Project body mesh vertices to screen coords.

        Returns (screen_xy (V,2), front_mask (V,), vertex_indices (V,))
        or None if no body mesh.
        """
        if self.body_mesh is None:
            return None

        pos = self.body_mesh.geometry.positions.reshape(-1, 3)
        V = len(pos)
        if V == 0:
            return None

        ones = np.ones((V, 1), dtype=np.float32)
        pos4 = np.hstack([pos, ones])

        clip = (vp_matrix @ pos4.T).T
        w = clip[:, 3]
        front = w > 0.01

        ndc = np.zeros((V, 3), dtype=np.float32)
        ndc[front] = clip[front, :3] / w[front, np.newaxis]

        sx = (ndc[:, 0] + 1.0) * viewport_w * 0.5
        sy = (1.0 - ndc[:, 1]) * viewport_h * 0.5

        screen = np.column_stack([sx, sy])
        return screen, front, np.arange(V)

    def _select_click(
        self,
        click_x: float,
        click_y: float,
        vp_matrix: Mat4,
        viewport_w: int,
        viewport_h: int,
        shift: bool,
        ctrl: bool,
    ) -> None:
        """Select the nearest vertex to click position."""
        best_dist = CLICK_THRESHOLD
        best_bi = -1
        best_vi = -1
        best_is_body = False

        for bi, binding in enumerate(self.skinning.bindings):
            if binding.is_muscle:
                continue
            screen, front, indices = self._project_vertices(
                binding, vp_matrix, viewport_w, viewport_h
            )
            if not np.any(front):
                continue

            # Only consider front-facing vertices
            fscreen = screen[front]
            findices = indices[front]

            dx = fscreen[:, 0] - click_x
            dy = fscreen[:, 1] - click_y
            dists = np.sqrt(dx * dx + dy * dy)

            min_idx = np.argmin(dists)
            if dists[min_idx] < best_dist:
                best_dist = dists[min_idx]
                best_bi = bi
                best_vi = int(findices[min_idx])
                best_is_body = False

        # Also check body mesh
        body_proj = self._project_body_mesh(vp_matrix, viewport_w, viewport_h)
        if body_proj is not None:
            screen, front, indices = body_proj
            if np.any(front):
                fscreen = screen[front]
                findices = indices[front]
                dx = fscreen[:, 0] - click_x
                dy = fscreen[:, 1] - click_y
                dists = np.sqrt(dx * dx + dy * dy)
                min_idx = np.argmin(dists)
                if dists[min_idx] < best_dist:
                    best_dist = dists[min_idx]
                    best_vi = int(findices[min_idx])
                    best_bi = -1
                    best_is_body = True

        if best_vi < 0:
            if not shift and not ctrl:
                self.selection.clear()
                self.body_selection.clear()
            return

        if best_is_body:
            if ctrl:
                self.body_selection.discard(best_vi)
            elif shift:
                self.body_selection.add(best_vi)
            else:
                self.selection.clear()
                self.body_selection = {best_vi}
        else:
            if ctrl:
                self.selection.remove(best_bi, {best_vi})
            elif shift:
                self.selection.add(best_bi, {best_vi})
            else:
                self.selection.clear()
                self.body_selection.clear()
                self.selection.add(best_bi, {best_vi})

    def _select_lasso(
        self,
        vp_matrix: Mat4,
        viewport_w: int,
        viewport_h: int,
        shift: bool,
        ctrl: bool,
    ) -> None:
        """Select all vertices inside the lasso polygon."""
        if len(self.lasso_points) < 3:
            return

        poly = np.array(self.lasso_points, dtype=np.float32)

        if not shift and not ctrl:
            self.selection.clear()
            self.body_selection.clear()

        for bi, binding in enumerate(self.skinning.bindings):
            if binding.is_muscle:
                continue
            screen, front, indices = self._project_vertices(
                binding, vp_matrix, viewport_w, viewport_h
            )
            if not np.any(front):
                continue

            fscreen = screen[front]
            findices = indices[front]

            inside = _points_in_polygon(fscreen, poly)
            selected = set(findices[inside].tolist())

            if ctrl:
                self.selection.remove(bi, selected)
            else:
                self.selection.add(bi, selected)

        # Also check body mesh
        body_proj = self._project_body_mesh(vp_matrix, viewport_w, viewport_h)
        if body_proj is not None:
            screen, front, indices = body_proj
            if np.any(front):
                fscreen = screen[front]
                findices = indices[front]
                inside = _points_in_polygon(fscreen, poly)
                selected = set(findices[inside].tolist())
                if ctrl:
                    self.body_selection -= selected
                else:
                    self.body_selection |= selected


def _points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Ray-casting point-in-polygon test.

    Parameters
    ----------
    points : (N, 2) array of test points
    polygon : (M, 2) array of polygon vertices

    Returns
    -------
    (N,) bool array — True for points inside polygon
    """
    N = len(points)
    M = len(polygon)
    inside = np.zeros(N, dtype=bool)

    px, py = points[:, 0], points[:, 1]

    j = M - 1
    for i in range(M):
        xi, yi = float(polygon[i, 0]), float(polygon[i, 1])
        xj, yj = float(polygon[j, 0]), float(polygon[j, 1])

        # Edge crosses the horizontal ray from point?
        # cond1 is True when the edge straddles py (one endpoint above, one below).
        # This also guarantees yi != yj, so division is safe.
        cond1 = (yi > py) != (yj > py)
        if np.any(cond1):
            dy = yj - yi  # non-zero where cond1 is True
            x_intersect = xi + (xj - xi) * (py - yi) / dy
            crossing = cond1 & (px < x_intersect)
            inside[crossing] = ~inside[crossing]

        j = i

    return inside
