"""Face features: eyes, ears, nose cartilage, eyebrows, throat structures.

Loads 8 STL meshes from ``face_features.json`` and deforms animated features
per-frame based on face state (eye look, AU1/AU2/AU4 for eyebrows,
AU9 for nasal cartilage, ear wiggle).

Bilateral meshes (eyeballs, ears) are split by X coordinate into left/right
halves.  Eyeballs use pivot groups for rotation and include procedural iris,
pupil, cornea, and limbal ring geometry for realistic anatomical appearance.

This module has ZERO GL imports; all vertex math is done with NumPy.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from faceforge.core.math_utils import (
    Vec3, Quat, vec3,
    quat_from_axis_angle, quat_multiply, deg_to_rad,
)
from faceforge.core.mesh import BufferGeometry, MeshInstance
from faceforge.core.material import Material
from faceforge.core.scene_graph import SceneNode
from faceforge.core.state import FaceState
from faceforge.loaders.stl_batch_loader import CoordinateTransform, load_stl_batch, STLBatchResult

logger = logging.getLogger(__name__)


# Eye rotation limits (degrees)
EYE_H_MAX_DEG = 15.0
EYE_V_MAX_DEG = 10.0

# Eyebrow displacement magnitudes (face-mesh local units)
EYEBROW_AU1_INNER_Y = 0.6   # Inner raise
EYEBROW_AU2_OUTER_Y = 0.5   # Outer raise
EYEBROW_AU4_LOWER_Y = -0.4  # Lower

# Nasal cartilage AU9 displacement
NASAL_AU9_Y = 0.2    # Scrunch up
NASAL_AU9_Z = -0.15  # Pull back

# Ear wiggle displacement
EAR_WIGGLE_Y = 0.3   # Up
EAR_WIGGLE_Z = -0.15  # Back

# ── Eye anatomy constants ──
IRIS_ANGULAR_RADIUS_DEG = 28.0   # From front pole
CORNEA_ANGULAR_RADIUS_DEG = 32.0  # Slightly wider than iris
PUPIL_REST_ANGULAR_DEG = 9.0     # Pupil angular radius at rest
PUPIL_MIN_ANGULAR_DEG = 5.0      # Fully constricted
PUPIL_MAX_ANGULAR_DEG = 16.0     # Fully dilated
PUPIL_REST_RADIUS = 0.35
PUPIL_MIN_RADIUS = 0.20
PUPIL_MAX_RADIUS = 0.55
CORNEA_BULGE = 0.06  # Forward offset from sphere surface
LIMBAL_WIDTH_DEG = 2.0  # Limbal ring width in degrees
ANNULUS_SEGMENTS = 64
DISC_SEGMENTS = 32
DOME_SEGMENTS_THETA = 32
DOME_SEGMENTS_PHI = 8

# Colors (RGB float tuples)
SCLERA_COLOR = (0.93, 0.91, 0.88)
IRIS_OUTER_COLOR = (0.30, 0.18, 0.08)
IRIS_MID_COLOR = (0.42, 0.26, 0.13)
IRIS_INNER_COLOR = (0.55, 0.35, 0.18)
PUPIL_COLOR = (0.02, 0.02, 0.02)
CORNEA_COLOR = (1.0, 1.0, 1.0)
LIMBAL_COLOR = (0.12, 0.10, 0.08)


@dataclass
class EyeAssembly:
    """Runtime data for a single anatomical eye (left or right)."""
    side: str
    pivot: SceneNode
    orient_node: SceneNode  # intermediate node for iris orientation debug
    sclera: MeshInstance
    iris: MeshInstance
    pupil: MeshInstance
    cornea: MeshInstance
    limbal: MeshInstance
    center: NDArray[np.float64]
    forward: NDArray[np.float64]
    radius: float
    semi_axes: tuple[float, float, float]
    iris_rest_positions: NDArray[np.float32]
    pupil_rest_positions: NDArray[np.float32]


@dataclass
class EyeballData:
    """Legacy eyeball data — kept for backward compatibility."""
    mesh: MeshInstance
    node: SceneNode
    pivot: SceneNode
    rest_positions: NDArray[np.float32]
    center: NDArray[np.float64]
    side: str


@dataclass
class FeatureMeshData:
    """Runtime data for a generic animatable feature."""
    mesh: MeshInstance
    node: SceneNode
    rest_positions: NDArray[np.float32]
    rest_normals: NDArray[np.float32]
    vert_count: int
    category: str
    animated: bool


# ── Procedural geometry helpers ──


def _build_basis(forward: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """Build right-handed orthonormal basis from *forward* direction."""
    up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(forward, up)) > 0.99:
        up = np.array([0.0, 0.0, 1.0])
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    actual_up = np.cross(forward, right)
    actual_up /= np.linalg.norm(actual_up)
    return forward, right, actual_up


def _ellipsoid_point(
    center: NDArray,
    forward: NDArray,
    right: NDArray,
    actual_up: NDArray,
    phi: float,
    theta: float,
    semi_axes: tuple[float, float, float],
    offset: float = 0.0,
) -> tuple[NDArray, NDArray]:
    """Compute a point on an ellipsoid surface and its outward normal.

    *phi* = polar angle from forward pole (0 = pole, π/2 = equator).
    *theta* = azimuthal angle around forward axis.
    *semi_axes* = (a_x, a_y, a_z) semi-axis lengths of the sclera ellipsoid.

    The ellipsoid radius at direction *d* is::

        r(d) = 1 / sqrt((dx/ax)^2 + (dy/ay)^2 + (dz/az)^2)

    This ensures procedural geometry (iris, pupil, cornea) conforms to the
    actual sclera surface shape instead of clipping into it.
    """
    sp, cp = math.sin(phi), math.cos(phi)
    ct, st = math.cos(theta), math.sin(theta)
    direction = forward * cp + (right * ct + actual_up * st) * sp

    ax, ay, az = semi_axes
    dx, dy, dz = float(direction[0]), float(direction[1]), float(direction[2])
    r = 1.0 / math.sqrt((dx / ax) ** 2 + (dy / ay) ** 2 + (dz / az) ** 2)
    r += offset

    # Outward normal on ellipsoid: gradient of (x/a)^2+(y/b)^2+(z/c)^2
    # is proportional to (x/a^2, y/b^2, z/c^2) = (dx*r/a^2, dy*r/b^2, dz*r/c^2)
    raw_normal = np.array([dx / (ax * ax), dy / (ay * ay), dz / (az * az)])
    n_len = np.linalg.norm(raw_normal)
    normal = raw_normal / n_len if n_len > 1e-12 else direction

    return center + direction * r, normal


def _make_ellipsoid_cap(
    center: NDArray,
    forward: NDArray,
    semi_axes: tuple[float, float, float],
    angular_radius_deg: float,
    seg_theta: int,
    seg_phi: int,
    color: tuple[float, float, float],
    opacity: float = 1.0,
    shininess: float = 60.0,
    radius_offset: float = 0.0,
) -> MeshInstance:
    """Create a filled cap on an ellipsoid surface.

    Vertices conform to the ellipsoid shape defined by *semi_axes*,
    covering a cone of half-angle *angular_radius_deg* around *forward*.
    """
    fwd, right, actual_up = _build_basis(forward)
    max_phi = math.radians(angular_radius_deg)

    verts: list[NDArray] = []
    norms: list[NDArray] = []

    for i in range(seg_phi):
        phi0 = max_phi * i / seg_phi
        phi1 = max_phi * (i + 1) / seg_phi
        for j in range(seg_theta):
            theta0 = 2.0 * math.pi * j / seg_theta
            theta1 = 2.0 * math.pi * (j + 1) / seg_theta

            p00, n00 = _ellipsoid_point(center, fwd, right, actual_up, phi0, theta0, semi_axes, radius_offset)
            p10, n10 = _ellipsoid_point(center, fwd, right, actual_up, phi1, theta0, semi_axes, radius_offset)
            p01, n01 = _ellipsoid_point(center, fwd, right, actual_up, phi0, theta1, semi_axes, radius_offset)
            p11, n11 = _ellipsoid_point(center, fwd, right, actual_up, phi1, theta1, semi_axes, radius_offset)

            verts.extend([p00, p10, p11])
            norms.extend([n00, n10, n11])
            verts.extend([p00, p11, p01])
            norms.extend([n00, n11, n01])

    positions = np.array(verts, dtype=np.float32).ravel()
    normals_arr = np.array(norms, dtype=np.float32).ravel()

    geom = BufferGeometry(positions=positions, normals=normals_arr)
    mat = Material(
        color=color, opacity=opacity, shininess=shininess,
        transparent=opacity < 1.0, depth_write=opacity >= 0.5,
        double_sided=True,
    )
    return MeshInstance(name="ellipsoid_cap", geometry=geom, material=mat)


def _make_ellipsoid_annulus(
    center: NDArray,
    forward: NDArray,
    semi_axes: tuple[float, float, float],
    inner_angular_deg: float,
    outer_angular_deg: float,
    seg_theta: int,
    seg_phi: int,
    color: tuple[float, float, float],
    opacity: float = 1.0,
    shininess: float = 45.0,
    radius_offset: float = 0.0,
) -> MeshInstance:
    """Create an annulus (ring) on an ellipsoid surface.

    Covers the region between *inner_angular_deg* and *outer_angular_deg*
    from the forward pole, conforming to the ellipsoid shape.
    """
    fwd, right, actual_up = _build_basis(forward)
    inner_phi = math.radians(inner_angular_deg)
    outer_phi = math.radians(outer_angular_deg)

    verts: list[NDArray] = []
    norms: list[NDArray] = []

    for i in range(seg_phi):
        phi0 = inner_phi + (outer_phi - inner_phi) * i / seg_phi
        phi1 = inner_phi + (outer_phi - inner_phi) * (i + 1) / seg_phi
        for j in range(seg_theta):
            theta0 = 2.0 * math.pi * j / seg_theta
            theta1 = 2.0 * math.pi * (j + 1) / seg_theta

            p00, n00 = _ellipsoid_point(center, fwd, right, actual_up, phi0, theta0, semi_axes, radius_offset)
            p10, n10 = _ellipsoid_point(center, fwd, right, actual_up, phi1, theta0, semi_axes, radius_offset)
            p01, n01 = _ellipsoid_point(center, fwd, right, actual_up, phi0, theta1, semi_axes, radius_offset)
            p11, n11 = _ellipsoid_point(center, fwd, right, actual_up, phi1, theta1, semi_axes, radius_offset)

            verts.extend([p00, p10, p11])
            norms.extend([n00, n10, n11])
            verts.extend([p00, p11, p01])
            norms.extend([n00, n11, n01])

    positions = np.array(verts, dtype=np.float32).ravel()
    normals_arr = np.array(norms, dtype=np.float32).ravel()

    geom = BufferGeometry(positions=positions, normals=normals_arr)
    mat = Material(
        color=color, opacity=opacity, shininess=shininess,
        transparent=opacity < 1.0, depth_write=opacity >= 0.5,
        double_sided=True,
    )
    mesh = MeshInstance(name="ellipsoid_annulus", geometry=geom, material=mat)
    mesh.store_rest_pose()
    return mesh


def _make_dome_cap(
    center: NDArray,
    forward: NDArray,
    semi_axes: tuple[float, float, float],
    angular_radius_deg: float,
    seg_theta: int,
    seg_phi: int,
    color: tuple[float, float, float],
    opacity: float = 0.12,
    shininess: float = 120.0,
    radius_offset: float = 0.0,
) -> MeshInstance:
    """Create a dome cap on an ellipsoid surface (cornea).

    The cap extends from the front pole to *angular_radius_deg* around
    the *forward* axis, conforming to the ellipsoid shape.
    """
    mesh = _make_ellipsoid_cap(
        center=center,
        forward=forward,
        semi_axes=semi_axes,
        angular_radius_deg=angular_radius_deg,
        seg_theta=seg_theta,
        seg_phi=seg_phi,
        color=color,
        opacity=opacity,
        shininess=shininess,
        radius_offset=radius_offset,
    )
    # Override material for cornea (transparent, no depth write)
    mesh.material = Material(
        color=color, opacity=opacity, shininess=shininess,
        transparent=True, depth_write=False, double_sided=True,
    )
    mesh.name = "dome_cap"
    return mesh


def _make_iris_annulus(
    center: NDArray,
    forward: NDArray,
    semi_axes: tuple[float, float, float],
    inner_angular_deg: float,
    outer_angular_deg: float,
    seg_theta: int = ANNULUS_SEGMENTS,
    seg_phi: int = 4,
    radius_offset: float = 0.01,
) -> MeshInstance:
    """Create iris as an ellipsoidal annulus on the sclera surface.

    Uses ellipsoid-surface geometry so the iris follows the actual sclera
    curvature and is always visible from the front.
    """
    mesh = _make_ellipsoid_annulus(
        center=center,
        forward=forward,
        semi_axes=semi_axes,
        inner_angular_deg=inner_angular_deg,
        outer_angular_deg=outer_angular_deg,
        seg_theta=seg_theta,
        seg_phi=seg_phi,
        color=IRIS_MID_COLOR,
        opacity=1.0,
        shininess=45.0,
        radius_offset=radius_offset,
    )
    mesh.name = "iris"
    mesh.store_rest_pose()
    return mesh


class FaceFeatureSystem:
    """Manages face feature STL meshes: eyes, ears, nose cartilage, eyebrows, throat.

    Parameters
    ----------
    face_feature_defs:
        List of feature definition dicts loaded from ``face_features.json``.
        Each dict has: ``name``, ``stl``, ``type`` (bilateral/single),
        ``category``, ``color``, ``opacity``, ``shininess``, ``animated``.
    transform:
        BP3D-to-skull coordinate transform.
    """

    def __init__(
        self,
        face_feature_defs: list[dict],
        transform: Optional[CoordinateTransform] = None,
    ) -> None:
        self._defs = face_feature_defs
        self._transform = transform or CoordinateTransform()
        self._group: Optional[SceneNode] = None

        # Category-specific storage
        self._eyeballs: list[EyeballData] = []  # Legacy
        self._eye_assemblies: list[EyeAssembly] = []
        self._features: list[FeatureMeshData] = []

        # Categorized references for layer toggle
        self.categories: dict[str, list[SceneNode]] = {}

        # Track last pupil dilation to avoid redundant updates
        self._last_pupil_dilation: float = -1.0

        # Original bilateral eyeball mesh (hidden; replaced by per-side assemblies)
        self._hidden_bilateral_node: Optional[SceneNode] = None
        self._hidden_bilateral_mesh: Optional[MeshInstance] = None

    @property
    def group(self) -> Optional[SceneNode]:
        return self._group

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, stl_dir=None) -> SceneNode:
        """Load all face feature STL meshes.

        Bilateral meshes (eyeballs, ears) are split into left and right
        halves by X coordinate (X >= 0 is left, X < 0 is right).

        Returns the SceneNode group containing all feature nodes.
        """
        from faceforge.constants import STL_DIR
        kwargs = {}
        if stl_dir is not None:
            kwargs["stl_dir"] = stl_dir

        result: STLBatchResult = load_stl_batch(
            self._defs,
            label="face_features",
            transform=self._transform,
            indexed=True,
            **kwargs,
        )

        self._group = result.group

        for mesh, node, defn in zip(result.meshes, result.nodes, self._defs):
            category = defn.get("category", "other")
            animated = defn.get("animated", False)
            feat_type = defn.get("type", "single")

            # Track by category for layer toggles
            if category not in self.categories:
                self.categories[category] = []
            self.categories[category].append(node)

            if feat_type == "bilateral" and category == "eyes":
                # Split eyeball into left and right, create pivot groups
                # with full anatomical eye assembly
                self._create_eyeball_pair(mesh, node, defn)
            elif feat_type == "bilateral" and category != "eyes":
                # Bilateral non-eye features (ears): store as feature
                self._store_feature(mesh, node, category, animated)
            else:
                # Single features: nose cartilage, eyebrows, throat
                self._store_feature(mesh, node, category, animated)

        return self._group

    def _create_eyeball_pair(self, mesh: MeshInstance, node: SceneNode, defn: dict) -> None:
        """Split a bilateral eyeball mesh into L/R sclera and add procedural eye parts."""
        pos = mesh.geometry.positions.reshape(-1, 3)
        indices = mesh.geometry.indices

        # Hide the original shared mesh permanently.
        # Track it so set_category_visible won't re-enable it.
        node.visible = False
        mesh.visible = False
        self._hidden_bilateral_node = node
        self._hidden_bilateral_mesh = mesh

        for side, x_test in [("left", lambda x: x >= 0), ("right", lambda x: x < 0)]:
            # Find vertices belonging to this side
            side_mask = x_test(pos[:, 0])
            if not side_mask.any():
                continue

            # Extract triangles where ALL 3 verts are on this side
            if indices is not None:
                tri_indices = indices.reshape(-1, 3)
                tri_mask = side_mask[tri_indices[:, 0]] & side_mask[tri_indices[:, 1]] & side_mask[tri_indices[:, 2]]
                side_tris = tri_indices[tri_mask]

                # Compact: remap to new vertex indices
                unique_verts = np.unique(side_tris.ravel())
                new_positions = pos[unique_verts].copy()
                remap = np.zeros(len(pos), dtype=np.uint32)
                remap[unique_verts] = np.arange(len(unique_verts), dtype=np.uint32)
                new_indices = remap[side_tris.ravel()].astype(np.uint32)
            else:
                # Non-indexed: filter triangles by checking each group of 3
                tri_pos = pos.reshape(-1, 3, 3)
                tri_mask = x_test(tri_pos[:, :, 0]).all(axis=1)
                new_positions = tri_pos[tri_mask].reshape(-1, 3).copy()
                new_indices = None

            center = new_positions.mean(axis=0).astype(np.float64)

            # Forward direction: anatomical anterior = +Z in BP3D coordinate
            # system, which maps to +Z in skull space.
            forward = np.array([0.0, 0.0, 1.0])

            # Compute semi-axes of the sclera ellipsoid.  The BP3D transform
            # applies non-uniform scaling, making the eyeball an ellipsoid
            # (Y longest, Z shortest).  Using ellipsoidal geometry ensures
            # the iris/pupil/cornea conform to the actual sclera shape.
            local_pos_f64 = (new_positions - center).astype(np.float64)
            semi_x = float(local_pos_f64[:, 0].ptp() / 2)
            semi_y = float(local_pos_f64[:, 1].ptp() / 2)
            semi_z = float(local_pos_f64[:, 2].ptp() / 2)
            semi_axes = (semi_x, semi_y, semi_z)

            # The "radius" for compatibility is the polar radius at +Z
            radius = semi_z

            logger.info(
                "Eye %s: center=(%.2f, %.2f, %.2f) semi_axes=(%.3f, %.3f, %.3f) "
                "forward=(%.3f, %.3f, %.3f)",
                side, center[0], center[1], center[2],
                semi_x, semi_y, semi_z,
                forward[0], forward[1], forward[2],
            )

            # Create pivot node at eyeball center
            pivot = SceneNode(name=f"eyeball_pivot_{side}")
            pivot.set_position(float(center[0]), float(center[1]), float(center[2]))

            # ── 1. Sclera (the STL mesh, re-centered) ──
            local_positions = (new_positions - center).astype(np.float32)
            sclera_geom = BufferGeometry(
                positions=local_positions.ravel(),
                normals=np.zeros_like(local_positions).ravel(),
                indices=new_indices,
            )
            sclera_geom.compute_normals()
            sclera_mat = Material(
                color=SCLERA_COLOR, opacity=0.35, shininess=20.0,
                double_sided=True, transparent=True,
            )
            sclera = MeshInstance(name=f"sclera_{side}", geometry=sclera_geom, material=sclera_mat)
            sclera.store_rest_pose()

            sclera_node = SceneNode(name=f"sclera_{side}")
            sclera_node.mesh = sclera
            pivot.add(sclera_node)

            # Orientation node: sits between pivot and procedural parts so we
            # can rotate iris/pupil/cornea/limbal independently of the sclera.
            # Hardcoded pitch=90° rotates the iris to face forward.
            orient_node = SceneNode(name=f"iris_orient_{side}")
            q_orient = quat_from_axis_angle(vec3(1, 0, 0), deg_to_rad(90.0))
            orient_node.set_quaternion(q_orient)
            pivot.add(orient_node)

            # All procedural eye parts are generated in pivot-local coords
            # (center = origin).  They use ellipsoid-surface geometry so they
            # conform to the sclera's non-spherical shape.
            local_center = np.zeros(3, dtype=np.float64)

            # ── 2. Iris (ellipsoid annulus on sclera surface) ──
            iris = _make_iris_annulus(
                center=local_center,
                forward=forward,
                semi_axes=semi_axes,
                inner_angular_deg=PUPIL_REST_ANGULAR_DEG,
                outer_angular_deg=IRIS_ANGULAR_RADIUS_DEG,
                seg_theta=ANNULUS_SEGMENTS,
                seg_phi=4,
                radius_offset=0.02,
            )
            iris.name = f"iris_{side}"

            iris_node = SceneNode(name=f"iris_{side}")
            iris_node.mesh = iris
            orient_node.add(iris_node)

            # ── 3. Pupil (ellipsoid cap at front pole) ──
            pupil = _make_ellipsoid_cap(
                center=local_center,
                forward=forward,
                semi_axes=semi_axes,
                angular_radius_deg=PUPIL_REST_ANGULAR_DEG,
                seg_theta=DISC_SEGMENTS,
                seg_phi=4,
                color=PUPIL_COLOR,
                opacity=1.0,
                shininess=60.0,
                radius_offset=0.03,
            )
            pupil.name = f"pupil_{side}"
            pupil.store_rest_pose()

            pupil_node = SceneNode(name=f"pupil_{side}")
            pupil_node.mesh = pupil
            orient_node.add(pupil_node)

            # ── 4. Cornea (dome cap on ellipsoid, slightly larger) ──
            cornea = _make_dome_cap(
                center=local_center,
                forward=forward,
                semi_axes=semi_axes,
                angular_radius_deg=CORNEA_ANGULAR_RADIUS_DEG,
                seg_theta=DOME_SEGMENTS_THETA,
                seg_phi=DOME_SEGMENTS_PHI,
                color=CORNEA_COLOR,
                opacity=0.12,
                shininess=120.0,
                radius_offset=CORNEA_BULGE,
            )
            cornea.name = f"cornea_{side}"

            cornea_node = SceneNode(name=f"cornea_{side}")
            cornea_node.mesh = cornea
            orient_node.add(cornea_node)

            # ── 5. Limbal ring (ellipsoid annulus at iris-sclera boundary) ──
            limbal = _make_ellipsoid_annulus(
                center=local_center,
                forward=forward,
                semi_axes=semi_axes,
                inner_angular_deg=IRIS_ANGULAR_RADIUS_DEG - LIMBAL_WIDTH_DEG / 2,
                outer_angular_deg=IRIS_ANGULAR_RADIUS_DEG + LIMBAL_WIDTH_DEG / 2,
                seg_theta=ANNULUS_SEGMENTS,
                seg_phi=2,
                color=LIMBAL_COLOR,
                opacity=0.85,
                shininess=15.0,
                radius_offset=0.01,
            )
            limbal.name = f"limbal_{side}"

            limbal_node = SceneNode(name=f"limbal_{side}")
            limbal_node.mesh = limbal
            orient_node.add(limbal_node)

            # Log iris orientation info
            iris_v = iris.geometry.positions.reshape(-1, 3)
            logger.info(
                "Eye %s iris verts[0..2]: (%.3f,%.3f,%.3f) (%.3f,%.3f,%.3f) (%.3f,%.3f,%.3f)",
                side,
                iris_v[0, 0], iris_v[0, 1], iris_v[0, 2],
                iris_v[1, 0], iris_v[1, 1], iris_v[1, 2],
                iris_v[2, 0], iris_v[2, 1], iris_v[2, 2],
            )

            # Add pivot to the feature group
            self._group.add(pivot)

            # Store eye assembly
            self._eye_assemblies.append(EyeAssembly(
                side=side,
                pivot=pivot,
                orient_node=orient_node,
                sclera=sclera,
                iris=iris,
                pupil=pupil,
                cornea=cornea,
                limbal=limbal,
                center=center,
                forward=forward,
                radius=radius,
                semi_axes=semi_axes,
                iris_rest_positions=iris.geometry.positions.copy(),
                pupil_rest_positions=pupil.geometry.positions.copy(),
            ))

            # Also keep legacy eyeball data for compatibility
            self._eyeballs.append(EyeballData(
                mesh=sclera,
                node=sclera_node,
                pivot=pivot,
                rest_positions=sclera.geometry.positions.copy(),
                center=center,
                side=side,
            ))

    def _store_feature(self, mesh: MeshInstance, node: SceneNode, category: str, animated: bool) -> None:
        """Store a generic feature mesh for per-frame updates."""
        rest_pos = mesh.geometry.positions.copy()
        rest_nrm = mesh.geometry.normals.copy()
        vert_count = mesh.geometry.vertex_count

        self._features.append(FeatureMeshData(
            mesh=mesh,
            node=node,
            rest_positions=rest_pos,
            rest_normals=rest_nrm,
            vert_count=vert_count,
            category=category,
            animated=animated,
        ))

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def update(self, face_state: FaceState) -> None:
        """Update all animated face features from current face state.

        - Eyeballs: rotate based on ``eye_look_x`` and ``eye_look_y``
        - Pupil: dilate/constrict based on ``pupil_dilation``
        - Eyebrows: deform based on AU1, AU2, AU4
        - Nasal cartilages: deform based on AU9
        - Ears: displace based on ``ear_wiggle``
        - Throat structures: no deformation (static)
        """
        self._update_eyeballs(face_state)
        self._update_animated_features(face_state)

    def _update_eyeballs(self, face_state: FaceState) -> None:
        """Rotate eyeball meshes and handle pupil dilation."""
        h_angle = deg_to_rad(face_state.eye_look_x * EYE_H_MAX_DEG)
        v_angle = deg_to_rad(face_state.eye_look_y * EYE_V_MAX_DEG)

        # Compose rotation: horizontal around Z, then vertical around X
        qh = quat_from_axis_angle(vec3(0, 0, 1), -h_angle)
        qv = quat_from_axis_angle(vec3(1, 0, 0), v_angle)
        q = quat_multiply(qh, qv)

        # Rotate all eye assemblies (orient_node quaternion is set once at creation)
        for eye in self._eye_assemblies:
            eye.pivot.set_quaternion(q)
            eye.pivot.mark_dirty()

        # Also rotate legacy eyeball pivots (if any without assemblies)
        assembly_pivots = {id(e.pivot) for e in self._eye_assemblies}
        for eye in self._eyeballs:
            if id(eye.pivot) not in assembly_pivots:
                eye.pivot.set_quaternion(q)
                eye.pivot.mark_dirty()

        # Pupil dilation
        dilation = face_state.pupil_dilation
        if abs(dilation - self._last_pupil_dilation) > 1e-4:
            self._last_pupil_dilation = dilation
            self._apply_pupil_dilation(dilation)

    def _apply_pupil_dilation(self, dilation: float) -> None:
        """Scale pupil cap and iris inner ring based on dilation value (0-1).

        Both pupil and iris use ellipsoid-surface geometry.  Dilation works by
        rescaling each vertex's polar angle from the forward axis, then
        recomputing its position on the ellipsoid surface at the new angle.
        """
        new_pupil_deg = PUPIL_MIN_ANGULAR_DEG + (PUPIL_MAX_ANGULAR_DEG - PUPIL_MIN_ANGULAR_DEG) * dilation
        angular_scale = new_pupil_deg / PUPIL_REST_ANGULAR_DEG

        for eye in self._eye_assemblies:
            fwd = eye.forward
            ax, ay, az = eye.semi_axes

            # ── Pupil: rescale angular positions on ellipsoid ──
            rest_pupil = eye.pupil_rest_positions.reshape(-1, 3)
            out_pupil = rest_pupil.copy()

            for i in range(len(out_pupil)):
                v = rest_pupil[i].astype(np.float64)
                v_len = np.linalg.norm(v)
                if v_len < 1e-6:
                    continue
                direction = v / v_len
                fwd_comp = float(np.dot(direction, fwd))
                fwd_comp = max(-1.0, min(1.0, fwd_comp))
                phi = math.acos(fwd_comp)
                if phi < 1e-6:
                    continue
                new_phi = phi * angular_scale
                lateral = direction - fwd * fwd_comp
                lat_len = np.linalg.norm(lateral)
                if lat_len < 1e-6:
                    continue
                lateral /= lat_len
                new_dir = fwd * math.cos(new_phi) + lateral * math.sin(new_phi)
                # Recompute ellipsoidal radius at new direction
                dx, dy, dz = float(new_dir[0]), float(new_dir[1]), float(new_dir[2])
                new_r = 1.0 / math.sqrt((dx / ax) ** 2 + (dy / ay) ** 2 + (dz / az) ** 2)
                # Preserve the original offset above surface
                orig_r = 1.0 / math.sqrt((direction[0] / ax) ** 2 + (direction[1] / ay) ** 2 + (direction[2] / az) ** 2)
                offset = v_len - orig_r
                out_pupil[i] = (new_dir * (new_r + offset)).astype(np.float32)

            eye.pupil.geometry.positions[:] = out_pupil.ravel()
            eye.pupil.needs_update = True

            # ── Iris: scale inner ring vertices to match new pupil edge ──
            rest_iris = eye.iris_rest_positions.reshape(-1, 3)
            out_iris = rest_iris.copy()
            rest_pupil_rad = math.radians(PUPIL_REST_ANGULAR_DEG)
            threshold_rad = rest_pupil_rad * 1.3

            for i in range(len(out_iris)):
                v = rest_iris[i].astype(np.float64)
                v_len = np.linalg.norm(v)
                if v_len < 1e-6:
                    continue
                direction = v / v_len
                fwd_comp = float(np.dot(direction, fwd))
                fwd_comp = max(-1.0, min(1.0, fwd_comp))
                phi = math.acos(fwd_comp)
                if phi > threshold_rad:
                    continue
                new_phi = phi * angular_scale
                lateral = direction - fwd * fwd_comp
                lat_len = np.linalg.norm(lateral)
                if lat_len < 1e-6:
                    continue
                lateral /= lat_len
                new_dir = fwd * math.cos(new_phi) + lateral * math.sin(new_phi)
                dx, dy, dz = float(new_dir[0]), float(new_dir[1]), float(new_dir[2])
                new_r = 1.0 / math.sqrt((dx / ax) ** 2 + (dy / ay) ** 2 + (dz / az) ** 2)
                orig_r = 1.0 / math.sqrt((direction[0] / ax) ** 2 + (direction[1] / ay) ** 2 + (direction[2] / az) ** 2)
                offset = v_len - orig_r
                out_iris[i] = (new_dir * (new_r + offset)).astype(np.float32)

            eye.iris.geometry.positions[:] = out_iris.ravel()
            eye.iris.needs_update = True

    def _update_animated_features(self, face_state: FaceState) -> None:
        """Deform animated feature meshes based on AU values and ear wiggle."""
        for feat in self._features:
            if not feat.animated:
                continue

            rest = feat.rest_positions.reshape(-1, 3)
            out = rest.copy()

            if feat.category == "eyebrows":
                out = self._deform_eyebrows(out, feat, face_state)
            elif feat.category == "nose":
                out = self._deform_nasal_cartilage(out, feat, face_state)
            elif feat.category == "ears":
                out = self._deform_ears(out, feat, face_state)

            feat.mesh.geometry.positions[:] = out.ravel()
            feat.mesh.needs_update = True

    def _deform_eyebrows(
        self,
        pos: NDArray[np.float32],
        feat: FeatureMeshData,
        face_state: FaceState,
    ) -> NDArray[np.float32]:
        """Apply AU1 (inner raise), AU2 (outer raise), AU4 (lower) to eyebrows.

        Inner/outer weighting is based on X distance from the nose bridge
        (midline X=0).  Vertices closer to midline are 'inner', farther are 'outer'.
        """
        au1 = face_state.AU1
        au2 = face_state.AU2
        au4 = face_state.AU4

        if au1 < 1e-4 and au2 < 1e-4 and au4 < 1e-4:
            return pos

        abs_x = np.abs(pos[:, 0])
        x_min, x_max = abs_x.min(), abs_x.max()
        x_range = x_max - x_min
        if x_range < 1e-6:
            return pos

        # Normalised: 0 = inner, 1 = outer
        inner_outer = (abs_x - x_min) / x_range
        inner_w = 1.0 - inner_outer  # Higher near midline
        outer_w = inner_outer         # Higher near temples

        # AU1: raise inner brows
        pos[:, 1] += EYEBROW_AU1_INNER_Y * au1 * inner_w

        # AU2: raise outer brows
        pos[:, 1] += EYEBROW_AU2_OUTER_Y * au2 * outer_w

        # AU4: lower all brow verts
        pos[:, 1] += EYEBROW_AU4_LOWER_Y * au4

        return pos

    def _deform_nasal_cartilage(
        self,
        pos: NDArray[np.float32],
        feat: FeatureMeshData,
        face_state: FaceState,
    ) -> NDArray[np.float32]:
        """Apply AU9 (nose wrinkle / scrunch) to nasal cartilages."""
        au9 = face_state.AU9
        if au9 < 1e-4:
            return pos

        pos[:, 1] += NASAL_AU9_Y * au9   # Up
        pos[:, 2] += NASAL_AU9_Z * au9   # Back

        return pos

    def _deform_ears(
        self,
        pos: NDArray[np.float32],
        feat: FeatureMeshData,
        face_state: FaceState,
    ) -> NDArray[np.float32]:
        """Apply ear wiggle displacement."""
        wiggle = face_state.ear_wiggle
        if abs(wiggle) < 1e-4:
            return pos

        pos[:, 1] += EAR_WIGGLE_Y * wiggle   # Up
        pos[:, 2] += EAR_WIGGLE_Z * wiggle   # Back

        return pos

    # ------------------------------------------------------------------
    # Eye color
    # ------------------------------------------------------------------

    def set_eye_color(self, color_rgb: tuple[float, float, float]) -> None:
        """Set iris color on both eye assemblies.

        *color_rgb* is an (R, G, B) tuple with values in 0-1.
        """
        for eye in self._eye_assemblies:
            eye.iris.material.color = color_rgb
            eye.iris.needs_update = True

    # ------------------------------------------------------------------
    # Layer visibility
    # ------------------------------------------------------------------

    def set_category_visible(self, category: str, visible: bool) -> None:
        """Toggle visibility of a category of features.

        Categories: ``"eyes"``, ``"ears"``, ``"nose"``, ``"eyebrows"``, ``"throat"``.
        """
        for node in self.categories.get(category, []):
            # Never re-show the original bilateral eyeball mesh — it's
            # replaced by per-side EyeAssembly meshes.
            if node is self._hidden_bilateral_node:
                node.visible = False
                if node.mesh is not None:
                    node.mesh.visible = False
                continue
            node.visible = visible
            if node.mesh is not None:
                node.mesh.visible = visible

        # For eyes, also toggle eye assembly pivots
        if category == "eyes":
            for eye in self._eye_assemblies:
                eye.pivot.visible = visible
                for child in eye.pivot.children:
                    child.visible = visible
                    if child.mesh is not None:
                        child.mesh.visible = visible
