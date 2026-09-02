"""One-sided distance-constraint relaxation for skin, after position-based dynamics.

Why this exists
---------------
``shoulder_deformation_diagnosis.md`` records two things about the existing
neighbour clamp in ``soft_tissue._apply_neighbor_clamp``:

* it does not converge -- per-pass clamp counts at full shoulder flexion run
  ``4234, 3944, 4167, 4240, 4310, 4275, ...`` and are still ~2,967 on the 13th
  and final pass, so it runs out of iterations mid-cascade; and
* it manufactures some of the separation it exists to remove -- 16% of
  separating edges join two vertices that are rigidly bound to the *same*
  joint, which undergo an identical affine transform and therefore cannot
  separate by deformation at all.

Both follow from what it does: it compares each vertex against the *average of
its neighbours* and snaps it back by a ratio.  That target moves as neighbours
move, so the iteration chases itself, and a vertex can be displaced even when
its own transform was correct.

This module constrains the quantity actually being measured instead.  For every
mesh edge it enforces

    |p_a - p_b| <= rest_length * (1 + slack)

by the position-based-dynamics projection of Müller et al. (2007): compute the
constraint violation, move both endpoints along the edge to remove it, and
iterate.  Three properties matter here:

* **One-sided.**  Skin stretches; only edges *longer* than the limit are
  projected.  Edges within the slack are untouched, so ordinary deformation is
  not stiffened -- which is what keeps median edge stretch at 1.000.
* **Local and symmetric.**  Each projection conserves the edge midpoint, so the
  pass adds no net translation and cannot drift the mesh.
* **Convergent.**  Each projection strictly reduces its own constraint's
  violation, and the Jacobi averaging below keeps a shared vertex from being
  over-corrected by several constraints at once.  The residual is reported so
  convergence is measured rather than assumed.

This is a step toward simulating the tissue rather than correcting the skinning
after the fact, but it is *not* a volumetric or finite-element model: there is
no volume preservation, no material model, no collision response, and no
inertia.  It is a geometric constraint solve on the existing surface mesh.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def relax_edges(
    positions: np.ndarray,
    edges: np.ndarray,
    rest_lengths: np.ndarray,
    *,
    slack: float = 0.05,
    iterations: int = 12,
    omega: float = 1.0,
    tol: float = 1e-4,
) -> dict:
    """Project over-long edges back toward their rest length, in place.

    Parameters
    ----------
    positions : (V, 3) float64
        Deformed positions, modified in place.
    edges : (E, 2) int
        Unique vertex pairs.
    rest_lengths : (E,) float64
        Edge lengths in the rest pose.
    slack :
        Fractional stretch allowed before an edge is constrained.  0.05 permits
        5% extension, which ordinary skin deformation stays inside.
    iterations :
        Maximum Jacobi sweeps.
    omega :
        Relaxation factor.  1.0 removes the whole violation per sweep; below 1
        under-relaxes, which trades sweeps for stability on stiff clusters.
    tol :
        Stop once the largest remaining violation falls below
        ``tol * median(rest_lengths)``.

    Returns
    -------
    dict
        ``iterations_run``, ``violations`` (count per sweep), ``max_residual``
        (per sweep) and ``converged``.  Reported so non-convergence is visible
        -- the failure mode of the pass this replaces.
    """
    pos = positions
    a = edges[:, 0]
    b = edges[:, 1]
    limit = rest_lengths * (1.0 + slack)
    stop = tol * float(np.median(rest_lengths))

    # Constraints touching each vertex: Jacobi averaging divides each vertex's
    # accumulated correction by this, so a vertex shared by many over-long
    # edges is not moved several times over for the same violation.
    counts = np.bincount(np.concatenate([a, b]), minlength=len(pos))
    counts = np.maximum(counts, 1).astype(np.float64)[:, None]

    violations: list[int] = []
    residuals: list[float] = []
    converged = False
    run = 0

    for run in range(1, iterations + 1):
        d = pos[b] - pos[a]
        length = np.linalg.norm(d, axis=1)
        over = length > limit
        n_over = int(over.sum())
        violations.append(n_over)

        if n_over == 0:
            residuals.append(0.0)
            converged = True
            break

        excess = length[over] - limit[over]
        residuals.append(float(excess.max()))
        if excess.max() < stop:
            converged = True
            break

        # Unit vector along each violating edge. length[over] > limit >= 0 and
        # limit is strictly positive for a real mesh, so no zero division.
        direction = d[over] / length[over][:, None]
        # Half the excess to each endpoint conserves the edge midpoint.
        correction = (0.5 * omega) * excess[:, None] * direction

        delta = np.zeros_like(pos)
        np.add.at(delta, a[over], correction)
        np.add.at(delta, b[over], -correction)
        pos += delta / counts

    if not converged:
        logger.debug(
            "Edge relaxation did not converge in %d sweeps: %d edges still "
            "over the limit, largest violation %.4g",
            run, violations[-1], residuals[-1],
        )

    return {
        "iterations_run": run,
        "violations": violations,
        "max_residual": residuals,
        "converged": converged,
    }
