"""MakeHuman .target file parser → dense delta array."""

import numpy as np
from numpy.typing import NDArray


def parse_target(text: str, vertex_count: int) -> NDArray[np.float32]:
    """Parse a MakeHuman ``.target`` file into a dense (V, 3) delta array.

    Target files are sparse ASCII: each data line has
    ``vertex_index dx dy dz``.  Lines starting with ``#`` are comments.

    Parameters
    ----------
    text : str
        The target file contents.
    vertex_count : int
        Total number of vertices in the base mesh (for output array size).

    Returns
    -------
    NDArray[np.float32]
        Shape ``(vertex_count, 3)`` with deltas at modified vertex indices
        and zeros elsewhere.
    """
    deltas = np.zeros((vertex_count, 3), dtype=np.float32)

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            idx = int(parts[0])
            dx, dy, dz = float(parts[1]), float(parts[2]), float(parts[3])
        except (ValueError, IndexError):
            continue
        if 0 <= idx < vertex_count:
            deltas[idx] = [dx, dy, dz]

    return deltas


def load_target_file(path, vertex_count: int) -> NDArray[np.float32]:
    """Load a MakeHuman ``.target`` file from disk.

    Parameters
    ----------
    path : str or Path
        Path to the ``.target`` file.
    vertex_count : int
        Total number of vertices in the base mesh.

    Returns
    -------
    NDArray[np.float32]
        Shape ``(vertex_count, 3)`` delta array.
    """
    with open(path, "r") as f:
        text = f.read()
    return parse_target(text, vertex_count)
