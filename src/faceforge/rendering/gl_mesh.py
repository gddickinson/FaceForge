"""VAO / VBO management for uploading and drawing mesh geometry.

Uses OpenGL 3.3 core profile. Each GLMesh owns one VAO with:
  - VBO slot 0: positions  (vec3, location 0)
  - VBO slot 1: normals    (vec3, location 1)
  - Optional EBO for indexed geometry
"""

import logging

import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_DYNAMIC_DRAW,
    GL_ELEMENT_ARRAY_BUFFER,
    GL_FLOAT,
    GL_FALSE,
    GL_POINTS,
    GL_STATIC_DRAW,
    GL_TRIANGLES,
    GL_UNSIGNED_INT,
    glBindBuffer,
    glBindVertexArray,
    glBufferData,
    glBufferSubData,
    glDeleteBuffers,
    glDeleteVertexArrays,
    glDrawArrays,
    glDrawElements,
    glEnableVertexAttribArray,
    glGenBuffers,
    glGenVertexArrays,
    glVertexAttribPointer,
)

from faceforge.core.material import RenderMode
from faceforge.core.mesh import BufferGeometry

logger = logging.getLogger(__name__)


class GLMesh:
    """GPU-side representation of a BufferGeometry.

    Call :meth:`upload` once (or after topology changes) and :meth:`draw`
    each frame. For per-frame vertex animation use :meth:`update_positions`
    and :meth:`update_normals` which stream new data via ``glBufferSubData``.
    """

    def __init__(self, geometry: BufferGeometry, *, dynamic: bool = False) -> None:
        self._geometry = geometry
        self._dynamic = dynamic

        self._vao: int = 0
        self._vbo_pos: int = 0
        self._vbo_norm: int = 0
        self._ebo: int = 0

        self._vertex_count: int = geometry.vertex_count
        self._index_count: int = 0
        self._has_indices: bool = geometry.has_indices
        self._uploaded: bool = False

    # ------------------------------------------------------------------
    # Upload (create GL objects)
    # ------------------------------------------------------------------

    def upload(self) -> None:
        """Create VAO, VBOs (and optional EBO) and upload vertex data."""
        if self._uploaded:
            self.destroy()

        usage = GL_DYNAMIC_DRAW if self._dynamic else GL_STATIC_DRAW

        # Generate GL objects
        self._vao = glGenVertexArrays(1)
        self._vbo_pos = glGenBuffers(1)
        self._vbo_norm = glGenBuffers(1)

        glBindVertexArray(self._vao)

        # --- Positions (location 0) ---
        pos_data = self._geometry.positions.astype(np.float32)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_pos)
        glBufferData(GL_ARRAY_BUFFER, pos_data.nbytes, pos_data, usage)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)

        # --- Normals (location 1) ---
        norm_data = self._geometry.normals.astype(np.float32)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_norm)
        glBufferData(GL_ARRAY_BUFFER, norm_data.nbytes, norm_data, usage)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)

        # --- Index buffer (optional) ---
        if self._has_indices:
            idx_data = self._geometry.indices.astype(np.uint32)
            self._index_count = len(idx_data)
            self._ebo = glGenBuffers(1)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._ebo)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx_data.nbytes, idx_data, GL_STATIC_DRAW)

        # Unbind VAO (leave EBO bound inside VAO state)
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        self._uploaded = True
        logger.debug(
            "GLMesh uploaded: %d verts, %d indices, dynamic=%s",
            self._vertex_count, self._index_count, self._dynamic,
        )

    # ------------------------------------------------------------------
    # Dynamic updates
    # ------------------------------------------------------------------

    def update_positions(self, positions: np.ndarray) -> None:
        """Stream new position data into the existing VBO."""
        if not self._uploaded:
            return
        data = positions.astype(np.float32)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_pos)
        glBufferSubData(GL_ARRAY_BUFFER, 0, data.nbytes, data)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def update_normals(self, normals: np.ndarray) -> None:
        """Stream new normal data into the existing VBO."""
        if not self._uploaded:
            return
        data = normals.astype(np.float32)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_norm)
        glBufferSubData(GL_ARRAY_BUFFER, 0, data.nbytes, data)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw(self, mode: RenderMode = RenderMode.SOLID) -> None:
        """Bind the VAO and issue the appropriate draw call.

        The caller is responsible for binding the shader and setting uniforms
        before calling this method.
        """
        if not self._uploaded:
            return

        gl_mode = self._gl_primitive(mode)

        glBindVertexArray(self._vao)

        if self._has_indices:
            glDrawElements(gl_mode, self._index_count, GL_UNSIGNED_INT, None)
        else:
            glDrawArrays(gl_mode, 0, self._vertex_count)

        glBindVertexArray(0)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def destroy(self) -> None:
        """Delete all owned GL resources."""
        if self._ebo:
            glDeleteBuffers(1, [self._ebo])
            self._ebo = 0
        if self._vbo_norm:
            glDeleteBuffers(1, [self._vbo_norm])
            self._vbo_norm = 0
        if self._vbo_pos:
            glDeleteBuffers(1, [self._vbo_pos])
            self._vbo_pos = 0
        if self._vao:
            glDeleteVertexArrays(1, [self._vao])
            self._vao = 0
        self._uploaded = False

    @property
    def uploaded(self) -> bool:
        return self._uploaded

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _gl_primitive(mode: RenderMode) -> int:
        """Map RenderMode to GL primitive type."""
        if mode == RenderMode.POINTS:
            return GL_POINTS
        # SOLID, WIREFRAME, and XRAY all draw triangles.
        # Wireframe rendering is handled by glPolygonMode(GL_LINE)
        # in gl_material.py, not by changing the primitive type.
        return GL_TRIANGLES
