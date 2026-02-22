"""Compile and link GLSL shader programs for OpenGL 3.3 core profile."""

import logging
from pathlib import Path

import numpy as np
from OpenGL.GL import (
    GL_COMPILE_STATUS,
    GL_FLOAT,
    GL_FRAGMENT_SHADER,
    GL_INFO_LOG_LENGTH,
    GL_LINK_STATUS,
    GL_VERTEX_SHADER,
    glAttachShader,
    glCompileShader,
    glCreateProgram,
    glCreateShader,
    glDeleteProgram,
    glDeleteShader,
    glGetAttribLocation,
    glGetProgramInfoLog,
    glGetProgramiv,
    glGetShaderInfoLog,
    glGetShaderiv,
    glGetUniformLocation,
    glLinkProgram,
    glShaderSource,
    glUniform1f,
    glUniform1i,
    glUniform3f,
    glUniform4f,
    glUniformMatrix3fv,
    glUniformMatrix4fv,
    glUseProgram,
)

logger = logging.getLogger(__name__)

# Directory containing GLSL shader source files
_SHADER_DIR = Path(__file__).parent / "shaders"


def load_shader_source(filename: str) -> str:
    """Load shader source from the shaders/ directory."""
    path = _SHADER_DIR / filename
    return path.read_text(encoding="utf-8")


class ShaderProgram:
    """Manages an OpenGL shader program (vertex + fragment).

    Compiles GLSL sources, links them into a program, and provides helpers
    for setting uniform values. Uniform locations are cached after first lookup.
    """

    def __init__(self, vertex_source: str, fragment_source: str) -> None:
        self._vertex_source = vertex_source
        self._fragment_source = fragment_source
        self._program: int = 0
        self._uniform_cache: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    def compile(self) -> None:
        """Compile vertex and fragment shaders and link the program.

        Raises ``RuntimeError`` on compilation or linking failure.
        """
        vert = self._compile_shader(GL_VERTEX_SHADER, self._vertex_source)
        frag = self._compile_shader(GL_FRAGMENT_SHADER, self._fragment_source)

        program = glCreateProgram()
        glAttachShader(program, vert)
        glAttachShader(program, frag)
        glLinkProgram(program)

        # Check link status
        if glGetProgramiv(program, GL_LINK_STATUS) != 1:
            info = glGetProgramInfoLog(program)
            if isinstance(info, bytes):
                info = info.decode("utf-8", errors="replace")
            glDeleteProgram(program)
            glDeleteShader(vert)
            glDeleteShader(frag)
            raise RuntimeError(f"Shader program link error:\n{info}")

        # Shaders can be freed after linking
        glDeleteShader(vert)
        glDeleteShader(frag)

        self._program = program
        self._uniform_cache.clear()
        logger.debug("Shader program %d compiled and linked.", program)

    # ------------------------------------------------------------------
    # Usage
    # ------------------------------------------------------------------

    def use(self) -> None:
        """Bind this shader program for subsequent draw calls."""
        glUseProgram(self._program)

    @property
    def program_id(self) -> int:
        return self._program

    # ------------------------------------------------------------------
    # Uniform setters
    # ------------------------------------------------------------------

    def set_uniform_mat4(self, name: str, mat4_array: np.ndarray) -> None:
        """Set a mat4 uniform. *mat4_array* must be a (4,4) float array."""
        loc = self.get_uniform_location(name)
        if loc < 0:
            return
        # Our math_utils stores row-major; OpenGL expects column-major.
        # Pre-transpose in Python and pass GL_FALSE because macOS core
        # profile rejects transpose=1.
        col_major = np.ascontiguousarray(mat4_array.T, dtype=np.float32)
        glUniformMatrix4fv(loc, 1, False, col_major)

    def set_uniform_mat3(self, name: str, mat3_array: np.ndarray) -> None:
        """Set a mat3 uniform. *mat3_array* must be a (3,3) float array."""
        loc = self.get_uniform_location(name)
        if loc < 0:
            return
        col_major = np.ascontiguousarray(mat3_array.T, dtype=np.float32)
        glUniformMatrix3fv(loc, 1, False, col_major)

    def set_uniform_vec3(self, name: str, vec3: tuple | np.ndarray) -> None:
        """Set a vec3 uniform from a 3-element sequence."""
        loc = self.get_uniform_location(name)
        if loc < 0:
            return
        glUniform3f(loc, float(vec3[0]), float(vec3[1]), float(vec3[2]))

    def set_uniform_vec4(self, name: str, vec4: tuple | np.ndarray) -> None:
        """Set a vec4 uniform from a 4-element sequence."""
        loc = self.get_uniform_location(name)
        if loc < 0:
            return
        glUniform4f(loc, float(vec4[0]), float(vec4[1]), float(vec4[2]), float(vec4[3]))

    def set_uniform_float(self, name: str, value: float) -> None:
        loc = self.get_uniform_location(name)
        if loc < 0:
            return
        glUniform1f(loc, float(value))

    def set_uniform_int(self, name: str, value: int) -> None:
        loc = self.get_uniform_location(name)
        if loc < 0:
            return
        glUniform1i(loc, int(value))

    # ------------------------------------------------------------------
    # Attribute / uniform location queries
    # ------------------------------------------------------------------

    def get_attrib_location(self, name: str) -> int:
        return glGetAttribLocation(self._program, name)

    def get_uniform_location(self, name: str) -> int:
        """Return the uniform location, caching results."""
        cached = self._uniform_cache.get(name)
        if cached is not None:
            return cached
        loc = glGetUniformLocation(self._program, name)
        self._uniform_cache[name] = loc
        if loc < 0:
            logger.debug("Uniform '%s' not found (may be optimised out).", name)
        return loc

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def destroy(self) -> None:
        """Delete the GL program."""
        if self._program:
            glDeleteProgram(self._program)
            self._program = 0
            self._uniform_cache.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compile_shader(shader_type: int, source: str) -> int:
        """Compile a single shader stage and return its GL handle."""
        shader = glCreateShader(shader_type)
        glShaderSource(shader, source)
        glCompileShader(shader)

        if glGetShaderiv(shader, GL_COMPILE_STATUS) != 1:
            info = glGetShaderInfoLog(shader)
            if isinstance(info, bytes):
                info = info.decode("utf-8", errors="replace")
            kind = "vertex" if shader_type == GL_VERTEX_SHADER else "fragment"
            glDeleteShader(shader)
            raise RuntimeError(f"{kind} shader compile error:\n{info}")

        return shader

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_files(cls, vert_filename: str, frag_filename: str) -> "ShaderProgram":
        """Create a ShaderProgram by loading source files from shaders/."""
        vert_src = load_shader_source(vert_filename)
        frag_src = load_shader_source(frag_filename)
        return cls(vert_src, frag_src)
