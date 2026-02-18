"""Minimal GL rendering diagnostic — renders a colored triangle.

Run: python gl_test.py
"""

import OpenGL
OpenGL.ERROR_CHECKING = False

import sys
import traceback
import numpy as np

from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import QTimer

from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST,
    GL_FLOAT, GL_FALSE, GL_TRIANGLES,
    GL_ARRAY_BUFFER, GL_STATIC_DRAW, GL_NO_ERROR,
    glClear, glClearColor, glEnable, glViewport, glGetError,
    glGenVertexArrays, glBindVertexArray, glGenBuffers, glBindBuffer,
    glBufferData, glVertexAttribPointer, glEnableVertexAttribArray,
    glDrawArrays, glUseProgram,
)


def log(msg):
    print(msg, flush=True)


def check_gl_error(label=""):
    err = glGetError()
    if err != GL_NO_ERROR:
        log(f"  GL ERROR at {label}: 0x{err:04X}")
        return False
    return True


class DiagWidget(QOpenGLWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GL Diagnostic")
        self.resize(800, 600)
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self.update)
        self._shader = 0
        self._vao = 0
        self._frame = 0
        self._init_ok = False

    def initializeGL(self):
        try:
            self._do_init()
        except Exception:
            log("!!! initializeGL CRASHED !!!")
            traceback.print_exc()
            sys.stdout.flush()

    def _do_init(self):
        log("\n=== GL Diagnostic ===")

        from OpenGL.GL import glGetString, GL_VERSION, GL_RENDERER, GL_SHADING_LANGUAGE_VERSION
        ver = glGetString(GL_VERSION)
        log(f"GL Version:  {ver.decode() if ver else 'None'}")
        rend = glGetString(GL_RENDERER)
        log(f"GL Renderer: {rend.decode() if rend else 'None'}")
        glsl = glGetString(GL_SHADING_LANGUAGE_VERSION)
        log(f"GLSL:        {glsl.decode() if glsl else 'None'}")
        check_gl_error("glGetString")

        # Device pixel ratio (Retina)
        dpr = self.devicePixelRatio()
        log(f"Device pixel ratio: {dpr}")
        log(f"Widget size: {self.width()}x{self.height()}")
        log(f"Framebuffer size: {int(self.width()*dpr)}x{int(self.height()*dpr)}")

        glClearColor(0.12, 0.12, 0.15, 1.0)
        glEnable(GL_DEPTH_TEST)

        # === Shader ===
        from OpenGL.GL import (
            glCreateShader, glShaderSource, glCompileShader,
            glGetShaderiv, glGetShaderInfoLog,
            glCreateProgram, glAttachShader, glLinkProgram,
            glGetProgramiv, glGetProgramInfoLog, glDeleteShader,
            GL_VERTEX_SHADER, GL_FRAGMENT_SHADER,
            GL_COMPILE_STATUS, GL_LINK_STATUS,
        )

        vert_src = """#version 330 core
layout(location = 0) in vec3 aPosition;
uniform mat4 uMVP;
void main() {
    gl_Position = uMVP * vec4(aPosition, 1.0);
}
"""
        frag_src = """#version 330 core
uniform vec3 uColor;
out vec4 fragColor;
void main() {
    fragColor = vec4(uColor, 1.0);
}
"""

        vs = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vs, vert_src)
        glCompileShader(vs)
        status = glGetShaderiv(vs, GL_COMPILE_STATUS)
        log(f"Vertex shader compile status: {status}")
        if status != 1:
            log(f"  ERROR: {glGetShaderInfoLog(vs)}")
            return

        fs = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fs, frag_src)
        glCompileShader(fs)
        status = glGetShaderiv(fs, GL_COMPILE_STATUS)
        log(f"Fragment shader compile status: {status}")
        if status != 1:
            log(f"  ERROR: {glGetShaderInfoLog(fs)}")
            return

        self._shader = glCreateProgram()
        glAttachShader(self._shader, vs)
        glAttachShader(self._shader, fs)
        glLinkProgram(self._shader)
        status = glGetProgramiv(self._shader, GL_LINK_STATUS)
        log(f"Program link status: {status}")
        if status != 1:
            log(f"  ERROR: {glGetProgramInfoLog(self._shader)}")
            return
        glDeleteShader(vs)
        glDeleteShader(fs)
        check_gl_error("shader compile/link")

        # === Geometry: triangle in clip space ===
        positions = np.array([
            -0.5, -0.5, 0.0,
             0.5, -0.5, 0.0,
             0.0,  0.5, 0.0,
        ], dtype=np.float32)

        self._vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        log(f"VAO={self._vao}, VBO={vbo}")

        glBindVertexArray(self._vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, positions.nbytes, positions, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        check_gl_error("VAO setup")

        # === Test matrix upload ===
        from OpenGL.GL import glUniformMatrix4fv, glGetUniformLocation, glUniform3f
        glUseProgram(self._shader)
        mvp_loc = glGetUniformLocation(self._shader, "uMVP")
        color_loc = glGetUniformLocation(self._shader, "uColor")
        log(f"Uniform locations: uMVP={mvp_loc}, uColor={color_loc}")

        identity = np.eye(4, dtype=np.float32)

        # Test transpose=False (should always work)
        col_major = np.ascontiguousarray(identity.T, dtype=np.float32)
        glUniformMatrix4fv(mvp_loc, 1, False, col_major)
        ok1 = check_gl_error("mat4 transpose=False")
        log(f"Matrix upload (transpose=False): {'OK' if ok1 else 'FAILED'}")

        # Test transpose=True (may fail on macOS)
        glUniformMatrix4fv(mvp_loc, 1, True, identity)
        ok2 = check_gl_error("mat4 transpose=True")
        log(f"Matrix upload (transpose=True):  {'OK' if ok2 else 'FAILED'}")

        glUseProgram(0)

        self._init_ok = True
        log("\n=== Init complete, starting render loop ===")
        self._timer.start()

    def resizeGL(self, w, h):
        dpr = self.devicePixelRatio()
        pw, ph = int(w * dpr), int(h * dpr)
        glViewport(0, 0, pw, ph)
        if self._frame < 3:
            log(f"resizeGL: logical={w}x{h}, pixels={pw}x{ph}")

    def paintGL(self):
        try:
            self._do_paint()
        except Exception:
            if self._frame < 3:
                log("!!! paintGL CRASHED !!!")
                traceback.print_exc()
                sys.stdout.flush()

    def _do_paint(self):
        from OpenGL.GL import glUniformMatrix4fv, glGetUniformLocation, glUniform3f

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if self._shader and self._vao:
            glUseProgram(self._shader)

            mvp_loc = glGetUniformLocation(self._shader, "uMVP")
            identity = np.ascontiguousarray(np.eye(4, dtype=np.float32).T)
            glUniformMatrix4fv(mvp_loc, 1, False, identity)

            color_loc = glGetUniformLocation(self._shader, "uColor")
            glUniform3f(color_loc, 1.0, 0.8, 0.2)

            glBindVertexArray(self._vao)
            glDrawArrays(GL_TRIANGLES, 0, 3)
            glBindVertexArray(0)
            glUseProgram(0)

            if self._frame == 0:
                check_gl_error("first draw")
                log("First frame rendered (should see gold triangle)")
        else:
            if self._frame == 0:
                log(f"!!! No shader/VAO: shader={self._shader}, vao={self._vao}")

        self._frame += 1


def main():
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setSamples(4)
    fmt.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    w = DiagWidget()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
