#version 330 core

in vec3 vNormal;
in vec3 vViewPos;
in vec3 vVertexColor;

uniform vec3 uColor;
uniform float uOpacity;

out vec4 fragColor;

void main() {
    fragColor = vec4(uColor, uOpacity);
}
