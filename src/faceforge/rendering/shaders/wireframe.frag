#version 330 core

in vec3 vNormal;
in vec3 vViewPos;
in vec3 vVertexColor;
in vec3 vWorldPos;

uniform vec3 uColor;
uniform float uOpacity;
uniform int uUseVertexColor;

// Clip plane
uniform int uClipEnabled;
uniform vec4 uClipPlane;

out vec4 fragColor;

void main() {
    if (uClipEnabled != 0 && dot(vWorldPos, uClipPlane.xyz) + uClipPlane.w < 0.0) discard;

    vec3 color = uUseVertexColor != 0 ? vVertexColor : uColor;
    fragColor = vec4(color, uOpacity);
}
