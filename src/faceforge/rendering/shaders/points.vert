#version 330 core

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;

uniform mat4 uModelView;
uniform mat4 uProjection;
uniform float uPointSize;
uniform mat4 uModelMatrix;

out vec3 vNormal;
out vec3 vViewPos;
out vec3 vWorldPos;

void main() {
    vec4 mvPos = uModelView * vec4(aPosition, 1.0);
    vViewPos = mvPos.xyz;
    vWorldPos = (uModelMatrix * vec4(aPosition, 1.0)).xyz;
    vNormal = aNormal;  // Pass through for potential coloring

    gl_Position = uProjection * mvPos;

    // Scale point size by distance from camera for perspective effect
    float dist = length(mvPos.xyz);
    gl_PointSize = uPointSize * (100.0 / max(dist, 1.0));
}
