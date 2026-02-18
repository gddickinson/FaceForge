#version 330 core

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;

uniform mat4 uModelView;
uniform mat4 uProjection;
uniform mat3 uNormalMatrix;

out vec3 vNormal;
out vec3 vViewPos;

void main() {
    vec4 mvPos = uModelView * vec4(aPosition, 1.0);
    vViewPos = mvPos.xyz;
    vNormal = normalize(uNormalMatrix * aNormal);
    gl_Position = uProjection * mvPos;
}
