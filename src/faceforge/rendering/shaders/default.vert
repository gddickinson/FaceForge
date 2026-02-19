#version 330 core

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aColor;

uniform mat4 uModelView;
uniform mat4 uProjection;
uniform mat3 uNormalMatrix;
uniform int uUseVertexColor;

out vec3 vNormal;
out vec3 vViewPos;
out vec3 vVertexColor;

void main() {
    vec4 mvPos = uModelView * vec4(aPosition, 1.0);
    vViewPos = mvPos.xyz;
    vNormal = normalize(uNormalMatrix * aNormal);
    vVertexColor = uUseVertexColor != 0 ? aColor : vec3(0.0);
    gl_Position = uProjection * mvPos;
}
