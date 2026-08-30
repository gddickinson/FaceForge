#version 330 core

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;

uniform mat4 uModelView;
uniform mat4 uProjection;
uniform float uPointSize;
uniform mat4 uModelMatrix;

// Clip plane -- see default.vert.  points.vert MUST write gl_ClipDistance[0]
// whenever GL_CLIP_DISTANCE0 can be enabled on the context: an unwritten
// clip-distance output is undefined, which would clip point sprites at random.
uniform int uClipEnabled;
uniform vec4 uClipPlane;

out vec3 vNormal;
out vec3 vViewPos;
out vec3 vWorldPos;

void main() {
    vec4 mvPos = uModelView * vec4(aPosition, 1.0);
    vViewPos = mvPos.xyz;
    vWorldPos = (uModelMatrix * vec4(aPosition, 1.0)).xyz;
    vNormal = mat3(uModelView) * aNormal;  // view-space normal, matches default.vert

    gl_Position = uProjection * mvPos;

    // A point primitive is clipped as a whole, and all fragments of a point
    // sprite receive the same vWorldPos, so this is exactly equivalent to the
    // per-fragment test points.frag used to run -- points.frag keeps only its
    // circular-sprite discard.
    gl_ClipDistance[0] = uClipEnabled != 0
        ? dot(vWorldPos, uClipPlane.xyz) + uClipPlane.w
        : 1.0;

    // Scale point size by distance from camera for perspective effect
    float dist = length(mvPos.xyz);
    gl_PointSize = uPointSize * (100.0 / max(dist, 1.0));
}
