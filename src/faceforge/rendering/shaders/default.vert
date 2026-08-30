#version 330 core

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aColor;

uniform mat4 uModelView;
uniform mat4 uProjection;
uniform int uUseVertexColor;
uniform mat4 uModelMatrix;

// Clip plane, applied here rather than in the fragment shader.  See
// _common.glsl for why: a `discard` anywhere in a fragment shader statically
// disables early-Z on most drivers.
uniform int uClipEnabled;
uniform vec4 uClipPlane;

out vec3 vNormal;
out vec3 vViewPos;
out vec3 vVertexColor;
out vec3 vWorldPos;

void main() {
    vec4 mvPos = uModelView * vec4(aPosition, 1.0);
    vViewPos = mvPos.xyz;
    vWorldPos = (uModelMatrix * vec4(aPosition, 1.0)).xyz;

    // Normal matrix derived on the GPU instead of uploaded per mesh.
    // FaceForge applies no non-uniform node scale (the only set_scale() call in
    // the tree, anatomy/face.py:77, is uniform), so the inverse transpose of the
    // model-view 3x3 equals mat3(uModelView) up to a positive scalar, and the
    // fragment shader's normalize() removes that scalar.  Dropping
    // uNormalMatrix removes one np.linalg.inv plus one glUniformMatrix3fv per
    // mesh per frame.
    // The pre-interpolation normalize() is also gone: normalising before
    // interpolation does not make the interpolated value unit-length, and every
    // fragment shader normalises again anyway.
    // NOTE: if a non-uniform node scale is ever introduced, restore
    // uNormalMatrix -- lighting on those meshes would otherwise be wrong.
    vNormal = mat3(uModelView) * aNormal;

    vVertexColor = uUseVertexColor != 0 ? aColor : vec3(0.0);
    gl_Position = uProjection * mvPos;

    // Hardware clipping.  Negative => clipped, which reproduces the old
    // fragment test `dot(vWorldPos, uClipPlane.xyz) + uClipPlane.w < 0.0`
    // exactly, boundary included.  1.0 when the cutaway is off keeps every
    // vertex on the visible side even if GL_CLIP_DISTANCE0 is left enabled.
    // Writing index 0 with a constant implicitly sizes gl_ClipDistance to 1.
    gl_ClipDistance[0] = uClipEnabled != 0
        ? dot(vWorldPos, uClipPlane.xyz) + uClipPlane.w
        : 1.0;
}
