// Shared fragment-shader prelude for the 15 modes that pair with default.vert.
//
// Pulled in with `#include "_common.glsl"`, which load_shader_source() in
// shader_program.py expands textually before the source reaches the driver.
// GLSL has no native #include (ARB_shading_language_include is not part of
// 3.30 core), so the expansion is done in Python; the directive is therefore
// FaceForge's own, not the driver's.
//
// Before this file, the varying block, the uniform block and the
// clip/lighting/Fresnel arithmetic were copy-pasted into 14-16 fragment
// shaders.  phong.frag is what that costs: it had already drifted out of sync
// and silently lost the clip-plane uniforms, so reinstating it would have given
// a shader that ignores the cutaway with no error (set_uniform_* quietly
// no-ops on a -1 location).
//
// Deliberately NOT included here:
//
//   * the `#version` directive -- it must be the first line of the translation
//     unit, so each mode file keeps its own;
//   * `uShininess` -- 13 of 16 modes use it, but ffSpecular() takes it as a
//     parameter, so the three that do not need never declare it;
//   * the directional-light uniforms -- see _lighting.glsl;
//   * anything used by only one mode.
//
// The split is not cosmetic.  Nothing in this file is *used* by a helper unless
// every including mode already declared it, so no mode gains a live uniform it
// did not have before, and per-mode uniform-upload counts are unchanged.
// points.frag deliberately does not include this prelude: it pairs with
// points.vert, which emits no vVertexColor, and a fragment `in` with no
// matching output in the previous stage is a link error on strict drivers.

// ---------------------------------------------------------------------------
// Varyings written by default.vert
// ---------------------------------------------------------------------------
in vec3 vNormal;
in vec3 vViewPos;
in vec3 vVertexColor;
in vec3 vWorldPos;

// ---------------------------------------------------------------------------
// Uniforms declared by all 15 modes that include this file
// ---------------------------------------------------------------------------
uniform vec3 uColor;
uniform float uOpacity;
uniform int uUseVertexColor;

out vec4 fragColor;

// ---------------------------------------------------------------------------
// Clip plane
// ---------------------------------------------------------------------------
// NOTE: there is deliberately no clip test in this prelude, and no
// uClipEnabled / uClipPlane declaration.  Clipping is done in the VERTEX stage
// via gl_ClipDistance[0] (see default.vert / points.vert).  The old
// fragment-stage form,
//
//     if (uClipEnabled != 0 && dot(vWorldPos, uClipPlane.xyz)
//                              + uClipPlane.w < 0.0) discard;
//
// appeared in 15 of 16 fragment shaders, and the mere *presence* of `discard`
// in a fragment shader is a static property of the program: most drivers
// respond by disabling early depth rejection for the whole draw call, whether
// or not any fragment is discarded and whether or not the cutaway is even
// switched on.  points.frag is the one exception -- it needs `discard` for the
// circular point sprite regardless.

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Interpolated surface normal, renormalised.  (Interpolating unit vectors does
/// not preserve unit length, which is why every mode needs this.)
vec3 ffNormal() { return normalize(vNormal); }

/// Direction from the fragment toward the eye, in view space.
vec3 ffViewDir() { return normalize(-vViewPos); }

/// Per-material base colour, honouring the vertex-colour override.
vec3 ffBaseColor() { return uUseVertexColor != 0 ? vVertexColor : uColor; }

/// Fresnel-style silhouette term: 0 where the surface faces the camera,
/// 1 at the silhouette edge.  11 of 16 modes are built on this.
float ffEdge(vec3 N, vec3 V) { return 1.0 - abs(dot(N, V)); }

/// Rec.601 luma, used by the 6 modes that desaturate or tone-map.
float ffLuma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

/// Phong specular term.  Takes shininess as a parameter rather than reading
/// uShininess, so that the three modes without that uniform can still include
/// this prelude.
float ffSpecular(vec3 N, vec3 V, vec3 L, float shininess) {
    return pow(max(dot(V, reflect(-L, N)), 0.0), shininess);
}
