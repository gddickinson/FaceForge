#version 330 core

in vec3 vNormal;
in vec3 vViewPos;
in vec3 vWorldPos;

uniform vec3 uColor;
uniform float uOpacity;

out vec4 fragColor;

// This file deliberately does NOT include _common.glsl: it pairs with
// points.vert, which emits no vVertexColor, and a fragment `in` with no
// matching output in the preceding stage is a link error on strict drivers.
//
// It is also the one fragment shader that keeps a `discard` -- the circular
// sprite mask below needs it.  The clip-plane discard is gone: points.vert
// writes gl_ClipDistance[0], and a point primitive is clipped as a whole, which
// is exactly what the old per-fragment test did (all fragments of a point
// sprite receive the same vWorldPos).

void main() {
    // Discard fragments outside a circular point (anti-aliased disc)
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    float dist = dot(coord, coord);
    if (dist > 1.0) {
        discard;
    }

    // Soft edge for anti-aliasing
    float alpha = uOpacity * smoothstep(1.0, 0.8, dist);

    fragColor = vec4(uColor, alpha);
}
