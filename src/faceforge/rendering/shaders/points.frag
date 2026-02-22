#version 330 core

in vec3 vNormal;
in vec3 vViewPos;
in vec3 vWorldPos;

uniform vec3 uColor;
uniform float uOpacity;

// Clip plane
uniform int uClipEnabled;
uniform vec4 uClipPlane;

out vec4 fragColor;

void main() {
    if (uClipEnabled != 0 && dot(vWorldPos, uClipPlane.xyz) + uClipPlane.w < 0.0) discard;

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
