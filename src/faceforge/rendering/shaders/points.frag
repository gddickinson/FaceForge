#version 330 core

in vec3 vNormal;
in vec3 vViewPos;

uniform vec3 uColor;
uniform float uOpacity;

out vec4 fragColor;

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
