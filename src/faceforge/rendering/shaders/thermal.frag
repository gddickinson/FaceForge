#version 330 core

// Thermal / infrared imaging.
// Maps surface angle and position to a heat-map colour ramp.

#include "_common.glsl"
#include "_lighting.glsl"

uniform float uShininess;

void main() {
    vec3 N = ffNormal();
    vec3 V = ffViewDir();
    vec3 L = ffLightDir();

    // Use facing angle + diffuse as "heat" proxy
    float facing = 1.0 - ffEdge(N, V);
    float NdotL = dot(N, L);
    float diff = clamp(NdotL, 0.0, 1.0);

    // Heat value: combination of facing (0=edge=cold) and light (warm)
    float heat = facing * 0.6 + diff * 0.4;

    // Use material colour luminance to vary heat across structures
    float lum = ffLuma(ffBaseColor());
    heat = heat * 0.7 + lum * 0.3;
    heat = clamp(heat, 0.0, 1.0);

    // 5-stop thermal ramp: black → blue → magenta → yellow → white
    vec3 col;
    if (heat < 0.25) {
        col = mix(vec3(0.0, 0.0, 0.08), vec3(0.0, 0.0, 0.8), heat / 0.25);
    } else if (heat < 0.50) {
        col = mix(vec3(0.0, 0.0, 0.8), vec3(0.85, 0.0, 0.65), (heat - 0.25) / 0.25);
    } else if (heat < 0.75) {
        col = mix(vec3(0.85, 0.0, 0.65), vec3(1.0, 0.9, 0.0), (heat - 0.50) / 0.25);
    } else {
        col = mix(vec3(1.0, 0.9, 0.0), vec3(1.0, 1.0, 1.0), (heat - 0.75) / 0.25);
    }

    // Faint scanline noise for camera effect
    float scanline = 0.92 + 0.08 * step(0.5, fract(gl_FragCoord.y * 0.5));

    fragColor = vec4(col * scanline, uOpacity);
}
