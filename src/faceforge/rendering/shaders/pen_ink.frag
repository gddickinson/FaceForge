#version 330 core

// Pure pen-and-ink rendering.
// White fill with stark black outlines and stipple shading — no colour.

#include "_common.glsl"
#include "_lighting.glsl"

uniform float uShininess;

void main() {
    vec3 N = ffNormal();
    vec3 V = ffViewDir();
    vec3 L = ffLightDir();

    float NdotL = dot(N, L);
    float diff = clamp(NdotL, 0.0, 1.0);

    // Very strong edge detection — thick outlines
    float outline = smoothstep(0.0, 0.55, ffEdge(N, V));
    // Hard black outline
    float ink = 1.0 - outline * 0.95;

    // Stipple / dot pattern for shadow areas (like pointillism)
    vec2 sc = gl_FragCoord.xy;
    float shadow = 1.0 - diff;

    // Grid-based stipple: more dots in darker areas
    float dotSpacing = 4.0;
    vec2 cell = mod(sc, dotSpacing);
    float dotDist = length(cell - dotSpacing * 0.5);

    // Dot radius scales with shadow depth
    float dotRadius = shadow * 1.8;
    float stipple = smoothstep(dotRadius, dotRadius + 0.5, dotDist);

    // Fine line hatching on top of stipple for deep shadow
    float hatch = step(0.8, mod(sc.x - sc.y, 3.0));
    float deepHatch = 1.0;
    if (shadow > 0.60) {
        deepHatch = mix(1.0, hatch, smoothstep(0.60, 0.90, shadow) * 0.50);
    }

    float brightness = stipple * deepHatch * ink;
    brightness = clamp(brightness, 0.0, 1.0);

    // Pure black and white
    vec3 finalColor = vec3(brightness);

    fragColor = vec4(finalColor, uOpacity);
}
