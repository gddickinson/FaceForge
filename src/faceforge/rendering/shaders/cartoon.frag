#version 330 core

// Cartoon / cel-shading.
// Quantised lighting bands with bold black outlines — comic book look.

#include "_common.glsl"
#include "_lighting.glsl"

uniform float uShininess;

void main() {
    vec3 N = ffNormal();
    vec3 V = ffViewDir();
    vec3 L = ffLightDir();

    vec3 baseColor = ffBaseColor();

    // Quantise diffuse into 4 bands
    float NdotL = dot(N, L);
    float diff = clamp(NdotL, 0.0, 1.0);

    float band;
    if (diff > 0.85)      band = 1.0;
    else if (diff > 0.55) band = 0.75;
    else if (diff > 0.25) band = 0.50;
    else                  band = 0.30;

    // Boost colour saturation for cartoon pop
    float lum = ffLuma(baseColor);
    vec3 saturated = mix(vec3(lum), baseColor, 1.40);
    saturated = clamp(saturated, 0.0, 1.0);

    vec3 color = saturated * band;

    // Specular highlight (sharp cartoon glint)
    float spec = ffSpecular(N, V, L, 80.0);
    float specBand = step(0.60, spec);  // hard cutoff
    color += vec3(specBand * 0.45);

    // Bold black outline
    float outline = smoothstep(0.0, 0.30, 1.0 - ffEdge(N, V));
    color *= outline;  // goes to black at silhouettes

    fragColor = vec4(clamp(color, 0.0, 1.0), uOpacity);
}
