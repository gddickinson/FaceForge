#version 330 core

// Clean medical atlas rendering.
// High-contrast, saturated colours with sharp directional lighting.
// Emulates modern full-colour medical textbook plates (Netter-style).

#include "_common.glsl"
#include "_lighting.glsl"

uniform float uShininess;

void main() {
    vec3 N = ffNormal();
    vec3 V = ffViewDir();
    vec3 L = ffLightDir();

    vec3 baseColor = ffBaseColor();

    // Boost saturation slightly
    float lum = ffLuma(baseColor);
    vec3 saturated = mix(vec3(lum), baseColor, 1.25);
    saturated = clamp(saturated, 0.0, 1.0);

    // Strong directional lighting
    float NdotL = dot(N, L);
    float diff = clamp(NdotL, 0.0, 1.0);

    // Fill light from opposite side (half-intensity)
    vec3 fillDir = normalize(vec3(-uLightDir.x, -uLightDir.y, uLightDir.z));
    float fillDiff = clamp(dot(N, fillDir), 0.0, 1.0) * 0.35;

    // Ambient occlusion approximation
    float ao = 0.45 + 0.55 * diff;

    // Specular highlight (wet tissue look)
    float spec = ffSpecular(N, V, L, max(uShininess, 40.0));

    // Rim light for edge definition
    float rim = 1.0 - max(dot(N, V), 0.0);
    rim = pow(rim, 3.0) * 0.15;

    vec3 color = saturated * (uAmbientColor * ao + diff * uLightColor + fillDiff * vec3(0.6, 0.65, 0.7))
               + spec * uLightColor * 0.25
               + vec3(rim);

    // Very subtle dark edge outline
    float edgeFade = smoothstep(0.0, 0.20, 1.0 - ffEdge(N, V));
    color *= edgeFade * 0.3 + 0.7;

    fragColor = vec4(clamp(color, 0.0, 1.0), uOpacity);
}
