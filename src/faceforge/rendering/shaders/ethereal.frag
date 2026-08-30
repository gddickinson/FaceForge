#version 330 core

// Ethereal / fantasy glow rendering.
// Soft luminous aura with iridescent colour shifts and bloom edges.

#include "_common.glsl"
#include "_lighting.glsl"

uniform float uShininess;

// NOTE: this mode's alpha is a fraction of uOpacity, so it renders as a dark
// solid unless GL_BLEND is on.  gl_material._MODE_NEEDS_BLENDING guarantees it.

void main() {
    vec3 N = ffNormal();
    vec3 V = ffViewDir();
    vec3 L = ffLightDir();

    vec3 baseColor = ffBaseColor();

    // Iridescent colour shift based on view angle
    float edge = ffEdge(N, V);
    float facing = 1.0 - edge;

    // Hue shift: rotate through pastel spectrum at edges
    float hueAngle = edge * 2.5 + dot(N, vec3(0.3, 0.6, 0.7)) * 1.5;
    vec3 iridescent = vec3(
        0.5 + 0.5 * cos(hueAngle),
        0.5 + 0.5 * cos(hueAngle + 2.094),    // +120°
        0.5 + 0.5 * cos(hueAngle + 4.189)     // +240°
    );

    // Blend base colour with iridescence at edges
    vec3 color = mix(baseColor * 0.6, iridescent, edge * 0.7);

    // Soft diffuse
    float NdotL = dot(N, L);
    float diff = clamp((NdotL + 0.5) / 1.5, 0.0, 1.0);  // wrapped

    // Glow: brighter at edges (bloom aura)
    float glow = pow(edge, 1.8) * 0.65;

    // Specular shimmer
    vec3 R = reflect(-L, N);
    float spec = pow(max(dot(V, R), 0.0), 60.0) * 0.35;

    // Inner luminosity (backlit effect)
    float backlight = clamp(-NdotL * 0.3, 0.0, 1.0);

    vec3 finalColor = color * diff * 0.7
                    + color * glow
                    + iridescent * spec
                    + baseColor * backlight * 0.15
                    + vec3(0.05, 0.03, 0.08);  // ambient magic

    // Soft alpha: semi-transparent with bright edges
    float alpha = uOpacity * (facing * 0.5 + glow + 0.15);
    alpha = clamp(alpha, 0.0, 1.0);

    fragColor = vec4(clamp(finalColor, 0.0, 1.0), alpha);
}
