#version 330 core

// Sci-fi hologram / HUD projection effect.
// Cyan/blue wireframe glow with scanline flicker and edge bloom.

#include "_common.glsl"
#include "_lighting.glsl"

uniform float uShininess;

// NOTE: this mode's alpha is a fraction of uOpacity, so it renders as a dark
// solid unless GL_BLEND is on.  gl_material._MODE_NEEDS_BLENDING guarantees it.
// It also reads vWorldPos for the interference bands, so the renderer keeps
// uploading uModelMatrix for this mode (renderer._NEEDS_WORLD_POS).

void main() {
    vec3 N = ffNormal();
    vec3 V = ffViewDir();

    // Fresnel: bright edges, transparent faces → holographic wireframe
    float edge = ffEdge(N, V);
    float facing = 1.0 - edge;
    float fresnel = pow(edge, 2.0);

    // Scanlines
    float scanline = 0.85 + 0.15 * step(0.5, fract(gl_FragCoord.y * 0.25));

    // Horizontal interference bands tied to world Z
    float interference = 0.90 + 0.10 * sin(vWorldPos.z * 3.0 + vWorldPos.y * 1.5);

    // Hologram colour: cyan core with blue-white edges
    // When vertex colors active, tint hologram with the vertex color
    vec3 coreColor = vec3(0.0, 0.85, 0.95);   // cyan
    vec3 edgeColor = vec3(0.4, 0.9, 1.0);     // bright cyan-white
    if (uUseVertexColor != 0) {
        coreColor = vVertexColor * 0.9;
        edgeColor = mix(vVertexColor, vec3(1.0), 0.4);
    }
    vec3 holoColor = mix(coreColor, edgeColor, fresnel);

    // Glow intensity
    float glow = fresnel * 0.85 + 0.08;
    glow *= scanline * interference;

    // Very faint interior fill for depth reading
    float interior = facing * 0.06;

    float alpha = uOpacity * (glow + interior);
    alpha = clamp(alpha, 0.0, 1.0);

    // Bloom: boost brightness at edges
    vec3 bloom = holoColor * glow * 1.3;
    vec3 finalColor = bloom + holoColor * interior;
    finalColor = clamp(finalColor, 0.0, 1.0);

    fragColor = vec4(finalColor, alpha);
}
