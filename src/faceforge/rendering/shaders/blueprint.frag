#version 330 core

// Blueprint / technical drawing.
// White lines on deep blue background with grid overlay.

#include "_common.glsl"
#include "_lighting.glsl"

uniform float uShininess;

// NOTE: this mode's alpha is a fraction of uOpacity, so it renders as a dark
// solid unless GL_BLEND is on.  gl_material._MODE_NEEDS_BLENDING guarantees it.
// It also reads vWorldPos for the grid, so the renderer keeps uploading
// uModelMatrix for this mode (renderer._NEEDS_WORLD_POS).

void main() {
    vec3 N = ffNormal();
    vec3 V = ffViewDir();
    vec3 L = ffLightDir();

    // Edge detection — Fresnel
    float edge = ffEdge(N, V);
    float wireframe = smoothstep(0.0, 0.50, edge);

    // Subtle grid from world position
    float gridSize = 5.0;
    vec2 gridUV = fract(vWorldPos.xz / gridSize);
    float gridLine = 1.0 - step(0.04, min(min(gridUV.x, 1.0 - gridUV.x),
                                           min(gridUV.y, 1.0 - gridUV.y)));
    // Fainter grid
    gridLine *= 0.12;

    // Faint surface shading for depth cue
    float NdotL = dot(N, L);
    float depthShade = clamp((NdotL + 1.0) * 0.5, 0.0, 1.0) * 0.08;

    // Compose: white lines on blue
    float lineIntensity = wireframe * 0.90 + gridLine + depthShade;
    lineIntensity = clamp(lineIntensity, 0.0, 1.0);

    // Blueprint blue background → white lines
    // When vertex colors active, use vertex color for lines
    vec3 bgBlue = vec3(0.05, 0.12, 0.28);
    vec3 lineWhite = uUseVertexColor != 0 ? vVertexColor : vec3(0.85, 0.90, 1.0);

    vec3 finalColor = mix(bgBlue, lineWhite, lineIntensity);

    // Alpha: mostly opaque at edges, semitransparent at faces
    float alpha = uOpacity * (wireframe * 0.80 + 0.15 + gridLine);
    alpha = clamp(alpha, 0.0, 1.0);

    fragColor = vec4(finalColor, alpha);
}
