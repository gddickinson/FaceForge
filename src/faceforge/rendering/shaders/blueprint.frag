#version 330 core

// Blueprint / technical drawing.
// White lines on deep blue background with grid overlay.

in vec3 vNormal;
in vec3 vViewPos;
in vec3 vVertexColor;
in vec3 vWorldPos;

uniform vec3 uColor;
uniform float uOpacity;
uniform float uShininess;
uniform vec3 uAmbientColor;
uniform vec3 uLightDir;
uniform vec3 uLightColor;
uniform int uUseVertexColor;

uniform int uClipEnabled;
uniform vec4 uClipPlane;

out vec4 fragColor;

void main() {
    if (uClipEnabled != 0 && dot(vWorldPos, uClipPlane.xyz) + uClipPlane.w < 0.0) discard;

    vec3 N = normalize(vNormal);
    vec3 V = normalize(-vViewPos);
    vec3 L = normalize(uLightDir);

    // Edge detection — Fresnel
    float facing = abs(dot(N, V));
    float edge = 1.0 - facing;
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
