#version 330 core

// Pure pen-and-ink rendering.
// White fill with stark black outlines and stipple shading — no colour.

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

    float NdotL = dot(N, L);
    float diff = clamp(NdotL, 0.0, 1.0);

    // Very strong edge detection — thick outlines
    float facing = abs(dot(N, V));
    float edge = 1.0 - facing;
    float outline = smoothstep(0.0, 0.55, edge);
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
