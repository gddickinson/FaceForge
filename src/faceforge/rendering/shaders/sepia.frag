#version 330 core

// Vintage sepia textbook illustration.
// Warm brown tones with ink-line contours — aged anatomical atlas feel.

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

    // Gooch-style tonal mapping
    float t = (NdotL + 1.0) * 0.5;
    float tone = mix(0.20, 0.88, t);

    // Contour edges
    float facing = abs(dot(N, V));
    float edge = 1.0 - facing;
    float contour = smoothstep(0.0, 0.40, edge);
    float edgeDarken = 1.0 - contour * 0.80;

    // Cross-hatching
    vec2 sc = gl_FragCoord.xy;
    float hatch1 = step(1.2, mod(sc.x - sc.y, 5.0));
    float hatch2 = step(1.2, mod(sc.x + sc.y, 5.0));
    float shadow = 1.0 - diff;
    float hm = 1.0;
    if (shadow > 0.30) hm *= mix(1.0, hatch1, smoothstep(0.30, 0.60, shadow) * 0.40);
    if (shadow > 0.60) hm *= mix(1.0, hatch2, smoothstep(0.60, 0.85, shadow) * 0.30);

    // Material tint
    vec3 bc = uUseVertexColor != 0 ? vVertexColor : uColor;
    float lum = dot(bc, vec3(0.299, 0.587, 0.114));
    float mt = mix(0.85, 1.0, lum);

    float grey = tone * edgeDarken * hm * mt;
    grey = clamp(grey, 0.0, 1.0);

    // Sepia colour ramp:  dark brown → parchment cream
    vec3 dark  = vec3(0.20, 0.12, 0.06);
    vec3 light = vec3(0.95, 0.88, 0.75);
    vec3 finalColor = mix(dark, light, grey);

    fragColor = vec4(finalColor, uOpacity);
}
