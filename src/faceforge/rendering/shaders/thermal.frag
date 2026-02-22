#version 330 core

// Thermal / infrared imaging.
// Maps surface angle and position to a heat-map colour ramp.

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

    // Use facing angle + diffuse as "heat" proxy
    float facing = abs(dot(N, V));
    float NdotL = dot(N, L);
    float diff = clamp(NdotL, 0.0, 1.0);

    // Heat value: combination of facing (0=edge=cold) and light (warm)
    float heat = facing * 0.6 + diff * 0.4;

    // Use material colour luminance to vary heat across structures
    vec3 bc = uUseVertexColor != 0 ? vVertexColor : uColor;
    float lum = dot(bc, vec3(0.299, 0.587, 0.114));
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
