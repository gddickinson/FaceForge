#version 330 core

// Cartoon / cel-shading.
// Quantised lighting bands with bold black outlines — comic book look.

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

    vec3 baseColor = uUseVertexColor != 0 ? vVertexColor : uColor;

    // Quantise diffuse into 4 bands
    float NdotL = dot(N, L);
    float diff = clamp(NdotL, 0.0, 1.0);

    float band;
    if (diff > 0.85)      band = 1.0;
    else if (diff > 0.55) band = 0.75;
    else if (diff > 0.25) band = 0.50;
    else                  band = 0.30;

    // Boost colour saturation for cartoon pop
    float lum = dot(baseColor, vec3(0.299, 0.587, 0.114));
    vec3 saturated = mix(vec3(lum), baseColor, 1.40);
    saturated = clamp(saturated, 0.0, 1.0);

    vec3 color = saturated * band;

    // Specular highlight (sharp cartoon glint)
    vec3 R = reflect(-L, N);
    float spec = pow(max(dot(V, R), 0.0), 80.0);
    float specBand = step(0.60, spec);  // hard cutoff
    color += vec3(specBand * 0.45);

    // Bold black outline
    float facing = abs(dot(N, V));
    float outline = smoothstep(0.0, 0.30, facing);
    color *= outline;  // goes to black at silhouettes

    fragColor = vec4(clamp(color, 0.0, 1.0), uOpacity);
}
