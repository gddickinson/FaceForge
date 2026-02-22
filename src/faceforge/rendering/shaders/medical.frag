#version 330 core

// Clean medical atlas rendering.
// High-contrast, saturated colours with sharp directional lighting.
// Emulates modern full-colour medical textbook plates (Netter-style).

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

    // Boost saturation slightly
    float lum = dot(baseColor, vec3(0.299, 0.587, 0.114));
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
    vec3 R = reflect(-L, N);
    float spec = pow(max(dot(V, R), 0.0), max(uShininess, 40.0));

    // Rim light for edge definition
    float rim = 1.0 - max(dot(N, V), 0.0);
    rim = pow(rim, 3.0) * 0.15;

    vec3 color = saturated * (uAmbientColor * ao + diff * uLightColor + fillDiff * vec3(0.6, 0.65, 0.7))
               + spec * uLightColor * 0.25
               + vec3(rim);

    // Very subtle dark edge outline
    float facing = abs(dot(N, V));
    float edgeFade = smoothstep(0.0, 0.20, facing);
    color *= edgeFade * 0.3 + 0.7;

    fragColor = vec4(clamp(color, 0.0, 1.0), uOpacity);
}
