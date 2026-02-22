#version 330 core

// Porcelain / doll-like rendering.
// Smooth, pale, subsurface-scatter approximation with soft light wrapping.
// Evokes a porcelain figurine or anatomical doll.

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

    // Desaturate and lighten toward porcelain white
    float lum = dot(baseColor, vec3(0.299, 0.587, 0.114));
    vec3 porcelainBase = mix(vec3(lum), baseColor, 0.35);  // mostly grey
    porcelainBase = mix(porcelainBase, vec3(0.92, 0.90, 0.88), 0.55);  // pull toward white

    // Wrapped diffuse (light bleeds around edges for SSS feel)
    float wrap = 0.45;
    float NdotL = dot(N, L);
    float wrapDiff = clamp((NdotL + wrap) / (1.0 + wrap), 0.0, 1.0);

    // Subsurface scatter colour: warm reddish in thin areas
    float scatter = clamp(1.0 - wrapDiff, 0.0, 1.0);
    vec3 sssColor = vec3(0.90, 0.55, 0.45) * scatter * 0.18;

    // Very smooth specular (glazed surface)
    vec3 halfVec = normalize(L + V);
    float specAngle = max(dot(N, halfVec), 0.0);
    float spec = pow(specAngle, 120.0) * 0.55;

    // Soft rim light
    float rim = 1.0 - max(dot(N, V), 0.0);
    rim = pow(rim, 2.5) * 0.12;

    // High ambient (porcelain in a lightbox)
    vec3 ambient = porcelainBase * 0.50;

    vec3 color = ambient
               + porcelainBase * wrapDiff * 0.65
               + sssColor
               + vec3(spec)
               + vec3(rim);

    fragColor = vec4(clamp(color, 0.0, 1.0), uOpacity);
}
