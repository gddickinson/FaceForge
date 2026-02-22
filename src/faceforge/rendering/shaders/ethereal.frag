#version 330 core

// Ethereal / fantasy glow rendering.
// Soft luminous aura with iridescent colour shifts and bloom edges.

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

    // Iridescent colour shift based on view angle
    float facing = abs(dot(N, V));
    float edge = 1.0 - facing;

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
