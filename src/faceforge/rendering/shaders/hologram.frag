#version 330 core

// Sci-fi hologram / HUD projection effect.
// Cyan/blue wireframe glow with scanline flicker and edge bloom.

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

    // Fresnel: bright edges, transparent faces → holographic wireframe
    float facing = abs(dot(N, V));
    float fresnel = pow(1.0 - facing, 2.0);

    // Scanlines
    float scanline = 0.85 + 0.15 * step(0.5, fract(gl_FragCoord.y * 0.25));

    // Horizontal interference bands tied to world Z
    float interference = 0.90 + 0.10 * sin(vWorldPos.z * 3.0 + vWorldPos.y * 1.5);

    // Hologram colour: cyan core with blue-white edges
    vec3 coreColor = vec3(0.0, 0.85, 0.95);   // cyan
    vec3 edgeColor = vec3(0.4, 0.9, 1.0);     // bright cyan-white
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
