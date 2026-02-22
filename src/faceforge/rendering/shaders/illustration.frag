#version 330 core

// Black-and-white textbook illustration shader.
//
// Combines several techniques to mimic ink-on-paper anatomical plates:
//   1. Gooch cool/warm tone mapping compressed to greyscale
//   2. Fresnel edge darkening for strong silhouette contours
//   3. Screen-space cross-hatching in shadowed regions
//   4. Rim highlight on light-facing edges for depth

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

// Clip plane
uniform int uClipEnabled;
uniform vec4 uClipPlane;

out vec4 fragColor;

void main() {
    if (uClipEnabled != 0 && dot(vWorldPos, uClipPlane.xyz) + uClipPlane.w < 0.0) discard;

    vec3 N = normalize(vNormal);
    vec3 V = normalize(-vViewPos);
    vec3 L = normalize(uLightDir);

    // ── 1. Diffuse (Lambertian) ──
    float NdotL = dot(N, L);
    float diff = clamp(NdotL, 0.0, 1.0);

    // ── 2. Gooch cool-warm → greyscale ──
    // Map NdotL from [-1,1] to a tonal range.
    // "Cool" (shadow) = dark grey,  "Warm" (lit) = near white.
    float t = (NdotL + 1.0) * 0.5;  // [0, 1]
    float coolTone = 0.25;    // darkest shadow value
    float warmTone = 0.92;    // brightest lit value
    float gooch = mix(coolTone, warmTone, t);

    // ── 3. Fresnel edge darkening (ink contour) ──
    float facing = abs(dot(N, V));
    float edge = 1.0 - facing;
    float contour = smoothstep(0.0, 0.45, edge);
    // Strong, dark contour lines at silhouette
    float edgeDarken = 1.0 - contour * 0.85;

    // ── 4. Screen-space cross-hatching ──
    // Use fragment screen coordinates for a repeating hatch pattern.
    // Only visible in shadowed regions.
    vec2 screenUV = gl_FragCoord.xy;

    // Primary hatch (diagonal /)
    float hatch1 = mod(screenUV.x - screenUV.y, 6.0);
    hatch1 = step(1.5, hatch1);  // thin lines every 6 pixels

    // Secondary hatch (diagonal \) — only in deep shadow
    float hatch2 = mod(screenUV.x + screenUV.y, 6.0);
    hatch2 = step(1.5, hatch2);

    // Blend hatching based on shadow depth
    float shadowFactor = 1.0 - diff;  // 0 = fully lit, 1 = fully shadowed
    float hatchMask = 1.0;
    // Single-direction hatch in moderate shadow
    if (shadowFactor > 0.35) {
        hatchMask *= mix(1.0, hatch1, smoothstep(0.35, 0.65, shadowFactor) * 0.45);
    }
    // Cross-hatch in deep shadow
    if (shadowFactor > 0.65) {
        hatchMask *= mix(1.0, hatch2, smoothstep(0.65, 0.9, shadowFactor) * 0.35);
    }

    // ── 5. Specular highlight (subtle rim light) ──
    vec3 R = reflect(-L, N);
    float spec = pow(max(dot(V, R), 0.0), max(uShininess, 20.0));
    float highlight = spec * 0.15;

    // ── 6. Compose final greyscale value ──
    // Use base material lightness to vary structure tones slightly
    vec3 baseColor = uUseVertexColor != 0 ? vVertexColor : uColor;
    float baseLum = dot(baseColor, vec3(0.299, 0.587, 0.114));
    // Subtle tone variation from material (keeps structures distinguishable)
    float materialTint = mix(0.85, 1.0, baseLum);

    float grey = gooch * edgeDarken * hatchMask * materialTint + highlight;
    grey = clamp(grey, 0.0, 1.0);

    // Paper-white background tint: shift slightly warm (sepia hint)
    vec3 paperWhite = vec3(0.98, 0.96, 0.93);
    vec3 inkBlack = vec3(0.06, 0.06, 0.08);
    vec3 finalColor = mix(inkBlack, paperWhite, grey);

    fragColor = vec4(finalColor, uOpacity);
}
