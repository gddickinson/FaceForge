#version 330 core

// Colour anatomical atlas — retains structure colours with ink-line contours
// and subtle cross-hatching.  Looks like a hand-coloured medical plate.

#include "_common.glsl"
#include "_lighting.glsl"

uniform float uShininess;

void main() {
    vec3 N = ffNormal();
    vec3 V = ffViewDir();
    vec3 L = ffLightDir();

    vec3 baseColor = ffBaseColor();

    // Diffuse
    float NdotL = dot(N, L);
    float diff = clamp(NdotL, 0.0, 1.0);
    float t = (NdotL + 1.0) * 0.5;

    // Warm/cool shift applied to the actual colour
    vec3 coolShift = baseColor * 0.55 + vec3(0.0, 0.0, 0.06);
    vec3 warmShift = baseColor * 1.05 + vec3(0.04, 0.02, 0.0);
    vec3 goochColor = mix(coolShift, warmShift, t);

    // Ink contour lines via Fresnel
    float contour = smoothstep(0.0, 0.42, ffEdge(N, V));
    float edgeDarken = 1.0 - contour * 0.75;

    // Light cross-hatching in shadow only
    vec2 sc = gl_FragCoord.xy;
    float hatch = step(1.5, mod(sc.x - sc.y, 7.0));
    float shadow = 1.0 - diff;
    float hm = 1.0;
    if (shadow > 0.40) {
        hm *= mix(1.0, hatch, smoothstep(0.40, 0.70, shadow) * 0.30);
    }

    // Specular highlight
    float spec = ffSpecular(N, V, L, max(uShininess, 25.0)) * 0.12;

    vec3 finalColor = goochColor * edgeDarken * hm + vec3(spec);
    finalColor = clamp(finalColor, 0.0, 1.0);

    fragColor = vec4(finalColor, uOpacity);
}
