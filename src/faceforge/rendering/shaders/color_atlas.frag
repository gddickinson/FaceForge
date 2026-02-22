#version 330 core

// Colour anatomical atlas — retains structure colours with ink-line contours
// and subtle cross-hatching.  Looks like a hand-coloured medical plate.

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

    // Diffuse
    float NdotL = dot(N, L);
    float diff = clamp(NdotL, 0.0, 1.0);
    float t = (NdotL + 1.0) * 0.5;

    // Warm/cool shift applied to the actual colour
    vec3 coolShift = baseColor * 0.55 + vec3(0.0, 0.0, 0.06);
    vec3 warmShift = baseColor * 1.05 + vec3(0.04, 0.02, 0.0);
    vec3 goochColor = mix(coolShift, warmShift, t);

    // Ink contour lines via Fresnel
    float facing = abs(dot(N, V));
    float edge = 1.0 - facing;
    float contour = smoothstep(0.0, 0.42, edge);
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
    vec3 R = reflect(-L, N);
    float spec = pow(max(dot(V, R), 0.0), max(uShininess, 25.0)) * 0.12;

    vec3 finalColor = goochColor * edgeDarken * hm + vec3(spec);
    finalColor = clamp(finalColor, 0.0, 1.0);

    fragColor = vec4(finalColor, uOpacity);
}
