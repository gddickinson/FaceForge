#version 330 core

in vec3 vNormal;
in vec3 vViewPos;
in vec3 vVertexColor;
in vec3 vWorldPos;

uniform vec3 uColor;
uniform float uOpacity;
uniform vec3 uAmbientColor;
uniform vec3 uLightDir;
uniform vec3 uLightColor;

// Clip plane
uniform int uClipEnabled;
uniform vec4 uClipPlane;

out vec4 fragColor;

void main() {
    if (uClipEnabled != 0 && dot(vWorldPos, uClipPlane.xyz) + uClipPlane.w < 0.0) discard;

    vec3 N = normalize(vNormal);
    vec3 V = normalize(-vViewPos);

    // Fresnel-like effect: more transparent when facing camera, visible at edges.
    // dot(N, V) ~ 1 when surface faces camera, ~ 0 at silhouette edges.
    float facing = abs(dot(N, V));

    // Invert so edges are bright and facing surfaces are faint
    float edgeFactor = 1.0 - facing;

    // Raise to a power for a sharper falloff
    float fresnel = pow(edgeFactor, 1.5);

    // Subtle directional light for depth cue
    vec3 L = normalize(uLightDir);
    float diff = max(dot(N, L), 0.0) * 0.3;

    vec3 color = uColor * (fresnel * 0.8 + diff + 0.1);

    // Opacity is strong at edges, faint when facing
    float alpha = uOpacity * (fresnel * 0.8 + 0.15);

    fragColor = vec4(color, alpha);
}
