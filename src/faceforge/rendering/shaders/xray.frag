#version 330 core

#include "_common.glsl"
#include "_lighting.glsl"

// NOTE: this mode's alpha is a fraction of uOpacity, so it renders as a dark
// solid unless GL_BLEND is on.  gl_material._MODE_NEEDS_BLENDING guarantees it.

void main() {
    vec3 N = ffNormal();
    vec3 V = ffViewDir();

    // Fresnel-like effect: more transparent when facing camera, visible at edges.
    // ffEdge() is 0 when the surface faces the camera, 1 at silhouette edges.
    float edgeFactor = ffEdge(N, V);
    float facing = 1.0 - edgeFactor;

    // Raise to a power for a sharper falloff
    float fresnel = pow(edgeFactor, 1.5);

    // Subtle directional light for depth cue
    vec3 L = ffLightDir();
    float diff = max(dot(N, L), 0.0) * 0.3;

    vec3 baseColor = ffBaseColor();
    vec3 color = baseColor * (fresnel * 0.8 + diff + 0.1);

    // Opacity is strong at edges, faint when facing
    float alpha = uOpacity * (fresnel * 0.8 + 0.15);

    fragColor = vec4(color, alpha);
}
