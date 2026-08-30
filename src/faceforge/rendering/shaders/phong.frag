#version 330 core

// Reference implementation of the plain directional-light Phong model, kept
// for comparison against phong_pointlight.frag (which is what SOLID and OPAQUE
// actually compile -- see renderer._compile_shaders).  Nothing loads this file.
//
// It had drifted: it declared neither the clip-plane uniforms nor vWorldPos, so
// reinstating it as the "simple Phong" path would have produced a shader that
// ignored the cutaway with no error at all, because set_uniform_* quietly
// no-ops on a -1 location.  Including the shared prelude is what stops that
// happening again -- the varying and uniform interface is now defined in one
// place for every mode.  Clipping itself needs nothing here: it moved to
// gl_ClipDistance[0] in default.vert.

#include "_common.glsl"
#include "_lighting.glsl"

uniform float uShininess;

void main() {
    vec3 baseColor = ffBaseColor();

    vec3 N = ffNormal();
    vec3 L = ffLightDir();

    // Diffuse (Lambertian)
    float diff = max(dot(N, L), 0.0);

    // Specular (Phong)
    vec3 V = ffViewDir();
    float spec = ffSpecular(N, V, L, uShininess);

    // Combine: ambient + diffuse + specular
    vec3 color = uAmbientColor * baseColor
               + diff * uLightColor * baseColor
               + spec * uLightColor * 0.3;

    fragColor = vec4(color, uOpacity);
}
