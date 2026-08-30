#version 330 core

// No _lighting.glsl include: this mode reads no light uniforms and must not
// start declaring any.

#include "_common.glsl"

void main() {
    fragColor = vec4(ffBaseColor(), uOpacity);
}
