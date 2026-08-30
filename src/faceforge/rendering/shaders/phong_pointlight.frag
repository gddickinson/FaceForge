#version 330 core

#include "_common.glsl"
#include "_lighting.glsl"

uniform float uShininess;

// Point light uniforms
uniform int uHasPointLight;
uniform vec3 uPointLightPos;   // view space
uniform vec3 uPointLightColor;
uniform float uPointLightIntensity;
uniform float uPointLightRange;

// Clipping is done in default.vert via gl_ClipDistance[0]; see _common.glsl.

void main() {
    vec3 baseColor = ffBaseColor();

    vec3 N = ffNormal();
    vec3 V = ffViewDir();

    // --- Directional light ---
    vec3 L = ffLightDir();
    float diff = max(dot(N, L), 0.0);
    float spec = ffSpecular(N, V, L, uShininess);

    vec3 color = uAmbientColor * baseColor
               + diff * uLightColor * baseColor
               + spec * uLightColor * 0.3;

    // --- Point light (additive) ---
    if (uHasPointLight != 0) {
        vec3 toLight = uPointLightPos - vViewPos;
        float dist = length(toLight);
        vec3 Lp = toLight / max(dist, 0.001);

        // Distance attenuation: smooth falloff
        float atten = 1.0 / (1.0 + dist * dist / (uPointLightRange * uPointLightRange));
        atten *= uPointLightIntensity;

        // Diffuse
        float diffP = max(dot(N, Lp), 0.0);

        // Specular
        vec3 Rp = reflect(-Lp, N);
        float specP = pow(max(dot(V, Rp), 0.0), uShininess);

        color += atten * (diffP * uPointLightColor * baseColor
                        + specP * uPointLightColor * 0.3);
    }

    fragColor = vec4(color, uOpacity);
}
