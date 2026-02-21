#version 330 core

in vec3 vNormal;
in vec3 vViewPos;
in vec3 vVertexColor;

uniform vec3 uColor;
uniform float uOpacity;
uniform float uShininess;
uniform vec3 uAmbientColor;
uniform vec3 uLightDir;
uniform vec3 uLightColor;
uniform int uUseVertexColor;

// Point light uniforms
uniform int uHasPointLight;
uniform vec3 uPointLightPos;   // view space
uniform vec3 uPointLightColor;
uniform float uPointLightIntensity;
uniform float uPointLightRange;

out vec4 fragColor;

void main() {
    vec3 baseColor = uUseVertexColor != 0 ? vVertexColor : uColor;

    vec3 N = normalize(vNormal);
    vec3 V = normalize(-vViewPos);

    // --- Directional light ---
    vec3 L = normalize(uLightDir);
    float diff = max(dot(N, L), 0.0);
    vec3 R = reflect(-L, N);
    float spec = pow(max(dot(V, R), 0.0), uShininess);

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
