#version 330 core

in vec3 vNormal;
in vec3 vViewPos;

uniform vec3 uColor;
uniform float uOpacity;
uniform float uShininess;
uniform vec3 uAmbientColor;
uniform vec3 uLightDir;
uniform vec3 uLightColor;

out vec4 fragColor;

void main() {
    vec3 N = normalize(vNormal);
    vec3 L = normalize(uLightDir);

    // Diffuse (Lambertian)
    float diff = max(dot(N, L), 0.0);

    // Specular (Phong)
    vec3 V = normalize(-vViewPos);
    vec3 R = reflect(-L, N);
    float spec = pow(max(dot(V, R), 0.0), uShininess);

    // Combine: ambient + diffuse + specular
    vec3 color = uAmbientColor * uColor
               + diff * uLightColor * uColor
               + spec * uLightColor * 0.3;

    fragColor = vec4(color, uOpacity);
}
