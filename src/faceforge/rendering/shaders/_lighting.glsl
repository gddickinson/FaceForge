// Directional-light uniform block, shared by the 14 modes that read it.
// Include AFTER "_common.glsl".  wireframe.frag and points.frag do not
// include this file -- they declare no light uniforms today and must not start.
//
// Only uLightDir is read by a helper here.  uAmbientColor and uLightColor are
// declared but not used by any shared function on purpose: several modes
// declare them without using them, and a helper that touched them would turn
// those dead declarations into live uniforms and add a per-mesh upload to modes
// that have no use for the value.

uniform vec3 uAmbientColor;
uniform vec3 uLightDir;
uniform vec3 uLightColor;

/// Direction toward the directional light.
/// uLightDir arrives unit-length from LightSetup, but it is a plain mutable
/// attribute that nothing in the tree renormalises after mutation, so the
/// normalise stays rather than trusting the caller.
vec3 ffLightDir() { return normalize(uLightDir); }
