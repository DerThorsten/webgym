// webgym: the liquid (GLLiquid) from the field of soft blobs (metaballs): where the
// field passes the threshold there is water, a lighter rim at its surface

#version 330

uniform sampler2D field;
uniform vec2 resolution;
uniform vec3 waterColor;

out vec4 fragColor;

void main()
{
    float f = texture(field, gl_FragCoord.xy / resolution).a;
    float threshold = 0.4;
    float inside = smoothstep(threshold - 0.05, threshold + 0.05, f);
    // (near the surface: lighter, deeper inside: the water's color)
    float rim = 1.0 - smoothstep(threshold, threshold + 0.4, f);
    vec3 color = mix(waterColor, vec3(0.85, 0.95, 1.0), 0.55 * rim);
    fragColor = vec4(color, 0.85 * inside);
}
