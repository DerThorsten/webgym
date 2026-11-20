// webgym: a soft blob for the liquid's field (GLLiquid): strongest in the middle,
// 0 at the circle's edge; added up (additive blending) in an offscreen texture

#version 330

in vec2 f_position;
in vec4 f_color;
in float f_thickness;

out vec4 fragColor;

void main()
{
    float f = max(0.0, 1.0 - dot(f_position, f_position));
    fragColor = vec4(f_color.rgb, f_color.a * f * f);
}
