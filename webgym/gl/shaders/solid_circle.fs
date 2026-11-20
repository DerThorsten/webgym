// SPDX-FileCopyrightText: 2024 Erin Catto
// SPDX-License-Identifier: MIT

#version 330

in vec2 f_position;
in vec4 f_color;
in float f_thickness;

out vec4 fragColor;

// https://en.wikipedia.org/wiki/Alpha_compositing
vec4 blend_colors(vec4 front, vec4 back)
{
    vec3 cSrc = front.rgb;
    float alphaSrc = front.a;
    vec3 cDst = back.rgb;
    float alphaDst = back.a;

    vec3 cOut = cSrc * alphaSrc + cDst * alphaDst * (1.0 - alphaSrc);
    float alphaOut = alphaSrc + alphaDst * (1.0 - alphaSrc);
    cOut = cOut / alphaOut;

    return vec4(cOut, alphaOut);
}

void main()
{
    // radius in unit quad
    float radius = 1.0;

    // distance to the circle (webgym: the outline only, no radius line)
    vec2 w = f_position;
    float dw = length(w);
    float d = abs(dw - radius);

    // webgym: a solid fill in the shape's color, a darker outline (both with its alpha)
    vec4 fillColor = f_color;
    vec4 borderColor = vec4(0.55 * f_color.rgb, f_color.a);

    // roll the fill alpha down at the border
    vec4 back = vec4(fillColor.rgb, fillColor.a * smoothstep(radius + f_thickness, radius, dw));

    // roll the border alpha down from 1 to 0 across the border thickness
    vec4 front = vec4(borderColor.rgb, borderColor.a * smoothstep(f_thickness, 0.0f, d));

    fragColor = blend_colors(front, back);
}
