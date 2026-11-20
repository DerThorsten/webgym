// SPDX-FileCopyrightText: 2024 Erin Catto
// SPDX-License-Identifier: MIT

#version 330

out vec4 FragColor;

uniform float time;
uniform vec2 resolution;
uniform vec3 baseColor;

// A simple pseudo-random function
float random(vec2 st)
{
    return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
}

void main()
{
    vec2 uv = gl_FragCoord.xy / resolution.xy;

    // webgym: a soft sky, a little lighter towards the bottom, a slight blue
    vec3 top = baseColor * vec3(0.62, 0.68, 0.82);
    vec3 bottom = baseColor * vec3(1.05, 1.08, 1.15);
    vec3 color = mix(bottom, top, uv.y);

    FragColor = vec4(color, 1.0);
}
