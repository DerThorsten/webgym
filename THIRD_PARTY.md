# Third-party work in webgym

## Environments after Gymnasium

[Gymnasium](https://github.com/Farama-Foundation/Gymnasium) (Copyright (c) 2016 OpenAI,
Copyright (c) 2022 Farama Foundation; [MIT license](https://github.com/Farama-Foundation/Gymnasium/blob/main/LICENSE))
is the model for webgym's API idea and for some of its environments. They are written
anew for box2d 3 / pyb2d3 (no Gymnasium code is copied), but follow the originals:

- `webgym/envs/cartpole.py`: Gymnasium's CartPole (the task, the limits, 1 reward per step).
- `webgym/envs/multi_cartpole.py`: CartPole with more poles.
- `webgym/envs/lunar_lander.py`: Gymnasium's LunarLander (the jagged surface with a pad
  between flags, the reward shaping: distance, speed, angle, legs; the fuel costs 0.3 /
  0.03; +100 / -100 for resting / crashing).
- `webgym/envs/walker.py`: in the spirit of Gymnasium's BipedalWalker.

## Code

- `webgym/gl/`: drawing code from the sandbox of [pyb2d3](https://github.com/DerThorsten/pyb2d3)
  (Copyright (c) 2024 Thorsten Beier, MIT license).

## Used (not part of this repository)

| | License |
|---|---|
| [Box2D](https://box2d.org) (Erin Catto), via [pyb2d3](https://github.com/DerThorsten/pyb2d3) | MIT |
| [pyjs](https://github.com/emscripten-forge/pyjs) | MPL-2.0 |
| [Monaco editor](https://microsoft.github.io/monaco-editor/) | MIT |
| [numpy](https://numpy.org), [pytorch](https://pytorch.org) | BSD-3-Clause |
| [PyOpenGL](https://pyopengl.sourceforge.net) | BSD-style |

The packages of the page come from [emscripten-forge](https://emscripten-forge.org), each
with its own license.
