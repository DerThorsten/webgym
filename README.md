# webgym

Gymnasium-like box2d ([pyb2d3](https://github.com/DerThorsten/pyb2d3)) environments with
live training, natively and **in the browser** (Python in WebAssembly via pyjs +
emscripten-forge). Watch a population of agents learn, drag the env around with the mouse.

```bash
pixi run test                      # python tests
pixi run python -m webgym.cli      # train CartPole headless
pixi run web-dev                   # the page with the vite dev server
pixi run web-build                 # static page in web/dist
pixi run -e web-test web-test      # the page in headless chromium
```

See [CLAUDE.md](CLAUDE.md) for how it is built (envs, algorithms, the browser setup).
