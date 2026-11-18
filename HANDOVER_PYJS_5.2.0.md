# Handover: pyjs 5.2.0 (bytecode cache saves on demand)

Everything in `HANDOVER_PYJS_5.1.0.md` still holds, except how the bytecode
cache saves. This is the only change of 5.2.0 that matters for webgym.

## What changed

- **5.0 / 5.1:** pyjs saved new `.pyc` files by itself, from a 3 s
  `setInterval` in every runtime (workers included). It listed every
  `__pycache__` directory of every package (pytorch: thousands) for as long as
  the runtime lived. That's recurring work during training and rendering. It's gone.
- **5.2:** saving is up to the app:
  - `await pyjs.saveBytecode()` saves what python wrote since the last save
    (packages, the python environment, archives with `cache: true`);
  - the only automatic save is best effort, when the page becomes hidden, and
    **only in the main thread**. Workers have no such event, so in a worker
    nothing is saved unless it calls `saveBytecode()`.
  - Restoring is unchanged: automatic, with the package / cached archive.

## What webgym should do

webgym already calls `saveBytecode()` after the torch import. Make it the
rule in **every worker**, after each group of heavy imports, and await it:

```js
pyjs.exec("import torch");               // (or jax, iree, ...)
await pyjs.saveBytecode();
```

- One call per import group is enough. It is cheap when nothing is new: one
  directory listing per `__pycache__` dir, no writes.
- The display, trainer and compile workers each save their own imports. The
  cache is shared per origin, so the next visit of any worker benefits.
- Imports that happen later (a new environment, a user's install) need
  another `saveBytecode()` afterwards. Otherwise their bytecode is compiled
  again on every visit.
- Keep `sys.dont_write_bytecode` off (see the 5.1.0 handover): with it, python
  writes no `.pyc` and there's nothing to save.

## Version

5.2.0 once it's on npm (PR on emscripten-forge/pyjs, branch
`bytecode-save-on-demand` of DerThorsten/pyjs). With 5.0.0 / 5.1.0 the explicit
calls are already right; the upgrade only removes the background interval.
