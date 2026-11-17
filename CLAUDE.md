we want something like https://gymnasium.farama.org/ https://github.com/Farama-Foundation/Gymnasium
but it should work **in the browser** via emscripten-forge / pyjs

There is a handover from two important sessions that worked on important projects related to this prohect

* RL_GYM_HANDOFF.md 
    is a document from another session that focused on running pyb2d3 (our python bindings for box2dm, that also run iun wasm0)

* HANDOVER_browser_gym.md
    is a handover from a session with knowledge about pyjs, recipes and pyb2d3


Read them carefully.





Goals:
    * the focus here is having something impressive in the browser.
    * it should a) showcase the impressive things we can do with emscripten-forge
    * but also be fun
    * therefore we focus on box2d environments (at the moment)
    * it should also work outside the browser st. we can iterate on the native version

    * we should focus on things that can be trained very fast, st. we can show a "live view" of 
    the current state, ie how is the agent currently doing, ie we see live what the agent is doing to the world

Things we need to justify:
    * we somewhat need to justify running python in the browser,so the python code should be editable in the browser.
    * we should add an option to upload python code for the environments (the thing we want to optimize/controll/learn)
      and also the code for solving /training these models.
    * this upload could be done with nice file-system-apis that open files from a machine


Technical considerations:
    * what runs in main thread / what runs in a worker.
    * if we can use pyopengl also on an offscreen canvas, it would be ideal to run box2d / the environments
    * if we want to allow the human to interact with the world in the offscreen canvas, we need to "pipe" event into the workers.
    * should the optimizer and the model live in the same worker or each in differnt works and communicate via messages?
    * what algorithms are fast enough
    * how can we make loading the page fast / how can we cache things optimal