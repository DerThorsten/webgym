# Env ideas

These are design notes, not implemented. The goal is something fun to watch while it learns
live in the browser, and fast to train.

## 1. Car designer ("evolve a car for a dynamic parcours")

The thing being optimized is the **car itself**, not a policy. A candidate is a genome:
- the chassis shape,
- the wheels,
- the motor,
- the suspension.

The car drives with a constant motor speed, so it has no controller. Whether it reaches the
end of the parcours is purely a question of design.

### Genome (~30 floats, every gene in [0, 1] and scaled when building)

| genes | meaning | range |
|---|---|---|
| 8 × radius | chassis vertex `k` sits at angle `k·45°` with this radius | 0.2 – 1.5 m |
| 8 × wheel_on | slot k gets a wheel if > 0.5 (wheels at the chassis vertices); at most 4 count, the strongest first | – |
| 8 × wheel_radius | radius of the wheel at slot k | 0.15 – 1.0 m |
| 1 × chassis density | heavier: more traction, slower on climbs | 0.5 – 5 |
| 1 × wheel density | | 0.5 – 5 |
| 1 × motor torque | max_motor_torque of all wheels (or per wheel: +8 genes) | 10 – 200 N·m |
| 1 × motor speed | target speed (rad/s) | 3 – 20 |
| 1 × spring hz | wheel joint spring (suspension) | 2 – 15 |
| 1 × damping ratio | | 0.1 – 1.0 |

### Building a candidate in pyb2d3

- **Chassis:**
  - Box2d polygons are convex with 3–8 points. If we put all 8 vertices into one
    `b2d.polygon(points=…)`, it builds the convex hull with `compute_hull` and silently
    removes every concave dent.
  - Instead, make **8 triangles** `(0,0), p_k, p_{k+1}` as shapes on one body
    (`self.body(position, triangle_0)`, then `self.shape(body, triangle_k)`). Any star-shaped car becomes possible, as in
    BoxCar2D.
  - A triangle with a tiny area makes `compute_hull` return `count == 0`. Clamp each radius
    to at least 0.2 m and skip degenerate triangles.
- **Wheels:**
  - Each wheel is a circle body joined by `b2d.wheel_joint_def(...)` →
    `world.create_wheel_joint`, with the axis along the local y axis.
  - The joint takes the motor and spring settings:
    - `enable_motor`, `max_motor_torque`, `motor_speed`;
    - `enable_spring`, `hertz`, `damping_ratio`;
    - a limit to keep the suspension travel small.
  - The `WheelJoint` bindings (motor, spring, limit) exist in `bind/joint.cpp`.
- **Collisions:** the car uses `AGENT_FILTER`, so it collides with the scenery but never with
  the other agents.

### The dynamic parcours (seeded, so identical for every candidate)

The parcours is a series of segments, getting harder from left to right:
- **Terrain:** rough ground from a `chain_def` built from seeded noise, plus steps, gaps and
  ramps.
- **Moving platform:** a kinematic body moving on a sine path (set its velocity each step).
  This is the "timing matters" element.
- **Seesaw:** a plank on a revolute joint. A car that is too light doesn't tip it, one that
  is too short falls off.
- **Collapsing bridge:** planks joined by revolute joints.
  - Each joint has a `force_threshold`. We read the joint's `constraint_force` every k steps
    and destroy the joint above the threshold, since pyb2d3 exposes no joint events
    (`PYB2D3_NOTES.md`).
  - A heavy car breaks the bridge, a fast car gets across before it falls.
- **Boulders:** rolling boulders released at fixed times, and a swinging pendulum.

The seed is the same for the whole population within an iteration (common random numbers: a
fair comparison). There are two options for the seed across iterations:
- **"Level" mode:** a fixed seed. The car overfits one parcours, which is fun to watch.
- **"Robust" mode:** a new seed every k iterations.

### Fitness

- **Return:** the furthest x the chassis reached, plus a bonus for reaching the goal of
  `(time left) · c`.
- **Termination:**
  - *stalled:* less than 0.2 m of progress in 2 s, the same rule as Walker's
    `STALL_SECONDS`/`STALL_DISTANCE`;
  - *flipped:* upside down for 1 s;
  - *fell:* below the parcours.
- **Truncation:** after 30 s.

Early termination is the main speed lever: most bad designs are done in about 2 s.

### "Cannot run multiple cars at once": what that means and what to do

The concern is real. Each car changes its world: it breaks the bridge, tips the seesaw, gets
hit by a boulder. So N cars in **one shared parcours** are not independent. Collision
filters only separate car from car; the shared dynamic objects still couple them. Options:

1. **One world per candidate**, in the training worker.
   - Box2d has 128 worlds, so a population of up to ~100 fits, leaving some for the display
     and the trainer.
   - The cost is not the ~8x loss we measured for the policy envs. That loss came from
     per-world *python* work: observations and actions per step, numpy per world.
   - A design env has **no per-step python work at all**: no policy, and the motors are set
     once at build time. Stepping is `world.step()` in C++ plus a cheap
     progress/termination check every k steps.
   - So `for w in worlds: w.step()` is close to the cost of one big world.
2. **Lanes: N copies of the parcours in one world**, each copy shifted by `y = i · 50 m`.
   Car i is built in lane i.
   - The copies never touch, so no filters are needed. The x/y batch reads still work.
   - Box2d is deterministic for identical setups. The shifted copies differ by float
     rounding, which is negligible here.
   - Lanes keep `Env(n)` exactly as it is, and they are nice for the display ("race").
3. **Sequentially, one car at a time:** restore the parcours snapshot, build the car, run
   until done. This is the simplest option, and per car-step it is just as fast (see the
   measurements below).
   - **This is the recommended option for the trainer.**
   - With early stopping, a population of 32 takes ~1–2 s per generation in wasm.

### Resetting instead of rebuilding: snapshots

pyb2d3 already exposes box2d's snapshots, and `core.Env.reset` uses them:
- `world.get_snapshot() -> bytes` and `world.restore_from_snapshot(bytes)`;
- `World.from_snapshot(bytes)` creates a new world from an image.

What a snapshot is (from box2d's docs in `include/pyb2d3/generated/docstrings.hpp`):
- It is the **whole simulation state**: bodies, shapes, joints, contacts, sleep state. It can
  only be taken at a step boundary.
- After a restore, the world id and every id of an object that existed at the snapshot stay
  valid.
- Objects created *after* the snapshot are gone after a restore, and user data is nulled.
- Box2d also has `b2World_GetStateHash` (not bound yet), useful to check determinism.

That gives exactly the per-candidate cycle we want:
1. Build the parcours once, then take `S0 = world.get_snapshot()`, without the car.
2. For each candidate: `world.restore_from_snapshot(S0)`, which resets the bridge, seesaw,
   boulders and platform and also removes the previous car. Then build the car (~10 bodies
   and shapes) and step.

Measured natively (`scratchpad/snap_bench.py`), with ~150 dynamic planks and 30 joints, a
180 KB snapshot, and a car of 8 triangles + 2 wheels:

| operation | time |
|---|---|
| get_snapshot | 22 µs |
| restore_from_snapshot | 18 µs |
| restore + build the car | 39 µs |
| rebuild the world from scratch in python | 259 µs |
| `World.from_snapshot` (a clone in a new world) | 20 µs |
| one step (1/60 s, 4 substeps; mostly sleeping) | 7 µs |

Expect roughly 3–5x slower in wasm.

**Conclusions:**
- Restoring is ~7–14x cheaper than rebuilding. Either way the reset is negligible next to an
  episode of hundreds of steps, so **restore + swap the car** is the right reset: it is
  simple and exact.
- **Several hidden worlds versus sequential resets in one world:**
  - The stepping cost per car is the same either way: box2d steps every body once, whether
    it sits in one world or many.
  - The browser has no threads in the worker (no pthreads build), so evaluating several
    worlds side by side gains nothing over evaluating them one after the other.
  - Sequential restore in **one** hidden world is the simplest and just as fast. It also
    stays far from the 128-world limit.
  - Extra worlds or lanes only pay off **for the display** (a "race" of the top-N designs
    in parallel) or with real threads natively (box2d `worker_count`).
- **Cloning:** `World.from_snapshot(S0)` (20 µs) clones the parcours into the display
  worker's world cheaply. The trainer could even send S0 to the display, which skips
  building there.

**Speed estimate:**
- A car plus parcours is ~100 bodies, at ~100–200 µs per step in wasm.
- With early stopping, the average episode is ~3 s of simulated time (180 steps).
- That gives ~30 ms per candidate, so a population of 32 is ~1 s per generation.
- CEM over 30 genes should find "a car that gets over the seesaw" in 10–30 generations,
  i.e. under a minute.

**Display:**
- The display worker shows the current best design driving the full parcours live, in its
  own world (it is the snapshot, as today).
- Optionally, with lanes, it shows `show_n` lanes stacked on top of each other: a "race" of
  the top-N designs, each on its own copy of the parcours. Zoomed out, that looks great.
- The ghost overlay doesn't apply here: the worlds differ, so ghosts would be misleading.
- The network section shows a **design card** (the genome as a table) instead of the
  network, since there is no network.

**Optional:** genome plus a tiny controller, so the car can brake or throttle based on its
tilt and speed (2 inputs → 1 output, a linear map of ~3 params). The params become
`concat(genome, controller)`. That still works with CEM/ES, and the controller part could
later be trained with PPO.

### What changes in the code

- **core.py:**
  - Add a design-env variant: class attribute `design_dim` (None for policy envs).
  - `reset(seed, designs=None)` builds agent i from `designs[i]`.
  - `step(actions)` accepts actions of shape `(n, 0)`.
  - Add a lane helper `lane_offset(i)` for envs that copy their scenery per agent;
    `build()` makes the scenery per lane.
- **policies.py:** a `DesignPolicy(design_dim)` with `n_params = design_dim` and
  `sizes = None`; its `__call__` returns empty actions. `policy_for(env, …)` returns it for
  design envs.
- **algos:**
  - `cem.rollout` / `es`: if the policy is a design policy, call
    `env.reset(seed, designs=params)` instead of building a forward function. That is a few
    lines.
  - `Algo.supports(env)`: PPO and DQN return False for design envs, unless the env has a
    controller part.
- **app.py:**
  - `network()` returns None, or a `design` dict for the design card.
  - `_watch_params` → `designs` for `watch()`.
  - `show_n` means lanes, not ghosts.
  - `_draw` without the ghost tint when lanes are used.
  - The camera follows nothing (as now). A "follow the car" toggle would be nice here,
    because the parcours is long.
- **host_web / main.js:** nothing, apart from rendering the design card. Snapshots already
  carry the params (= genome).

## 2. Which envs batch and which don't

Not every env can put its n agents into one world. The question is whether the agents can
affect each other's world.

| kind | examples | how the population is evaluated |
|---|---|---|
| **Batchable:** agents only touch static scenery; collision filters (`AGENT_FILTER`) make them pass through each other | CartPole, LunarLander, Walker, hopper, acrobot, blob crawler | n agents in one world, stepped together with numpy batch reads (today's `Env(n)`, ~8x faster than a world per agent) |
| **Not batchable:** agents change shared dynamic scenery (break, push, tip) or interact by design | car designer (bridge, seesaw, boulders), catapult (it hits targets), claw crane (shared boxes), sumo (two agents per arena) | (a) sequential: one hidden world, restore the snapshot, swap the agent, run; (b) lanes: a copy of the scenery per agent, spatially apart, in one world; (c) one world per agent |

What that means for the design:
- **Trainer:**
  - The algorithms don't care: CEM/ES only need `rollout(params) -> returns` for a batch of
    candidates.
  - For non-batchable envs, the env runs candidates one after the other inside its
    `reset` / `step` contract. For example, `Env(n)` keeps n "virtual" agents and steps
    the current one, or `rollout` gets a sequential path.
  - Cleanest: a `batchable = False` class attribute, with `cem.rollout` / `es` looping over
    candidates (restore → build → run until done). The loop still yields per step, so the
    frame loop and Stop keep working.
  - Policy algorithms (PPO, DQN) need parallel episodes. On non-batchable envs they would
    use lanes, or `n` = 1 (slow), so they fit the design envs less well.
- **Display:**
  - Batchable envs overlay ghosts in one view, as today.
  - Non-batchable envs show **one** agent per world: the current best, live. Optionally,
    with lanes, the top-N side by side as a race.
  - The "watch the population being evaluated" effect becomes "watch candidate k of 32 try
    the parcours" in the trainer's progress (state line `candidate 12/32`).
- **Speed:** sequential is not slower per agent-step, since box2d's cost is the same. It only
  loses the numpy batching of the obs/act python work, and design envs have none. For
  non-batchable *policy* envs (sumo, crane), lanes keep the batched reads, so prefer lanes
  there.

## 3. More envs (fast to train, fun live)

| env | obs → action | reward | batching | train time (browser) | wow |
|---|---|---|---|---|---|
| **Catapult / trebuchet designer** | design env: arm lengths, pivot height, counterweight mass, sling length, release angle (≈8 genes) | throw distance (one shot, ~3 s simulated) | lanes or worlds (design env) | CEM: 20–60 s | the design visibly evolves into a trebuchet; reuses all of the car designer's machinery |
| **Soft blob crawler** | ring of 8–12 circles joined by springy distance joints; the joints' rest lengths are actuated. Open loop: `L_k = L0 + a_k sin(ωt + φ_k)`, ~20 params; or closed loop: contacts/velocities → length targets | x velocity of the centroid | yes (AGENT_FILTER, one world) | CEM/ES: open loop 20–40 s, MLP 1–3 min | wobbly creatures discovering a gait; dragging them with the mouse is fun |
| **Acrobot swing-up on a cart** | cart x, v, two angles + angular velocities → cart force (+ elbow torque) | upright height of the tip, then a balancing bonus | yes (like CartPole) | ES/PPO: 1–3 min (CEM struggles with the swing-up) | the classic "how did it figure that out"; you can try it yourself as a human (hard) |
| **One-legged hopper / jumper** | torso angle, velocity, leg joint angles, foot contact → hip/knee torque | forward x + alive bonus, minus energy | yes (like Walker) | PPO: 3–5 min; ES: 2–4 min | hopping gaits emerge; a "jump over the wall" variant with a seeded wall height |
| **Claw crane** | trolley x, rope length, claw open, box relative position → trolley force, winch, open/close | shaped: distance claw–box, grasped, box in bin | yes, if every agent has its own box (filters: box i only with claw i and scenery; needs per-agent categories, 64 bits) | PPO: 5–10 min (sparse) | you can knock the box away while it works; a satisfying drop |
| **Sumo (self-play)** | own and opponent position/velocity, ring edge distance → wheel torques / push | pushes the opponent out | arenas side by side (lanes) in one world | ES self-play: minutes; best shown as champion vs challenger | two learned agents fighting; the human can play against the champion |

Notes:
- **Lanes are the common trick for all "dynamic scenery" envs** (car, catapult, sumo, the
  crane's boxes): copy the scenery per agent, spatially separated, in one world. It is one
  helper in core.py, worth building once with the car designer.
- **The design envs** (car, catapult) are the most "browser-demo" friendly: no torch or jax
  needed, CEM in numpy converges in under a minute, and the result is a funny-looking
  machine, not an opaque network.
- **The bridge builder** (place planks, a truck drives over, breakable joints) is fun too,
  but needs ~50+ genes and break detection every step. Do it later, after the car designer
  has proven the force-threshold path.

## 4. Recommended order

1. **Car designer.** Add the design-env infra: `design_dim`, `DesignPolicy`,
   `batchable = False` with a sequential snapshot-restore rollout in CEM/ES, and the design
   card. Lanes come later, for the display's race. Start with static terrain plus the seesaw and the
   moving platform; the collapsing bridge and the boulders come second.
2. **Catapult designer.** This is almost free once the infra exists.
3. **Soft blob crawler.** It trains fastest, looks funny, and is a policy env, so it needs no
   new infra.
4. **Acrobot swing-up.** A showcase for PPO/ES versus CEM.
5. **Hopper.**
6. **Sumo / claw crane.** Self-play and sparse rewards come last.
