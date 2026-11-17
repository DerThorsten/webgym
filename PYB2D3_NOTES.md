# pyb2d3: what webgym is missing (candidates for the `webgym` branch of pyb2d3)

Found while building webgym. Each item: what happened, the workaround in webgym, the
suggested change in pyb2d3.

1. **Creating a world beyond `B2_MAX_WORLDS` (128) fails silently.**
   `b2d.World()` returns `World(id=0)`, an invalid world; using it corrupted memory in wasm
   ("memory access out of bounds" at max population).
   Workaround: `webgym.core.new_world()` checks `world.is_valid` and raises.
   Suggestion: `World.__init__` / `create_world_id` raise when the id is invalid.

2. **No batch API for joint-type specific properties.** `b2d.Joints` only has the generic
   joint properties; revolute `target_motor_speed`, `max_motor_torque`, `current_angle`,
   `current_speed` (and prismatic / wheel equivalents) exist only per joint.
   Workaround (Walker): torques via the batched `Bodies.apply_torque` on both bodies, joint
   angles / speeds computed from body angles; a python loop over joints would cost
   4 x n calls per step.
   Suggestion: `RevoluteJoints` / `PrismaticJoints` batch classes (like `Bodies`).

3. **Resetting a subset of agents in one world.** `restore_from_snapshot` restores the
   whole world; per-agent episodes (auto reset, e.g. PPO) need to reset some agents only.
   Workaround: all agents reset together (episodic, fine for CEM / ES).
   Suggestion: batch "reset bodies to a stored state" (transforms + zero velocities,
   re-enable), or a snapshot restore restricted to a set of bodies.

4. **Hiding bodies from debug drawing without changing the simulation.** Disabling a body
   hides it but removes it from the broadphase (contacts lost); fine for finished agents,
   not for "draw only these agents" (e.g. the leader on top of the ghosts).
   Workaround: leader marked with a shape `custom_color`, ghosts recolored in numpy.
   Suggestion: a debug draw visibility flag per body / shape, or `BatchDebugDraw` options:
   draw only bodies of a `Bodies` set, color / alpha override.

5. **`BatchDebugDraw` arrays and `clear()`.** Unclear whether the arrays returned
   (`batch.polygons`, ...) stay valid after the next `clear()` / draw; drawing several
   worlds into one frame needs them to.
   Workaround: webgym copies every block.
   Suggestion: document it (or return owned arrays).

6. **No way to list a world's bodies** (`world.bodies`). Minor (tests, tools).

7. **Contact state per agent, batched.** "Is this foot touching the ground" is computed
   geometrically (leg tip height on flat ground); with terrain that breaks.
   Suggestion: batched contact / sensor queries, e.g. `Bodies.touching(category_bits)` ->
   bool array.

8. **Collision filters vs. queries (docs).** A shape whose `mask_bits` exclude category 1
   is invisible to the default query filter, so `MouseDrag` cannot grab it; it took a
   while to find (box2d checks the filter both ways).
   Suggestion: document it; let `MouseDrag` / `dynamic_body_at_point` take a query filter.
