# pyb2d3: what was done about webgym's notes (PYB2D3_NOTES.md)

Implemented on the `webgym` branch of pyb2d3 (not released yet: the version still says 1.0.0,
the published 1.0.0 on PyPI / emscripten-forge does **not** have these changes). To use them,
build pyb2d3 from that branch (natively `pixi run build` / `pip install .`; for the browser
`pixi run emscripten`, which writes the package to `emscripten_forge/output/`).

## 1. More than 128 worlds: now an error
`b2d.World()` raises `RuntimeError("could not create a world: box2d supports at most 128
worlds at a time ...")` instead of returning an invalid world. Slots of destroyed worlds are
reused. Destroy worlds you no longer need (`world.destroy()`, or let them be garbage collected).
webgym's own `world.is_valid` check in `webgym.core.new_world()` can go.

## 2. Batches for every joint type
Every joint type has its own batch class: `b2d.RevoluteJoints`, `PrismaticJoints`,
`WheelJoints`, `DistanceJoints`, `MotorJoints`, `MoverJoints`, `PogoJoints`, `WeldJoints`,
`FilterJoints`. They carry the joint type's whole batchable api, with numpy in / out like
`b2d.Bodies`:

```python
hips = b2d.RevoluteJoints(hip_joints)           # list of joints, or an id array
hips.motor_enabled = True
hips.max_motor_torque = 80.0                     # one value (broadcast) ...
hips.target_motor_speed = actions                # ... or one per joint, shape (n,)
angle, speed = hips.current_angle, hips.current_speed   # (n,) float32 arrays
hips.set_limits(-0.8, 0.8)                       # methods too
b2d.Joints(hips.ids).constraint_force            # the generic joint api: a Joints batch
```

- Construction / `append` checks that every id is a joint of that type (`TypeError`
  otherwise): box2d itself does not check this, a wrong type corrupts memory in release / wasm.
- Indexing like numpy: `hips[3]` is a `RevoluteJoint`, `hips[mask]` / slices a `RevoluteJoints`.
- The Walker workaround (torques via `Bodies.apply_torque`, angles from body angles) can be
  replaced by motors + `current_angle` / `current_speed`.
- Sample: "Batch API / Joint Batches" in the sandbox (`pixi run sandbox "Batch API/Joint
  Batches"`) drives every joint type each step, once with the batches and once with a python
  loop, and shows both times (natively ~8x faster with ~3600 joints, ~2x with ~600; in the
  browser the gap is smaller at small counts).

## 3. Resetting some agents: no new API, a recipe
`restore_from_snapshot` stays whole-world. To reset only some agents, store their start state
and write it back with the batch api:

```python
start_pos, start_rot = bodies.position, bodies.rotation   # once, at the start
...
done = bodies[mask]                                          # the agents whose episode ended
done.set_transform(start_pos[mask], start_rot[mask])
done.linear_velocity = (0, 0)
done.angular_velocity = 0.0
done.awake = True
```

Limits (this is a teleport, not a restore): joint motor targets / settings you changed stay as
they are (reset them too, e.g. with the joint batches), the solver's warm-starting values of
joints and contacts carry into the first step, and a body must not be put inside another one.
A test checks that a reset body then follows the same trajectory as an untouched one.

## 5. `BatchDebugDraw` arrays
Each buffer property (`batch.polygons`, `circles`, ...) returns a **new array that owns its
copy** of the data: it stays valid after `clear()` and later draws. webgym's extra copy is not
needed. Every property access copies, so read each one once per frame.

## 6. `world.bodies`
Not added: box2d has no function to list a world's bodies. Keep your own `b2d.Bodies`.

## 7. "Is this foot touching the ground", batched: sensors (no new API)
Give each foot a sensor shape whose filter selects the ground, and enable sensor events on the
ground shapes (off by default: without it the sensors see nothing). `sensor_capacity` is the
number of overlaps of the last step and is batched:

```python
GROUND, FOOT = 0x2, 0x4
ground_shape.filter = b2d.make_filter(category_bits=GROUND, mask_bits=0xFFFF)
ground_shape.sensor_events_enabled = True                 # required!

sensor_def = b2d.shape_def(is_sensor=True, enable_sensor_events=True)
sensor_def.filter = b2d.make_filter(category_bits=FOOT, mask_bits=GROUND)
foot_sensor = foot_body.create_polygon_shape(sensor_def, b2d.box(0.4, 0.1, center=(0, -0.5), rotation=0))
...
touching = b2d.Shapes(foot_sensors).sensor_capacity > 0   # bool per foot, after a step
```

Works on any terrain. `Bodies.contact_capacity` is no substitute: it counts every contact whose
bounding boxes overlap, touching or not.

## 8. Collision filters vs. queries
- `b2d.MouseDrag(world, query_filter=b2d.make_query_filter(category_bits=..., mask_bits=...))`
  grabs shapes the default query filter cannot see.
- Documented: box2d checks the filter both ways, so a shape whose `mask_bits` exclude category 1
  is invisible to the default query filter (also `dynamic_body_at_point`).

## 9. Snapshot restore: documented
`world.restore_from_snapshot(image)`: bodies / shapes / joints that existed at the snapshot get
their snapshot state back (pose, velocities, enabled, ...); objects created after the snapshot
are removed and their handles become invalid (covered by a test). Two things to know:

- the **`user_data` of bodies, shapes and joints is reset to 0** (keep your own id mapping),
- the snapshot API is marked experimental in box2d (the format may change between versions).

So "throw things at it, the next reset clears them" and "restore the parcours, build each car
anew" are fine to build on, within box2d's own guarantees.

## Also fixed (native sandbox, OpenGL frontend)
- Debug text (`draw_string`, body names) was drawn shifted left by the width of the left panel
  in every sample: fixed. (The browser and pygame frontends were not affected.)
- The start sample is centered again once the panel is docked (wide scenes were cut off).
- A sample's own debug draw settings (e.g. `self.debug_draw.draw_joints = False`) are no longer
  overridden at startup.

FYI noted: the wasm64 build works for all webgym envs. Thanks for the report.
