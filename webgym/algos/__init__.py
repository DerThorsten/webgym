"""Trainers. Each is a generator `train(env, policy, state=None, **hp)` on a batched
env (env.n agents): it yields None after every env step (so a frame loop can stop it
any time) and a stats dict at the end of every iteration; stats["state"] passed back
as `state` resumes training."""

from webgym.algos.cem import train as cem

ALGOS = {"cem": cem}
