"""Trainers. Each is a generator `train(envs, policy, **hp)`: it yields None after
every population step (so a frame loop can stop it any time) and a stats dict at
the end of every iteration."""

from webgym.algos.cem import train as cem

ALGOS = {"cem": cem}
