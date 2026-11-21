"""My algorithm: a starting point for your own learning algorithm (edit me!).

A hill climber, about the simplest thing that learns: try small random changes of the
best policy so far, one per agent, and keep the one that scored best. Open the Code
(pick "algorithm"), edit, press Apply (Ctrl/Cmd+Enter): the training goes on with it.
Some ideas:
- keep the best few and average them (that is close to CEM)
- shrink the changes over time
- run each candidate for several episodes (less luck)

The words used below:
- agent: one copy of the robot (ball, walker, ...); the env runs env.n of them at once,
  side by side in one world (they don't touch each other)
- step: one tick of the simulation (1/60 s): every agent sees, acts, gets a reward
- episode: one try from a fresh start until the agent is done (fallen, time is up, ...)
- weights: the numbers inside the policy's neural network; learning = finding good ones
- return: the sum of an agent's rewards over its episode: its score

The shapes (n = env.n agents, P = policy.num_params weights, O = len(env.obs_names)
observations, A = len(env.action_names) actions). Everything is a numpy array with ONE
ROW PER AGENT, so all agents are handled at once:
- weights of one policy: (P,)        e.g. best
- weights for all agents: (n, P)      e.g. candidates: row i = the weights agent i uses
- observations: (n, O)               what each agent sees NOW (this step only, not a
                                      history): row i = agent i's values, in the order of
                                      env.obs_names (MyEnvironment: distance, velocity)
- actions: (n, A)                    what each agent does now, each value in [-1, 1]
- reward, terminated, truncated, env.done, env.episode_return: (n,), one per agent
"""

import numpy as np


def train(env, policy, iterations=1000, change=0.1, seed=0, state=None):
    """Train `policy` on `env`.

    env:      the environment with env.n agents, all stepped at once
    policy:   the neural network: policy(weights (n, P), obs (n, O)) -> actions (n, A),
              row i of the actions = row i of the weights applied to row i of the obs;
              policy.num_params = P, how many weights one network has
    change:   how big the random changes are (a slider in the Training tab; the sliders
              are declared in webgym/algos/__init__.py, ALGOS["my_algorithm"])
    seed:     for the random numbers (the same seed: the same run)
    state:    None for a new run, else what an earlier run yielded as stats["state"]
              (the page passes it back to continue, e.g. after Stop / Train)

    This is a generator: it never returns, the page pulls from it.
    - `yield None` after every env step: the page gets a chance to draw, take a click,
      or stop the training
    - `yield {...}` after every iteration: the stats for the plot, the top bar and the
      policy you watch
    """
    rng = np.random.default_rng(seed)

    # where to start: a new run begins with all weights 0 (a policy that does nothing),
    # a continued run with what it had learned. best: (P,), one network's weights
    if state is None:
        state = {"iteration": 0, "best": np.zeros(policy.num_params)}
    best, start = state["best"], state["iteration"]

    for iteration in range(start + 1, start + iterations + 1):
        # 1. the candidates, (n, P): one row of weights per agent. Agent 0 keeps the best
        #    so far (so it is tested again: a lucky score doesn't stick), all others try
        #    a small random change of it (normal noise times `change`)
        noise = rng.standard_normal((env.n, policy.num_params))  # (n, P)
        candidates = best + change * noise  # (P,) + (n, P) -> (n, P): best in every row
        candidates[0] = best

        # 2. one episode, all agents at once: agent i acts with candidates[i].
        #    env.reset() puts every agent at a fresh start and returns the first
        #    observations, (n, O). The seed picks the random start (MyEnvironment: where
        #    the target is)
        obs = env.reset(seed=int(rng.integers(1 << 30)))
        # env.done, (n,) bools: True for an agent whose episode is over. Such an agent
        # waits (it is frozen, gets reward 0) until all are done
        while not env.done.all():
            # the policy looks at this step's observations only, (n, O), and decides
            # this step's actions, (n, A). (It has no memory: what it saw before is gone)
            actions = policy(candidates, obs)
            # one step of the world for all agents. Back come, one row / entry per agent:
            # obs (n, O): what they see now, after the step (used in the next loop pass)
            # reward (n,): what this step earned (MyEnvironment: minus the distance)
            # terminated (n,) bools: the episode ended just now (e.g. fallen)
            # truncated (n,) bools: the time ran out just now (env.max_steps)
            # (this hill climber only needs the final scores below; an algorithm that
            # learns from single steps, as PPO or DQN, uses reward / terminated here)
            obs, reward, terminated, truncated = env.step(actions)
            yield None  # (the page draws, then continues here)

        # 3. the score of each candidate. env.episode_return, (n,): the sum of each
        #    agent's rewards over this episode. np.argmax: the row with the highest
        #    score; its weights are the new best. (Ideas: average the best few, or adapt
        #    `change`: larger when nothing improves, smaller when it does)
        returns = env.episode_return
        best = candidates[np.argmax(returns)]  # (P,)

        # 4. tell the page how it went
        yield {
            "iteration": iteration,
            "mean_return": float(returns.mean()),  # (the blue line in the progress plot)
            "best_return": float(returns.max()),  # (the orange line)
            "mean_params": best,  # the weights of the policy you watch, (P,)
            # everything needed to continue this run later (also saved in checkpoints)
            "state": {"iteration": iteration, "best": best},
        }
