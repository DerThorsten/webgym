"""NEAT: NeuroEvolution of Augmenting Topologies (Stanley & Miikkulainen, 2002).

Evolves the weights *and the shape* of the network. Every genome starts small: its
inputs wired straight to its outputs. Mutations add connections and neurons (a neuron
splits a connection in two). New structure first costs fitness until its weights are
tuned: so the population is split into species of similar genomes, and genomes
compete mostly within their species (each species gets offspring by its mean fitness).

A genome: {"nodes": {node id: bias}, "conns": {(from, to): [weight, enabled]}}, node
ids: inputs 0 .. n_in - 1, outputs n_in .. n_in + n_out - 1, hidden neurons after that.
The pair (from, to) is the connection's innovation number: the same connection in two
genomes is the same gene (crossover and the species distance line genes up by it). A
neuron that splits (a, b) gets the same id in every genome that does it (`splits`).

Every node has an `order`: inputs 0, outputs 1, a new neuron halfway between the two
ends of the connection it splits; connections only go from a lower to a higher order,
so the network stays feed-forward. For the episodes every genome is written into the
flat vector of a webgym.policies.Graph: the whole population runs at once with numpy.
"""

import numpy as np

from webgym.algos.cem import population, rollout

SURVIVE = 0.2  # the best share of a species become parents
STALE = 15  # a species that didn't improve for this many generations dies out
MAX_WEIGHT = 8.0
REPLACE = 0.02  # the chance that a mutated weight gets a new random value


def train(env, policy, iterations=1000, add_node=0.1, add_conn=0.3, weight_power=0.3, species=5,
          seed=0, state=None):
    """NEAT on `policy` (a webgym.policies.Graph); population size: population(env).
    Yields None per env step and stats after each generation; `state` resumes a run."""
    rng = np.random.default_rng(seed)
    s = load(state) if state is not None else start(policy, population(env), rng)
    rates = {"add_node": add_node, "add_conn": add_conn, "power": weight_power}
    for iteration in range(s["iteration"], s["iteration"] + iterations):
        n = population(env)
        genomes = [s["genomes"][i % len(s["genomes"])] for i in range(n)]  # (another size: repeated / cut)
        params = np.stack([encode(g, policy, s["order"]) for g in genomes])
        returns = yield from rollout(env, policy, params, seed=int(rng.integers(1 << 30)))
        members = speciate(s, genomes, returns, iteration, species)
        s["genomes"] = reproduce(s, genomes, returns, members, n, rng, policy, rates)
        s["iteration"] = iteration + 1
        ranked = np.argsort(returns)[::-1]
        champion = genomes[ranked[0]]
        yield {
            "iteration": iteration + 1,
            "mean_return": float(returns.mean()),
            "best_return": float(returns.max()),
            "mean_params": params[ranked[0]],  # (the watched policy: this generation's best)
            "losses": {  # (plotted)
                "species": len(s["species"]),
                "neurons": len(champion["nodes"]) - policy.n_out,
                "connections": sum(on for _, on in champion["conns"].values()),
            },
            "state": save(s),
        }


# -- genomes ------------------------------------------------------------------------


def start(policy, n, rng):
    """the first generation: inputs connected to outputs, random weights"""
    inputs, outputs = range(policy.n_in), range(policy.n_in, policy.n_in + policy.n_out)
    return {
        "iteration": 0,
        "genomes": [{"nodes": {o: 0.0 for o in outputs},
                     "conns": {(i, o): [rng.normal(), True] for i in inputs for o in outputs}}
                    for _ in range(n)],
        "species": [],  # [{"rep": a genome, "best": best fitness, "since": generation it improved}]
        "order": {**{i: 0.0 for i in inputs}, **{o: 1.0 for o in outputs}},
        "splits": {},  # (from, to) -> the id of the neuron that splits it
        "next_node": policy.n_in + policy.n_out,
        "threshold": 1.0,  # species distance (adapted: about `species` species)
    }


def copy(g):
    return {"nodes": dict(g["nodes"]), "conns": {k: list(v) for k, v in g["conns"].items()}}


def encode(g, policy, order):
    """the genome as one flat vector of `policy` (a Graph): hidden neurons by order"""
    n_in, n_out = policy.n_in, policy.n_out
    hidden = sorted((k for k in g["nodes"] if k >= n_in + n_out), key=lambda k: (order[k], k))
    index = {i: i for i in range(n_in)}  # node id -> the Graph's node index
    index.update({o: n_in + policy.max_hidden + o - n_in for o in range(n_in, n_in + n_out)})
    index.update({k: n_in + j for j, k in enumerate(hidden)})
    edges = [(index[a], index[b], w) for (a, b), (w, on) in g["conns"].items() if on]
    return policy.pack(edges, {index[k]: bias for k, bias in g["nodes"].items()}, hidden)


def distance(g1, g2):
    """how different two genomes are: the share of genes only one has + the mean
    weight difference of the genes both have"""
    c1, c2 = g1["conns"], g2["conns"]
    common = c1.keys() & c2.keys()
    disjoint = len(c1) + len(c2) - 2 * len(common)
    weights = np.mean([abs(c1[k][0] - c2[k][0]) for k in common]) if common else 0.0
    return disjoint / max(len(c1), len(c2), 1) + 0.5 * weights


def crossover(a, b, rng):
    """a child of `a` (the fitter parent: its structure) and `b`: genes both have from
    either parent; a gene disabled in a parent is mostly disabled in the child"""
    child = copy(a)
    for k, gene in child["conns"].items():
        other = b["conns"].get(k)
        if other is not None:
            if rng.random() < 0.5:
                gene[0] = other[0]
            if not other[1] and rng.random() < 0.75:
                gene[1] = False
    for k in child["nodes"]:
        if k in b["nodes"] and rng.random() < 0.5:
            child["nodes"][k] = b["nodes"][k]
    return child


def perturb(values, rng, power):
    """mostly a small random change, sometimes a new random value"""
    k = len(values)
    new = np.where(rng.random(k) < REPLACE, rng.normal(size=k), np.asarray(values) + power * rng.normal(size=k))
    return np.clip(new, -MAX_WEIGHT, MAX_WEIGHT).tolist()


def mutate(g, s, rng, policy, rates):
    """perturb weights and biases; maybe a new connection, a new neuron, a toggled gene"""
    if rng.random() < 0.8:
        genes = list(g["conns"].values())
        for gene, w in zip(genes, perturb([w for w, _ in genes], rng, rates["power"])):
            gene[0] = w
        g["nodes"] = dict(zip(g["nodes"], perturb(list(g["nodes"].values()), rng, rates["power"])))
    order = s["order"]
    if rng.random() < rates["add_conn"]:  # a new connection (from a lower to a higher order)
        nodes = [*range(policy.n_in), *g["nodes"]]
        for _ in range(10):  # (a few tries: an unconnected pair)
            a, b = sorted(rng.choice(nodes, 2), key=lambda k: order[k])
            if order[a] < order[b] and (a, b) not in g["conns"]:
                g["conns"][(int(a), int(b))] = [rng.normal(), True]
                break
    hidden = len(g["nodes"]) - policy.n_out
    enabled = [k for k, (_, on) in g["conns"].items() if on]
    if rng.random() < rates["add_node"] and hidden < policy.max_hidden and enabled:
        a, b = enabled[rng.integers(len(enabled))]  # a new neuron splits this connection
        node = s["splits"].get((a, b))
        if node is None or node in g["nodes"]:  # (the same split as in other genomes: same id)
            node = s["next_node"]
            s["next_node"] += 1
            order[node] = (order[a] + order[b]) / 2
            s["splits"].setdefault((a, b), node)
        g["conns"][(a, b)][1] = False
        g["conns"][(a, node)] = [1.0, True]  # (in: 1, out: the old weight: about as before)
        g["conns"][(node, b)] = [g["conns"][(a, b)][0], True]
        g["nodes"][node] = 0.0
    if g["conns"] and rng.random() < 0.02:  # switch a connection on / off
        gene = list(g["conns"].values())[rng.integers(len(g["conns"]))]
        gene[1] = not gene[1]


# -- species and the next generation ---------------------------------------------------


def speciate(s, genomes, returns, generation, target):
    """put every genome in the first species whose representative is close enough
    (else a new species); drop empty and stale species; -> members per species"""
    members = [[] for _ in s["species"]]
    for i, g in enumerate(genomes):
        for k, sp in enumerate(s["species"]):
            if distance(g, sp["rep"]) < s["threshold"]:
                members[k].append(i)
                break
        else:
            s["species"].append({"rep": g, "best": float(returns[i]), "since": generation})
            members.append([i])
    best = int(np.argmax(returns))
    kept = []
    for sp, m in zip(s["species"], members):
        if not m:
            continue
        top = max(m, key=lambda i: returns[i])
        if returns[top] > sp["best"]:
            sp["best"], sp["since"] = float(returns[top]), generation
        if generation - sp["since"] < STALE or best in m:  # (the best genome's species stays)
            sp["rep"] = genomes[top]
            kept.append((sp, m))
    s["species"] = [sp for sp, _ in kept]
    # about `target` species: a smaller threshold makes more of them
    s["threshold"] = max(0.1, s["threshold"] + (0.1 if len(kept) > target else -0.1 if len(kept) < target else 0))
    return [m for _, m in kept]


def reproduce(s, genomes, returns, members, n, rng, policy, rates):
    """n children: each species gets offspring by its mean fitness (at least 2); its
    best genome goes on unchanged, the rest are mutated children of its best ones"""
    fitness = np.array([np.mean(returns[m]) for m in members]) - returns.min() + 1e-6
    counts = np.maximum(2, np.floor(fitness / fitness.sum() * n)).astype(int)
    while counts.sum() > n:
        counts[np.argmax(counts)] -= 1
    counts[np.argmax(fitness)] += n - counts.sum()
    children = []
    for m, count in zip(members, counts):
        ranked = sorted(m, key=lambda i: -returns[i])
        parents = ranked[: max(2, int(np.ceil(SURVIVE * len(ranked))))]
        for c in range(count):
            if c == 0:
                children.append(copy(genomes[ranked[0]]))  # (the elite)
                continue
            a, b = sorted(rng.choice(parents, 2), key=lambda i: -returns[i])
            child = crossover(genomes[a], genomes[b], rng) if a != b and rng.random() < 0.75 else copy(genomes[a])
            mutate(child, s, rng, policy, rates)
            children.append(child)
    return children


# -- the state (JSON-able: it goes into checkpoints) -------------------------------------


def save(s):
    """the bookkeeping as plain lists and numbers"""
    genome = lambda g: {"nodes": [[k, b] for k, b in g["nodes"].items()],  # noqa: E731
                        "conns": [[a, b, w, on] for (a, b), (w, on) in g["conns"].items()]}
    return {
        "iteration": s["iteration"],
        "genomes": [genome(g) for g in s["genomes"]],
        "species": [{"rep": genome(sp["rep"]), "best": sp["best"], "since": sp["since"]} for sp in s["species"]],
        "order": [[k, v] for k, v in s["order"].items()],
        "splits": [[a, b, k] for (a, b), k in s["splits"].items()],
        "next_node": s["next_node"],
        "threshold": s["threshold"],
    }


def load(state):
    """save()'s lists back to the bookkeeping"""
    genome = lambda g: {"nodes": {int(k): b for k, b in g["nodes"]},  # noqa: E731
                        "conns": {(int(a), int(b)): [w, bool(on)] for a, b, w, on in g["conns"]}}
    return {
        "iteration": int(state["iteration"]),
        "genomes": [genome(g) for g in state["genomes"]],
        "species": [{"rep": genome(sp["rep"]), "best": sp["best"], "since": sp["since"]} for sp in state["species"]],
        "order": {int(k): v for k, v in state["order"]},
        "splits": {(int(a), int(b)): int(k) for a, b, k in state["splits"]},
        "next_node": int(state["next_node"]),
        "threshold": state["threshold"],
    }
