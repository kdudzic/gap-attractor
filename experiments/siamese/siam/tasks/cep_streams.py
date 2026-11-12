"""Streams of CodeEval-Pro items with known dependencies (siam/tasks/codeeval_pro.py), and a manifest of every link:
the ground truth for analysing knowledge transfer between items.

Every agent's stream has the same length. The stream is laid out in blocks of 2 (gap + 1) positions; in a block the
first gap + 1 positions are "first" slots and the next gap + 1 are "second" slots, so slot j of a block links
position p (first) to position p + gap + 1 (second), with exactly `gap` positions between them. Each slot j of a
block has one type, the same for every agent:
- forward_self      agent a: base at p, its dependent at p + gap + 1
- forward_partner   agent a: base at p; agent a + 1 (a ring): its dependent at p + gap + 1
- backward_self     agent a: dependent at p, its base at p + gap + 1
- backward_partner  agent a: dependent at p; agent a + 1: its base at p + gap + 1
- control           HumanEval-Pro controls at p and p + gap + 1 (no link)
- solo              a base at p and a dependent at p + gap + 1 from two other pairs, whose other items appear nowhere
                    in the world (no link)
Positions left over after the last whole block hold controls. "Later" means later in the agent's own stream: with
agents taking one step per round, position p + gap + 1 of agent b comes after position p of agent a in round order
when they work at the same pace; the logs (results.jsonl, rounds) give the real times.

Arrangements: forward_self | forward_partner | backward_self | backward_partner (every slot that type);
control (slots alternate solo and control); mixed (the slot types of MIX in a seeded order, cycled when there are
more slots). A pair is used at most once in a world, a control at most once; everything is drawn by `seed`.
`rotate` turns the seeded order of the pairs by that many: the same slot types, other pairs in them (counterbalancing
pairs over link types across worlds of one seed).

`gate` (cross-agent links only): the second item of a *_partner link carries `after` = the first item's id, and the
world does not hand it out before that item is closed (siam/world.py next_problem; 31 Oct). Without the gate, "later"
is only a stream position, and a faster partner can reach the second item before the first is solved, so the link
tests nothing (or the reverse direction). Keep in mind as a variant to try: no gate, analysing only the links that
happened in order (no reordering of streams, fewer valid links).
"""
from __future__ import annotations

import copy
import random

LINKS = ("forward_self", "forward_partner", "backward_self", "backward_partner")
MIX = ("forward_self", "forward_partner", "backward_self", "backward_partner", "control", "control")
ARRANGEMENTS = LINKS + ("control", "mixed")


def slot_types(arrangement: str, n_slots: int, n_agents: int, rng: random.Random, mix=MIX) -> list[str]:
    if arrangement in LINKS:
        if arrangement.endswith("partner") and n_agents < 2:
            raise ValueError(f"{arrangement} needs at least 2 agents")
        return [arrangement] * n_slots
    if arrangement == "control":
        return ["solo" if i % 2 == 0 else "control" for i in range(n_slots)]
    if arrangement == "mixed":
        blend = [t for t in mix if n_agents > 1 or not t.endswith("partner")]
        types = [blend[i % len(blend)] for i in range(n_slots)]
        rng.shuffle(types)
        return types
    raise ValueError(f"arrangement must be one of {ARRANGEMENTS} (got {arrangement})")


def build(data: dict, n_agents: int, seed: int, arrangement: str, length: int = 12, gap: int = 2, mix=MIX, rotate: int = 0,
          gate: bool = False):
    """(streams, manifest): streams[a] is agent a + 1's list of items (copies); the manifest records every link and
    every position (agents and positions counted from 1, as in the world's texts)."""
    if n_agents < 1 or length < 1 or gap < 0:
        raise ValueError("n_agents and length must be at least 1, gap at least 0")
    rng = random.Random(seed)
    pairs = list(data["pairs"])
    controls = list(data["controls"])
    rng.shuffle(pairs)
    rng.shuffle(controls)
    r = rotate % len(pairs) if pairs else 0
    pairs = pairs[r:] + pairs[:r]                 # (pairs are taken from the end)
    half = gap + 1
    blocks = length // (2 * half)
    types = slot_types(arrangement, blocks * half, n_agents, rng, mix)
    streams = [[None] * length for _ in range(n_agents)]
    links, solo, used_controls = [], [], []

    def take_pair():
        if not pairs:
            raise ValueError(f"not enough pairs for {n_agents} agents x {length} items ({arrangement}, gap {gap})")
        return pairs.pop()

    def put(agent, pos, item):
        assert streams[agent][pos] is None
        streams[agent][pos] = copy.deepcopy(item)
        streams[agent][pos]["planned_position"] = pos + 1   # the gate may serve it later (results.jsonl has both)

    def put_control(agent, pos):
        if not controls:
            raise ValueError(f"not enough controls for {n_agents} agents x {length} items")
        c = controls.pop()
        put(agent, pos, c)
        used_controls.append({"id": c["id"], "agent": agent + 1, "position": pos + 1})

    for s, kind in enumerate(types):
        p = (s // half) * 2 * half + s % half
        q = p + half
        for a in range(n_agents):
            if kind == "control":
                put_control(a, p)
                put_control(a, q)
            elif kind == "solo":
                x, y = take_pair(), take_pair()
                put(a, p, x["base"])
                put(a, q, y["dependent"])
                solo += [{"pair_id": x["pair_id"], "id": x["base"]["id"], "role": "base", "agent": a + 1, "position": p + 1},
                         {"pair_id": y["pair_id"], "id": y["dependent"]["id"], "role": "dependent", "agent": a + 1, "position": q + 1}]
            else:
                pair = take_pair()
                b = a if kind.endswith("self") else (a + 1) % n_agents
                first, second = ("base", "dependent") if kind.startswith("forward") else ("dependent", "base")
                put(a, p, pair[first])
                put(b, q, pair[second])
                if gate and b != a:
                    streams[b][q]["after"] = pair[first]["id"]
                links.append({"pair_id": pair["pair_id"], "type": kind, "direction": kind.split("_")[0],
                              "who": kind.split("_")[1], "first": first, "gap": gap,
                              first: {"id": pair[first]["id"], "agent": a + 1, "position": p + 1},
                              second: {"id": pair[second]["id"], "agent": b + 1, "position": q + 1}})
    for a in range(n_agents):
        for pos in range(length):
            if streams[a][pos] is None:
                put_control(a, pos)
    manifest = {"arrangement": arrangement, "seed": seed, "rotate": rotate, "gate": gate, "n_agents": n_agents, "length": length, "gap": gap,
                "slot_types": types, "links": links, "solo": solo, "controls": used_controls,
                "streams": [[{"id": it["id"], "role": it["role"], "pair_id": it["pair_id"], "title": it["title"]} for it in st]
                            for st in streams]}
    return streams, manifest
