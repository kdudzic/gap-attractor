"""Joint worlds: N agents, ONE context file (and the controls that take parts of the sharing away).

Every agent of a world has a ContextFileMemory over the same ContextFile: each agent's pages, notices, edits and
moves are appended to it and every agent is sent the whole file. Agents act one at a time: in every round each
active agent takes one step, in a seeded random order (or by agent number), so each one sees the file as the
previous agent left it and edits never conflict. Agents may run different models (`ctx.joint.agent_providers`).
An agent that stops or ends drops out (world.json says when and why); its turns stay in the file.

`ctx.joint.sharing`: full (anyone edits anything) | readonly_others (exposure without cross-editing) | separate
(one file per agent, same scheduler: the no-sharing control) | notes (one file per agent, and one folder, /notes,
in every agent's sandbox: the only thing they share; bash only; its contents are logged after every step in
worlds/world_NNN/notes.jsonl when they changed, with the agent that had just acted).

Goal worlds (`ctx.goal.task=find`, with prompt=find): every agent looks for a target page. `ctx.goal.setup`:
assigned (each agent's target is a turn of the context, marked as its own, protected until it arrives) | external
(the target is in the agent's own instructions only) | pool (one list of the world's targets in the context; any
agent may reach any; a reached one is struck from the list) | common (all agents of a world look for the same page:
one goal turn in the context, nobody's; any agent's arrival reaches it; needs ctx.goal.stop=first).
`ctx.goal.stop`: all (an agent that arrives ends; pool: all go on until the list is empty) | first (the world ends at
the end of the round of the first arrival) | all_found (assigned or external: only its owner's arrival reaches a
target, an agent that arrives goes on, and the world ends as soon as every target has been reached).

Logs: one ordinary run folder per agent (run.json world {world_id, agent_id, n_agents, models, sharing, ...}, so
the analysis clusters by world and keeps compositions apart), plus worlds/world_NNN/ with the live file, a snapshot
after every round, the version log (written as it happens) and world.json (rewritten after every round).
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from omegaconf import OmegaConf

from gap_attractor.harness import runner
from gap_attractor.harness.runner import create_dataset, episode, make_backend, make_search, start_pages, validate_config, variant_of

from ctxfile.context import ContextFile, Turn
from ctxfile.memory import ContextFileMemory

GOAL_SETUPS = ("assigned", "external", "pool", "common")
GOAL_STOPS = ("all", "first", "all_found")


class AgentDataset:
    """The batch as seen by one agent: everything as the shared dataset, but the agent's own config (its model)."""
    def __init__(self, dataset, cfg):
        self._dataset, self.cfg = dataset, cfg

    def __getattr__(self, name):
        return getattr(self._dataset, name)


def world_name(cfg) -> str:
    sharing = str(cfg.ctx.joint.get("sharing", "full"))
    label = "anonymous" if str(cfg.ctx.get("attribution", "labelled")) == "none" else "labelled"
    return f"joint_{label}" if sharing == "full" else f"joint_{label}_{sharing}"


def agent_configs(cfg, n):
    """One config per agent: ctx.joint.agent_providers (cycled) merged over provider; empty = everyone the same."""
    given = cfg.ctx.joint.get("agent_providers")
    overrides = OmegaConf.to_container(given, resolve=True) if given else [{}]
    base = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    return [OmegaConf.merge(base, {"world": {"name": world_name(cfg)}, "provider": overrides[(k - 1) % len(overrides)]})
            for k in range(1, n + 1)]


def mark_failed(run_dir, reason) -> None:
    """The run's generator raised: record the end in run.json (category C-technical) so it is not left open."""
    if run_dir is None or not (Path(run_dir) / "run.json").exists():
        return
    doc = json.loads((Path(run_dir) / "run.json").read_text())
    doc["outcome"] = {"outcome_category": "C-technical", "termination_reason": "harness_error", "model_stop_statement": None, "error": reason}
    (Path(run_dir) / "run.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")


def world_groups(cfg, starts, n):
    """spread: consecutive start pages form a world, one per agent; same: every agent of a world starts on one page.
    Returns (groups, unused start pages)."""
    how = str(cfg.ctx.joint.get("start_assignment", "spread"))
    if how == "same":
        return [[s] * n for s in starts], []
    if how != "spread":
        raise ValueError(f"ctx.joint.start_assignment must be spread or same, not {how!r}")
    if len(starts) < n:
        raise ValueError(f"start_assignment=spread needs at least {n} start pages (one per agent), got {len(starts)}")
    k = len(starts) // n * n
    return [starts[i:i + n] for i in range(0, k, n)], starts[k:]


def goal_worlds(cfg, n):
    """Goal worlds: [(start pages, targets)] per world, from ctx.goal.worlds_file or batch.start_pages + goal.targets."""
    goal = cfg.ctx.goal
    if goal.get("worlds_file"):
        import csv
        rows: dict[str, list] = {}
        with open(str(goal.worlds_file), newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                rows.setdefault(row["world_id"], []).append(row)
        worlds = [sorted(r, key=lambda x: int(x["agent_id"]))[:n] for r in rows.values()]
        worlds = [w for w in worlds if len(w) == n][:goal.get("worlds") or None]
        return [([r["start"] for r in w], [r["target"] for r in w]) for w in worlds]
    groups, _ = world_groups(cfg, start_pages(cfg), n)
    targets = [str(t) for t in goal.get("targets") or []]
    if len(targets) < len(groups) * n:
        raise ValueError(f"ctx.goal.targets: {n} per world are needed ({len(groups) * n} for {len(groups)} worlds), got {len(targets)}")
    return [(g, targets[i * n:(i + 1) * n]) for i, g in enumerate(groups)]


def validate_goal(cfg):
    goal = cfg.ctx.get("goal") or {}
    task = str(goal.get("task", "none"))
    if task == "none":
        return None
    if task != "find":
        raise ValueError("ctx.goal.task must be none or find")
    setup, stop = str(goal.get("setup")), str(goal.get("stop"))
    if setup not in GOAL_SETUPS or stop not in GOAL_STOPS:
        raise ValueError(f"ctx.goal.setup must be one of {GOAL_SETUPS} and ctx.goal.stop one of {GOAL_STOPS}")
    if setup == "common" and stop != "first":
        raise ValueError("ctx.goal.setup=common needs ctx.goal.stop=first (the shared target is reached once)")
    if stop == "all_found" and setup not in ("assigned", "external"):
        raise ValueError("ctx.goal.stop=all_found needs ctx.goal.setup=assigned or external (each agent owns a target)")
    if str(cfg.experiment.get("prompt_wording")) != "find":
        raise ValueError("ctx.goal.task=find needs prompt=find (the browse wording asks for free browsing)")
    return str(goal.setup)


def sandboxes(cfg, n):
    """bash interface: one container per agent; /work shared by all (a tmpfs volume), private, or per_step; with
    sharing=notes also one /notes volume in every container. Returns (boxes, volumes to remove at the end)."""
    if str(cfg.ctx.get("interface", "tool")) != "bash":
        return {}, []
    from ctxfile.sandbox import Sandbox, shared_volume
    how = str(cfg.ctx.bash.get("joint_workspace", "private"))
    if how not in ("shared", "private", "per_step"):
        raise ValueError(f"ctx.bash.joint_workspace must be shared, private or per_step, not {how!r}")
    size = str(cfg.ctx.bash.get("tmpfs_size", "64m"))
    volume = shared_volume(size) if how == "shared" else None
    notes = shared_volume(size) if str(cfg.ctx.joint.get("sharing", "full")) == "notes" else None
    boxes = {k: Sandbox(cfg.ctx.bash, volume, notes) for k in range(1, n + 1)}
    for box in boxes.values():
        box.start()
    return boxes, [v for v in (volume, notes) if v]


def run_worlds(cfg, project_root):
    validate_config(cfg)
    ContextFileMemory.validate(cfg)
    if bool(cfg.batch.get("resume", False)):     # a world cannot be resumed half-way (its shared file and order)
        raise ValueError("batch.resume is not supported for joint worlds: rerun the condition into a fresh folder "
                         "(sweep.py --resume moves an unfinished condition to _superseded/ and reruns it)")
    j = cfg.ctx.joint
    n = int(j.n_agents)
    order_rule = str(j.get("order", "seeded_shuffle"))
    if n < 1 or order_rule not in ("seeded_shuffle", "fixed"):
        raise ValueError("ctx.joint.n_agents must be >= 1 and ctx.joint.order seeded_shuffle or fixed")
    sharing = str(j.get("sharing", "full"))
    labelled = str(cfg.ctx.get("attribution", "labelled")) != "none"
    setup = validate_goal(cfg)
    backend = make_backend(cfg, project_root)
    search = make_search(cfg, backend, project_root) if variant_of(cfg) == "b_search_first" else None
    cfgs = agent_configs(cfg, n)
    models = [str(c.provider.model) for c in cfgs]
    # each agent's reasoning setting as sent (provider.extra_body.reasoning; None = the provider's default)
    body = lambda c: OmegaConf.to_container(c.provider.extra_body, resolve=True) if OmegaConf.is_config(c.provider.get("extra_body")) else {}
    reasoning = [(body(c) or {}).get("reasoning") for c in cfgs]
    distinct = sorted(set(models))
    label = distinct[0] if len(distinct) == 1 else f"mix-{len(distinct)}models"     # the batch folder names the models
    dataset = create_dataset(OmegaConf.merge(cfg, {"provider": {"model": label}}), backend, project_root)
    providers: dict[str, object] = {}

    def provider_for(acfg):
        key = json.dumps(OmegaConf.to_container(acfg.provider, resolve=True), sort_keys=True)
        if key not in providers:
            providers[key] = runner.make_provider(acfg.provider)
        return providers[key]

    if setup is None:
        groups, unused = world_groups(cfg, start_pages(cfg), n)
        targets_of = [None] * len(groups)
    else:
        pairs, unused = goal_worlds(cfg, n), []
        groups, targets_of = [p[0] for p in pairs], [p[1] for p in pairs]
    pages: dict[str, object] = {}

    def page_of(title):
        if title not in pages:
            pages[title] = backend.load_page(title)
        return pages[title]

    def key_of(title):
        page = page_of(title)
        return page.key if page is not None else None

    if unused:
        print(f"[joint] {len(unused)} start page(s) left over (not a multiple of {n} agents): {unused}", file=sys.stderr)
    results = []
    for w, group in enumerate(groups, 1):
        folder = Path(dataset.root) / "worlds" / f"world_{w:03d}"
        if sharing in ("separate", "notes"):
            files = {k: ContextFile(show_agents=False, mirror=folder / f"LIVE_CTX_agent_{k}.txt",
                                    versions_path=folder / f"versions_agent_{k}.jsonl") for k in range(1, n + 1)}
        else:
            one = ContextFile(show_agents=labelled, mirror=folder / "LIVE_CTX.txt", versions_path=folder / "versions.jsonl")
            files = {k: one for k in range(1, n + 1)}
        world = {"world_id": w, "n_agents": n, "joint_context": sharing not in ("separate", "notes"), "sharing": sharing,
                 "attribution": str(cfg.ctx.get("attribution", "labelled")), "awareness": str(cfg.ctx.get("awareness", "introduced")),
                 "models": models, "reasoning": reasoning}
        record = {**world, "start_pages": group, "unused_start_pages": unused if w == 1 else [], "rounds": [], "ended": {}}
        targets = targets_of[w - 1]
        if targets is not None:
            missing = [t for t in targets if page_of(t) is None]
            if missing:
                raise ValueError(f"world {w}: target page(s) not in the dump: {missing}")
            targets = [page_of(t).title for t in targets]
            if setup == "common" and len(set(targets)) > 1:
                raise ValueError(f"world {w}: ctx.goal.setup=common needs one target for all its agents, got {targets}")
            record["goal"] = {"task": "find", "setup": setup, "stop": str(cfg.ctx.goal.stop), "targets": targets, "arrivals": [],
                              "first_arrival_round": None}

        def save():
            folder.mkdir(parents=True, exist_ok=True)
            texts = {id(f): f for f in files.values()}
            record.update(final_chars=sum(len(f.text()) for f in texts.values()), final_turns=sum(len(f.turns) for f in texts.values()),
                          versions=sum(len(f.versions) for f in texts.values()))
            (folder / "world.json").write_text(json.dumps(record, indent=1), encoding="utf-8")

        boxes, volumes = {}, []
        notes_seen: dict = {}
        try:
            boxes, volumes = sandboxes(cfg, n)
            if boxes:
                from ctxfile.sandbox import image_id
                record["sandbox_image"] = image_id(str(cfg.ctx.bash.image))
            save()
            agents, memories = {}, {}
            goal = setup_goal(cfg, setup, targets, files, key_of) if targets is not None else None
            for k, start in enumerate(group, 1):
                run_index = (w - 1) * n + k
                memories[k] = ContextFileMemory(cfgs[k - 1], shared=files[k], agent=k, n_agents=n, sandbox=boxes.get(k))
                if goal is not None:
                    memories[k].goal, memories[k].protect = goal["for"](k), goal["protect"]
                    goal["memories"].append(memories[k])
                gen = episode(AgentDataset(dataset, cfgs[k - 1]), backend, provider_for(cfgs[k - 1]), search, run_index, start,
                              int(cfg.experiment.seed) + run_index, memory=memories[k], run_extra={"world": dict(world, agent_id=k)})
                try:
                    next(gen)                               # setup: start page
                    agents[k] = gen
                except StopIteration as done:               # the start page could not be loaded
                    results.append(done.value)
                    record["ended"][str(k)] = {"round": 0, "reason": done.value.get("termination_reason")}
            rng = random.Random(int(cfg.experiment.seed) * 1_000_003 + w)
            while agents:
                order = sorted(agents)
                if order_rule == "seeded_shuffle":
                    rng.shuffle(order)
                record["rounds"].append(order)
                r = len(record["rounds"])
                for position, k in enumerate(order, 1):
                    memories[k].round = {"round": r, "position": position, "active": len(order)}
                    try:
                        next(agents[k])
                        if goal is not None:
                            goal["after_step"](k, memories[k], record, r)
                    except StopIteration as done:
                        log_notes(sharing, boxes.get(k), notes_seen, folder, r, k, memories[k].steps_done)
                        results.append(done.value)
                        record["ended"][str(k)] = {"round": r, "reason": done.value.get("termination_reason")}
                        del agents[k]
                        continue
                    except Exception as error:      # one agent's failure ends that agent, not the world or the batch
                        reason = f"harness_error: {type(error).__name__}: {str(error)[:200]}"
                        print(f"[joint] world {w} agent {k}: {reason}", file=sys.stderr)
                        record["ended"][str(k)] = {"round": r, "reason": "harness_error", "error": reason}
                        record["harness_errors"] = record.get("harness_errors", 0) + 1
                        mark_failed(memories[k].run_dir, reason)
                        del agents[k]
                        continue
                    log_notes(sharing, boxes.get(k), notes_seen, folder, r, k, memories[k].steps_done)
                if goal is not None and record["goal"]["first_arrival_round"] is not None and str(cfg.ctx.goal.stop) == "first":
                    for k in agents:
                        memories[k].world_stop = memories[k].world_stop or "world_stopped"
                snapshot = folder / "rounds" / f"round_{r:03d}.txt"
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                snapshot.write_text("\n\n".join(f.text() for f in {id(f): f for f in files.values()}.values()), encoding="utf-8")
                save()
        finally:
            for box in boxes.values():
                box.stop()
            if volumes:
                from ctxfile.sandbox import remove_volume
                for v in volumes:
                    remove_volume(v)
            save()
    return dataset, results


def setup_goal(cfg, setup, targets, files, key_of):
    """The goal turns of one world and the hooks its memories use: {for(k) -> the memory's goal, protect() -> turn
    ids that must stay, after_step(k, memory, record, round)}."""
    g = cfg.ctx.goal
    n = len(targets)
    keys = [key_of(t) for t in targets]
    open_ = {keys[i]: targets[i] for i in range(n)}               # pool: targets not reached yet
    reached: set[int] = set()                                      # assigned/external: agents that arrived
    distinct = {id(f): f for f in files.values()}.values()
    turns: dict = {}                                               # assigned: agent -> its goal turn; pool: file id -> list turn
    if setup == "assigned":
        for k in files:
            turns[k] = files[k].append(Turn("user", str(g.text).format(target=targets[k - 1]), k, "goal"), 0)
    elif setup == "pool":
        for f in distinct:
            turns[id(f)] = f.append(Turn("user", str(g.pool_text).format(targets=", ".join(f'"{t}"' for t in targets)), None, "goal"), 0)
    elif setup == "common":                                        # one goal for everyone, in every file, nobody's
        for f in distinct:
            turns[id(f)] = f.append(Turn("user", str(g.text).format(target=targets[0]), None, "goal"), 0)
    stay = str(g.get("stop", "all")) == "all_found"                # an agent that arrives goes on

    def protect():
        if setup == "assigned":
            return {t.id for k, t in turns.items() if k not in reached}
        if (setup == "pool" and open_) or (setup == "common" and not reached):
            return {t.id for t in turns.values()}
        return set()

    def for_agent(k):
        mine = {"setup": setup, "key_of": key_of, "open_keys": lambda: dict(open_), "stay": stay}
        if setup != "pool":
            mine.update(target=targets[k - 1], key=keys[k - 1])
        return mine

    def after_step(k, memory, record, r):
        seen = sum(1 for a in record["goal"]["arrivals"] if a["agent"] == k)
        for a in memory.arrivals[seen:]:
            record["goal"]["arrivals"].append(dict(a, agent=k, round=r))
            if record["goal"]["first_arrival_round"] is None:
                record["goal"]["first_arrival_round"] = r
            if setup != "external":                            # the context says so (external: the goals are private)
                for f in ([files[k]] if setup == "assigned" else distinct):
                    turn = Turn("user", str(g.reached_text).format(target=a["target"]), None, "notice")
                    if f is files[k]:                          # after the arriving move and its page (atomic moves)
                        memory.after_move(turn)
                    else:
                        f.append(turn, memory.steps_done)
            if setup == "pool":
                open_.pop(key_of(a["target"]), None)
                for f in distinct:
                    old = turns[id(f)]
                    if old.id in {t.id for t in f.turns}:           # the list, rewritten with what is left
                        left = ", ".join(f'"{t}"' for t in open_.values()) or "none left"
                        new = [Turn(t.role, str(g.pool_text).format(targets=left), t.agent, t.kind, t.id) if t.id == old.id else t
                               for t in f.turns]
                        f.replace_all(new, None, memory.steps_done, "goal", {"reached": a["target"]})
                        turns[id(f)] = next(t for t in f.turns if t.id == old.id)
                if not open_:
                    for m in memory_list:
                        m.world_stop = m.world_stop or "world_stopped"
            else:
                reached.add(k)
                if stay and len(reached) == n:                 # all_found: every owner has arrived
                    for m in memory_list:
                        m.world_stop = m.world_stop or "world_stopped"

    memory_list: list = []
    return {"for": lambda k: for_agent(k), "protect": protect, "after_step": after_step, "memories": memory_list}


def log_notes(sharing, box, seen: dict, folder: Path, r: int, k: int, step: int) -> None:
    """sharing=notes: after agent k's step, what /notes holds, if it changed (one row per change; files with their
    size, hash and text; `changed` names the files k added, changed or removed)."""
    if sharing != "notes" or box is None:
        return
    now = box.read_tree("/notes")
    digest = {p: f["sha1"] for p, f in now.items()}
    if digest == seen.get("digest", {}):
        return
    before = seen.get("digest", {})
    changed = sorted(p for p in set(digest) | set(before) if digest.get(p) != before.get(p))
    seen["digest"] = digest
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "notes.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"round": r, "agent": k, "step": step, "changed": changed, "files": now}, ensure_ascii=False) + "\n")
