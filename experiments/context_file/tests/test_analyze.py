"""analyze.py on scripted single and joint runs: every table is written and the numbers are those of the logs."""
import csv

import analyze
import run as entry
from gap_attractor.harness import runner

from helpers import Scripted, call, config, reply


def table(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_analyze_single_and_joint(tmp_path, monkeypatch):
    compact = lambda m: reply(call("edit_context", 1, replace_turns=[{"first": 1, "last": 4, "summary": "Saw a page."}]))
    single = Scripted([reply(call("follow_link", 0, title="Archive")), compact])
    monkeypatch.setattr(runner, "make_provider", lambda cfg: single)
    entry.run(config(tmp_path / "runs" / "single", ["experiment.condition=d_budget", "ctx.tokenizer=chars"]))
    providers = {"m1": Scripted(model="m1"), "m2": Scripted(model="m2")}
    monkeypatch.setattr(runner, "make_provider", lambda pcfg: providers[str(pcfg.model)])
    entry.run(config(tmp_path / "runs" / "joint", ["ctx=joint", "experiment.condition=j_full", "experiment.step_limit=2", "ctx.tokenizer=chars",
                                                   "batch.start_pages=[Civic Atlas,Archive,Bridge,Library,Market,Museum]",
                                                   "ctx.joint.agent_providers=[{model: m1},{model: m2}]"]))
    analyze.main([str(tmp_path / "runs"), "--out", str(tmp_path / "out")])
    runs = table(tmp_path / "out" / "runs.csv")
    assert len(runs) == 7 and {r["condition"] for r in runs} == {"d_budget", "j_full"}
    one = next(r for r in runs if r["condition"] == "d_budget")
    assert one["edits_applied"] == "1" and float(one["shrink_per_applied_edit"]) > 0 and one["memory"].startswith("context_file|")
    conditions = table(tmp_path / "out" / "conditions.csv")
    assert {c["world"] for c in conditions if c["condition"] == "j_full"} == {"joint_labelled|n=3|m1x2+m2"}
    assert {c["model"] for c in conditions if c["condition"] == "j_full"} == {"world:m1+m2"}      # a mixed world is not split
    assert {c["step_limit"] for c in conditions} == {"3", "2"} and all("context_budget_share" in c for c in conditions)
    survival = table(tmp_path / "out" / "survival.csv")
    assert {r["ending"] for r in survival if r["condition"] == "d_budget"} == {"all", "step_limit"}
    s = next(r for r in survival if r["condition"] == "d_budget" and r["ending"] == "all")
    assert s["page_turns"] == "3" and s["removed"] == "1" and s["removed_by_self"] == "1" and s["surviving_10_steps"] == "0.5"
    conv = table(tmp_path / "out" / "convergence.csv")
    assert conv and conv[0]["worlds"] == "2" and conv[0]["p"]
    exposure = table(tmp_path / "out" / "exposure.csv")
    assert exposure[0]["worlds"] == "2" and exposure[0]["unseen_baseline"] != ""
    assert (tmp_path / "out" / "summary.md").read_text().startswith("# Context-file conditions")
    rem = {(r["condition"], r["score"]): r for r in table(tmp_path / "out" / "removals.csv")}
    assert rem[("d_budget", "older")]["edits"] == "1" and rem[("d_budget", "others")]["edits"] == "0"   # one agent: no others
    agents = table(tmp_path / "out" / "identity_agents.csv")
    assert len(agents) == 7 and {a["task"] for a in agents} == {"browse"}
    assert {a["reasoning"] for a in agents} == {"default", "default+default+default"}            # a world: each agent's
    assert all("appropriation_final_names_unseen" in a for a in agents if a["condition"] == "j_full")   # worlds of 2+ agents


def test_survival_accounts_for_censoring():
    """Review N6: the same removal hazard must give the same curve for short and long runs."""
    import random
    def runs(length, hazard=0.2, n=3000, seed=1):
        rng, rows = random.Random(seed), []
        for _ in range(n):
            start = rng.randrange(length)
            life = next((t for t in range(1, length - start + 1) if rng.random() < hazard), None)
            rows.append({"lifetime": life, "removed": True} if life is not None else {"lifetime": length - start, "removed": False})
        return analyze.kaplan_meier(rows)
    short, long = runs(8), runs(30)
    assert abs(analyze.surviving(short, 3) - analyze.surviving(long, 3)) < 0.03 and abs(analyze.surviving(long, 3) - 0.8 ** 3) < 0.03


def test_convergence_null_keeps_world_size_and_is_calibrated():
    """Review N3: under no effect, p < 0.05 about 5 % of the time; null worlds have as many agents as real ones."""
    import random
    rng = random.Random(0)
    def world():
        return {k: {f"p{rng.randrange(60)}" for _ in range(8)} for k in (1, 2, 3)}
    ps = [analyze.convergence({w: world() for w in range(12)}, random.Random(r), draws=200)["p"] for r in range(150)]
    assert 0.01 <= sum(p < 0.05 for p in ps) / len(ps) <= 0.12
    two = analyze.convergence({w: world() for w in range(2)}, random.Random(0), draws=200)
    assert two["p"] > 0.2                                                     # 2 worlds of 3: no spurious effect


def test_exposure_baseline_comes_from_agents_that_could_not_be_seen():
    walk = lambda pages: [(r, 1, pages[r - 1], pages[r]) for r in range(1, len(pages))]
    worlds = {"w1": {1: walk(["a", "x", "y"]), 2: walk(["b", "y", "z"])},       # agent 1 picks y after agent 2 visited y
              "w2": {1: walk(["c", "q", "r"]), 2: walk(["d", "s", "t"])}}
    rows = {(r["world"], r["slot"]): r for r in analyze.exposure(worlds)}
    assert rows[("w1", 1)]["observed"] == 0.5 and rows[("w1", 1)]["baseline"] == 0.0
