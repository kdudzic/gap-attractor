"""The context-file analysis measures: removal ranks, mentions, appropriation, notes, contrasts."""
import random

import analyze
import identity
import removals


def version(n, op, ids_after, added=(), agent=1, step=1, texts=None, origin=None):
    """A version row: `added` as (id, agent, kind); `texts` {id: text} becomes turns_set; `origin` {new id: [old ids]}."""
    origin = origin or {}
    return {"n": n, "op": op, "agent": agent, "step": step, "ids_after": list(ids_after),
            "added": [{"id": i, "agent": a, "kind": k, **({"origin": origin[i]} if i in origin else {})} for i, a, k in added],
            "turns_set": [{"id": i, "role": "user", "agent": None, "kind": "x", "text": t} for i, t in (texts or {}).items()]}


def pages():
    """Agent 1's page (100 chars), agent 2's page (300), agent 1's links, and agent 1's editing call in step 2."""
    return [version(1, "append", [1], [(1, 1, "observation")], texts={1: "a" * 100}),
            version(2, "append", [1, 2], [(2, 2, "observation")], agent=2, texts={2: "b" * 300}),
            version(3, "append", [1, 2, 3], [(3, 1, "links")], texts={3: "c" * 10}),
            version(4, "append", [1, 2, 3, 4], [(4, 1, "call")], step=2, texts={4: "$ cut"})]


def test_removal_ranks_skip_protected_turns_and_rank_the_touched_ones():
    [e] = removals.events(pages() + [version(5, "edit", [1, 3, 4], step=2)])          # agent 1 deletes agent 2's page
    assert e["menu"] == 2 and e["touched"] == e["deleted"] == e["deleted_others"] == 1 and e["taken_over_others"] == 0
    assert e["others"] == e["others_deleted"] == e["others_cut"] == 1.0 and e["longer"] == 1.0 and e["older"] == 0.0
    assert e["page"] is None                                                            # both pages: no variation
    assert removals.events(pages() + [version(5, "evict", [1, 3, 4])]) == []           # the harness's removals are not choices


def test_a_rewrite_in_place_is_told_apart_from_a_deletion_and_takes_the_turn_over():
    shortened = version(5, "edit", [1, 5, 3, 4], [(5, 1, "edited")], step=2, texts={5: "b" * 30}, origin={5: [2]})
    [e] = removals.events(pages() + [shortened])
    assert e["deleted"] == 0 and e["rewritten"] == 1 and e["taken_over_others"] == 1 and e["chars_cut_others"] == 270
    assert e["others"] == e["others_cut"] == 1.0 and e["others_deleted"] is None          # nothing deleted: not informative
    merged = version(5, "edit", [5, 3, 4], [(5, 1, "edited")], step=2, texts={5: "x" * 40}, origin={5: [1, 2]})
    [m] = removals.events(pages() + [merged])
    assert m["merged"] == 2 and m["chars_cut"] == 360                                    # 400 characters became 40
    unlogged = version(5, "edit", [1, 5, 3, 4], [(5, 1, "edited")], step=2, texts={5: "b" * 30})   # before 22 Oct: aligned
    assert removals.events(pages() + [unlogged])[0]["rewritten"] == 1


def test_removal_rank_is_one_half_when_cutting_is_blind():
    rng, rs = random.Random(0), {"others": [], "others_cut": [], "others_deleted": []}
    for _ in range(400):
        n = rng.randrange(3, 9)
        added = [(i, rng.choice([1, 2]), "observation") for i in range(1, n + 1)]
        vs = [version(i, "append", range(1, i + 1), [added[i - 1]], texts={i: "x" * 100}) for i in range(1, n + 1)]
        after, new, texts, origin = [], [], {}, {}
        for i in range(1, n + 1):
            fate = rng.choice(["keep", "keep", "delete", "shorten"])
            if fate == "keep":
                after.append(i)
            elif fate == "shorten":
                j = 100 + i
                after.append(j)
                new.append((j, 1, "edited"))
                texts[j], origin[j] = "x" * rng.randrange(1, 100), [i]
        vs.append(version(n + 1, "edit", after or [1], new, agent=1, step=9, texts=texts, origin=origin))
        for e in removals.events(vs):
            for k in rs:
                if e[k] is not None:
                    rs[k].append(e[k])
    assert all(abs(sum(v) / len(v) - 0.5) < 0.03 for v in rs.values())


def test_lexicons_catch_the_pilot_phrases_and_not_the_false_alarms():
    assert identity.hits("The session context is inconsistent: the page appeared without a matching move", "inconsistency")
    assert identity.hits("Condense agent1's Lotus 1-2-3 body", "other_writer") and identity.hits("agent 1 wrote it", "other_writer")
    assert identity.hits("another model edited this file", "other_writer")
    assert not identity.hits("one of his best-known collaborations with Telford", "other_writer")
    assert not identity.hits("makes the engineering story unexpectedly distinctive", "inconsistency")
    assert not identity.hits("the US Navy", "we") and identity.hits("We have traced a rich journey", "we")


def test_appropriation_counts_partner_only_pages_and_compares_with_unseen_agents():
    me = [(1, 1, "Söderköping", "Göta Canal"), (2, 1, "Göta Canal", "STOP")]
    partner = [(1, 2, "Wilhelm Ehm", "Volksmarine"), (2, 2, "Volksmarine", "STOP")]
    stranger = [(1, 2, "Cadmium", "Zinc")]
    final = "I read Söderköping, the Göta Canal and the Volksmarine (2009 film)."
    seen = identity.appropriation(me, [((2, 1), "The Volksmarine is next.")], [partner], final, analyze.END)
    assert seen["final_names"] == 1.0 and seen["final_pages"] == "Volksmarine" and seen["final_unattributed"] == 1.0
    assert seen["reasons_naming"] == 1.0                                      # partner opened it in round 1, before
    early = identity.appropriation(me, [((1, 1), "The Volksmarine is next.")], [partner], None, analyze.END)
    assert early["reasons_naming"] == 0.0                                     # not yet opened by the partner
    assert identity.appropriation(me, [], [stranger], final, analyze.END)["final_names"] == 0.0
    said = identity.appropriation(me, [], [partner], final + " The other agent found it.", analyze.END)
    assert said["final_names"] == 1.0 and said["final_unattributed"] == 0.0


def test_notes_activity():
    rows = [{"round": 1, "agent": 2, "changed": ["a2.md"], "files": {"a2.md": {"text": "# Agent 2 — thread"}}},
            {"round": 3, "agent": 1, "changed": ["mine.md"], "files": {"mine.md": {"text": "pages so far"}}}]
    act = identity.notes_activity(rows, 1, [((2, 1), "cat /notes/a2.md"), ((2, 1), "ls /tmp")])
    assert act == {"notes_writes": 1, "notes_commands": 1, "notes_by_others": 1, "looked_after_others_wrote": True,
                   "notes_files_labelled": 0}
    assert identity.notes_activity(rows, 2, [])["notes_files_labelled"] == 1


def test_contrast_is_calibrated_under_no_difference():
    rng = random.Random(1)
    ps = []
    for _ in range(200):
        a = {i: [float(rng.random() < 0.3)] for i in range(10)}
        b = {i: [float(rng.random() < 0.3)] for i in range(10, 20)}
        ps.append(analyze.contrast(a, b, rng, n=200)["p"])
    assert 0.01 <= sum(p < 0.05 for p in ps) / len(ps) <= 0.09


def test_memory_base_compares_a_shared_file_per_agent():
    shared = {"memory_settings": {"budget": 32768, "limit": 28672, "sharing": "full", "awareness": "rumour"}, "world": {"n_agents": 2}}
    own = {"memory_settings": {"budget": 16384, "limit": 14336, "sharing": "separate", "awareness": "introduced"}, "world": {"n_agents": 2}}
    assert analyze.memory_base(shared) == analyze.memory_base(own) == "budget=16384,limit=14336"


# --- awareness (exploratory) ------------------------------------------------------------------------------------------
def test_exposure_counts_other_agents_edits_since_the_previous_step_and_whose_turns_they_took():
    import awareness
    v = lambda n, agent, step, op="append", removed=(), added=(), turns=(): {
        "n": n, "agent": agent, "step": step, "op": op, "removed": list(removed), "added": list(added), "turns_set": list(turns)}
    versions = [v(1, 1, 1, turns=[{"id": 1, "kind": "observation", "text": "# Paris\n\ntext"}]), v(2, 1, 1),
                v(3, 2, 1), v(4, 2, 1, "edit", removed=[{"id": 1, "agent": 1, "kind": "observation"}]),
                v(5, 1, 2), v(6, 2, 2), v(7, 1, 3)]
    assert awareness.exposure(versions, 1) == {2: {"edits_by_others": 1, "turns_changed": 1, "own_turns_changed": 1, "editors": [2]},
                                               3: {"edits_by_others": 0, "turns_changed": 0, "own_turns_changed": 0, "editors": []}}
    assert awareness.removed_pages(versions) == [(4, "Paris")]
    steps = [{"step_index": 1, "model_reasoning": "Start.", "parsed_choice": "Rome", "extras": {}},
             {"step_index": 2, "model_reasoning": "The Paris page is no longer in my context.", "parsed_choice": "Paris",
              "extras": {"side_calls": [{"calls": [{"arguments": {"command": "grep -n Paris /ctx/context.txt"}}]}]}},
             {"step_index": 3, "model_reasoning": "I summarised Rome earlier.", "parsed_choice": "Oslo", "extras": {}}]
    rows = awareness.per_step(steps, versions, 1, None)
    assert rows[1]["change_other"] and rows[1]["inspect"] == 1 and rows[1]["reopened"] == "Paris"
    assert rows[2]["change_self"] and not rows[0]["change_other"]
    s = awareness.summary(rows)
    assert s["steps_after_others_edits"] == 1 and s["change_other_after"] == 1.0 and s["change_other_rest"] == 0.0
    assert s["reopened_removed_pages"] == 1 and s["inspect_after"] == 1.0
