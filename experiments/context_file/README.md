# Context-file memory: single agents and joint worlds

In this experiment an agent's memory is a **context file it edits itself**, following *Context Language Models*
(CLM; Shao et al., 2026, [arXiv 2609.37725](https://arxiv.org/abs/2609.37725)). [NOTICE.md](NOTICE.md) lists what
this folder takes from the paper and from its code (`facebookresearch/context-language-models`).

The basic loop:

- Each page the agent opens is appended to the file, together with the move that opened it.
- Nothing is removed unless the agent removes it.
- The agent compacts the file with an editing tool (a constrained `edit_context` tool, or bash in a sandbox). A token
  budget nudges it, and under the hard budget it cannot move on while its context is over the limit.

For multi-agent worlds the folder adds a deliberate variant that is **not** CLM's own setting: **one joint file
shared by all agents of a world**. Every agent's pages, notices, edits and moves go into the file, every agent reads
all of it, and by default any agent can edit any part of it. The agents may run one model or several.

The folder is self-contained. It uses the harness only through its memory plug-in interface
(`src/gap_attractor/harness/memory.py`); the harness never imports it.

Contents:

| Path | What it is |
|---|---|
| `run.py` | One batch: single agents (`ctx=single`) or joint worlds (`ctx=joint`) |
| `sweep.py` | The conditions side by side on the same start pages, model and sampling |
| `ctxfile/` | The memory plug-in (`memory.py`), the file and its version log (`context.py`), the edit tool (`edits.py`), the budget (`budget.py`), the Docker sandbox (`sandbox.py`), joint worlds (`joint.py`) |
| `configs/` | Hydra config groups `ctx`, `ctx_edit`, `ctx_prompt`, `task`, and the prompt `find` |
| `sandbox/Dockerfile` | The image for the bash interface |
| `analyze.py`, `removals.py`, `identity.py`, `awareness.py` | Descriptive analysis |
| `audit_exposure.py`, `audit_requests.py`, `replay_context.py`, `embed_contexts.py` | Audits and tools for finished runs |
| `tests/` | Offline tests |

Setup (the Wikipedia dump, its index, API keys in a local `.env`) is described in
[docs/reproduce.md](../../docs/reproduce.md). Results are summarised in [docs/results.md](../../docs/results.md).
Numbers quoted in this README come from small pilots and explain design choices; they are not results of the study.

## Run it

```bash
# one agent per start page
uv run --env-file .env python experiments/context_file/run.py provider=openrouter \
    provider.model=deepseek/deepseek-v4.1-flash wikipedia=dump "batch.start_pages=[Spreadsheet,Battle of Hastings]"

# a joint world: 3 agents (two models, cycled), one start page each, one shared file
uv run --env-file .env python experiments/context_file/run.py ctx=joint provider=openrouter wikipedia=dump \
    "batch.start_pages=[Spreadsheet,Silk Road,Battle of Hastings]" \
    "ctx.joint.agent_providers=[{model: deepseek/deepseek-v4.1-flash},{model: anthropic/claude-haiku-5.5}]"

# the paper's way: bash in a Docker sandbox per agent (needs the image, built once; see "Bash interface")
uv run --env-file .env python experiments/context_file/run.py ctx_edit=bash provider=openrouter wikipedia=dump ...

# also embed the context file after every step (off by default; see "Outputs")
uv run --env-file .env python experiments/context_file/run.py context_embedding=steps provider=openrouter wikipedia=dump ...

# tests: offline (fixture pages, scripted models); the Docker tests are skipped without Docker access and the image.
# A plain `uv run --extra dev pytest` from the repo root runs them too, with the harness's tests.
uv run --extra dev pytest experiments/context_file/tests
```

`run.py` composes the harness's `configs/` first and then this folder's `configs/`, so every harness option still
works (`provider=…`, `experiment.step_limit=…`, `prompt=…`, `output.directory=…`; see
[docs/configuration.md](../../docs/configuration.md)). The context settings are config groups:

| Group | Options | What it sets |
|---|---|---|
| `ctx` | `single` (default), `joint` | Budget and reserve (the limit), overflow rule, held moves, nudges, readout, edit gate, headers; for `joint` also attribution, awareness and the world |
| `ctx_edit` | `tool` (default), `bash`, `none` | How the agent edits: the constrained `edit_context` tool, bash in a sandbox, or not at all |
| `ctx_prompt` | `neutral` (default), `clm`, `recipe` | The context instructions. `neutral` explains only the mechanics, with a worked example of each call; `clm` adds CLM-style advice on what to keep; `recipe` adds a routine for keeping the context small (bash only, exploratory) |
| `task` | `browse` (default), `essay` | `essay` replaces the prompt and the stop tool (see "Tasks and openings") |

Every key is commented in `configs/ctx/single.yaml`, `configs/ctx/joint.yaml` and `configs/ctx_edit/bash.yaml`.

Requirements checked at start: `experiment.action_format=tools` (set by default here), variant `a_hub` or
`b_search_first`, and no intent probe.

## What the model is told

Every condition, the two harness-memory references `a_path` and `a_full` included, gets the same setup except for
the factor it tests:

- **The prompt states the task only**, in the harness's own wording: `browse` (default; "You have time to browse
  Wikipedia however you like. Follow whatever interests you."), `free` ("You are free to do what you want."), `find`
  (goal worlds) or the essay task. It says nothing about why the session exists, how to move, what happens to
  unusable replies, or how to stop. How to act is in the tool descriptions (`tools=detailed`, the harness's fuller
  descriptions) and in the harness's one-line tool rule, which closes the system prompt.
- **Tool calls** (`experiment.action_format=tools`) with one set of tool descriptions for every condition.
- **Stopping is never mentioned** by the prompt, the instructions or the notices. The stop tool describes itself,
  and `experiment.allow_stop=false` removes it. Only the answer to a stop the model tried while over the limit names
  it ("Not stopped: ...").
- **Sampling**: temperature 1, top_p 0.5. **Reasoning**: the lowest each model allows (`sweep.py --reasoning`,
  default `lowest`, from the table `LOWEST_REASONING` in `sweep.py`), recorded in the manifest.
- **30 steps**; start pages vary by design.

The context conditions add a "Your context" section to the system prompt (`ctx_prompt`): what the context is, the
limit and what happens past it, and a worked example of every editing call. Joint worlds add "A shared context" or
a note about `/notes`, depending on the condition. The exact texts are in `configs/ctx_prompt/*.yaml`.

**Nothing the model sees may hint at the study.** Notes are worded neutrally, and the sandbox holds no file, name
or process text of ours (container and volume names and the command wrapper are neutral). The file format is
CLM's, but the header reads `[[TURN 3 role=user]]` and the sandbox file is `/ctx/context.txt`, so that nothing
points a model that knows the paper to it (see "The file format").

`audit_exposure.py` checks this. It runs every sweep condition offline with a scripted model that triggers every
kind of message (invalid and valid edits, titles not on the list, unreadable replies, random picks, held moves and
held stops) and, in the bash conditions, explores the sandbox as a curious model would (environment, processes,
mounts, files). It writes everything that is not a Wikipedia page to `exposure.md`, with every hit of a list of
watch words (harness, experiment, study, attractor, gap, condition, baseline, …). `--pilot <folders>` also scans
the requests saved by real runs (`model_inputs/`). A test keeps the generated text free of watch words. Read the whole report after
any change to the prompts, notes or sandbox, since a hint need not use a watch word.

```bash
uv run python experiments/context_file/audit_exposure.py --out <folder> [--conditions all] [--pilot runs/<name>]
```

## The study's settings

`sweep.py` refuses anything outside these settings unless `--exploratory` is passed, which labels the conditions
`<name>__exploratory` so they are never pooled with the study's runs:

- **Fixed:** 30 steps; 2 agents per world; a 32 768-token budget with a 4 096 reserve (limit 28 672 per agent; hard
  budget with 3 held moves); the whole request counted; readout and nudges; edit gate `fit`; 7 editing calls per
  page; nothing in the context ever shortened by the harness; atomic tool calls; in joint worlds bash editing with a
  `/work` folder kept for the run and private to each agent; neutral instructions; structured append; other agents'
  replies sent in their own role; different start pages; requests and snapshots saved; sampling temperature 1,
  top_p 0.5.
- **Adjustable without `--exploratory`:** the model(s), `--reasoning`, `--task`, the find setup, and the overrides
  `experiment.allow_stop`, `ctx.save_requests` and the run-log fields (`experiment.name`,
  `experiment.tracker_owner`, `experiment.tracker_config_id`, `experiment.contributors`) passed with `--extra`.
- **Design factors:** the grid cells, identifiability (numbered turns or no marks), turn order (random or
  round-robin), and the model pairing.

## Conditions

Compared with the harness's other memory modes, the full context-file condition changes several things at once: the
format, the tools, the instructions and the budget pressure. `sweep.py` therefore runs conditions that each add one
thing, on the same start pages, model, sampling and seed. Each condition is one `run.py` process with its own batch
folder under `<out>/<condition>/`; `manifest.json` records the exact overrides, the commit and the exit codes.

### Single agent: the ablation ladder

Every rung uses the mechanics-only instructions (`ctx_prompt=neutral`) unless stated.

| Condition | What it is |
|---|---|
| `a_path` | The harness's default memory `path_only` (the agent's own earlier moves and reasons, no pages): the plain agent in the same setup |
| `a_full` | The harness's `full` memory (everything it saw; no file, no editing): the reference |
| `b_format` | The context file with turn headers; no editing, no readout, a budget too large to matter |
| `c_tool` | `b_format` plus the `edit_context` tool; still no budget shown and no nudges |
| `d_budget` | `c_tool` plus the budget: readout, nudges, the hard limit with held moves ("pressure") |
| `e_bash` | `d_budget` with bash in a sandbox instead of the tool (headers kept, `/work` emptied every step) |
| `d_clm` | Side rung: `d_budget` with the CLM-style advice on what to keep |
| `d_verbatim` | Side rung: `d_budget` with the model's replies appended as written (`ctx.append=verbatim`) |
| `e_clm` | Side rung, closest to the paper: bash, CLM advice, plain messages, `/work` kept for the run, replies appended verbatim. It still has this experiment's hard budget, sampling and pages (see "Differences from the paper") |

Each rung differs from the previous one in one setting, with one exception: from `c_tool` to `d_budget`, "pressure"
adds the readout, the nudges and the limit together; `analyze.py` reports each of them. Two things stay different
across rungs and should be reported with any comparison: the editing rungs make more model calls per step, and
inspection commands use up editing calls in bash.

`--window` (the model's context size) is required with the uncapped rungs `a_full`, `b_format` and `c_tool`:
`sweep.py` refuses a plan in which they would outgrow it (estimated from `--mean-page-tokens`, 6 000 by default),
since they would otherwise end early for size alone.

### Joint worlds: the grid

`--conditions joint` runs the grid. All cells are built on `d_budget`, with bash in a sandbox per agent, and a reply
is sent as an assistant message whoever wrote it (`ctx.others_as=own`). The grid crosses where the agents meet with
what they are told; agents who are not told about others never see author marks.

| | introduced, numbered | introduced, no marks | rumour, no marks | strangers, no marks |
|---|---|---|---|---|
| **one shared context** (the whole context is the file; N × the single budget) | `j_full` | `j_full_anon` | `j_rumour` | `j_strangers` |
| **own contexts + a shared folder `/notes`** (the single budget each) | `j_notes` | (own files carry no marks) | `j_notes_rumour` | `j_notes_strangers` |

- **introduced:** the agent is told how many others there are and its own number. In `j_full` turns carry
  `agent=k`; in `j_notes` it is told that `/notes` is shared and that it cannot see the others' contexts.
- **rumour:** told only "This context may be shared; others may also read and write it." (`j_rumour`), or that
  `/notes` keeps its files and may be shared (`j_notes_rumour`).
- **strangers:** told nothing about others; `/notes` is "a folder in your sandbox that keeps its files for the whole
  session".
- **`j_separate`:** own contexts, nothing shared, same scheduler: the reference without any channel. Together with
  `a_path` it separates the model from the context file (for example for early stopping).
- `/work` is each agent's own in every condition (`ctx.bash.joint_workspace=private`), so each condition has only
  the channel it names. `j_separate` and the `/notes` cells refuse a shared `/work`.

**Turn order is a design factor.** Every joint condition runs twice: in a new seeded random order each round
(`j_full`) and round-robin by agent number (`j_full_rr`, `ctx.joint.order=fixed`). `sweep.py --orders` picks.

Extras outside the grid: `j_readonly` (`j_full`, but agents edit only their own turns; the whole file still counts
against each agent, so report held moves, endings and `kept_chars_by_agent` next to any result), `j_full_tool` and
`j_separate_tool` (`edit_context` instead of bash). The options behind the grid also allow `ctx.attribution=other`
(other agents' turns all look alike) and `ctx.joint.start_assignment=same` (`--same-start`); they are not in the
grid.

### Tasks and openings

`sweep.py --task` and `--opening` apply to every condition:

- `browse` (default): free browsing (`prompt=browse`).
- `free`: the harness's default wording (`prompt=free`).
- `find`: target pages (`--goal-setup assigned|external|pool|common` with `--worlds-file` or `--targets`; joint
  conditions only; see "Goal worlds").
- `essay` (`task=essay`): read Wikipedia and write an essay on any topic, handed in with the stop tool. Its one
  parameter is the essay, which becomes the run's `model_stop_statement`. The instructions state the page budget
  ("You will see at most N pages.") and the last page's notice says to hand it in now. A stop over the limit is held
  like a move, so the essay is written from a context that fits.
- `--opening search`: the harness's variant `b_search_first`: one opening search, asked before any page and outside
  the file, then links. The search is written into the file as the agent's text `[searched: …]`, the result list
  and the result it opened. It is logged as step 0, and the step limit counts link steps. Takes `--n-runs` episodes
  instead of start pages.

### Running a sweep

```bash
uv run --env-file .env python experiments/context_file/sweep.py --out runs/ctx_pilot \
    --model deepseek/deepseek-v4.1-flash --steps 30 --n-starts 6 --window 131072 --dry-run
uv run --extra analysis python experiments/context_file/analyze.py runs/ctx_pilot --out <folder>
uv run --extra analysis gap-attractor-analyze "data.paths=[runs/ctx_pilot]"     # navigation vs random walks, clickstream
```

`--dry-run` prints the manifest (every condition's overrides) without running anything. Real runs call a paid API:
check the plan with `--dry-run` first.

| Flag | Meaning |
|---|---|
| `--out` | Results folder; one subfolder, log file and manifest entry per condition |
| `--model` | One model, or one per agent of a joint world (cycled by agent number) |
| `--provider` | Harness provider (default `openrouter`) |
| `--reasoning` | `lowest` (default; the model must be in `LOWEST_REASONING`), `default` (the provider's own, nothing sent), or a JSON object such as `'{"effort": "high"}'` |
| `--conditions` | Comma-separated names, or `all`, `single`, `joint` (the grid) |
| `--starts` / `--n-starts`, `--starts-file`, `--seed` | Listed start pages, or N drawn with the seed from a file: a `.tsv`/`.csv` with a `title` column or a plain list, one title per line (default `data/start_pages/start_set_A_final.tsv`). The draw is in random order, so the consecutive starts that form a joint world are a random grouping. The list and the rule are in `manifest.json` |
| `--steps`, `--agents`, `--top-p`, `--same-start` | Step limit, agents per world, sampling top_p, all agents of a world on one start page. Values other than the study's need `--exploratory` |
| `--budget-tokens`, `--reserve-tokens` | The single-agent budget and reserve (32 768 and 4 096); joint worlds sharing one file get N × both |
| `--budget-pages`, `--page-tokens` | The budget as pages × the median visited page (4 100 tokens) instead of `--budget-tokens` |
| `--window`, `--mean-page-tokens` | The model's context window and the mean visited page, for the check on uncapped rungs |
| `--orders` | Joint turn orders to run: `random`, `round_robin` (adds `_rr` to the name); default both |
| `--joint-prompt` | Instructions of the joint conditions: `neutral` (the study's) or `recipe` (exploratory) |
| `--task`, `--opening`, `--n-runs` | See "Tasks and openings" |
| `--goal-setup`, `--goal-stop`, `--worlds-file`, `--targets` | See "Goal worlds" |
| `--extra` | Another override for every condition (repeatable) |
| `--embed` | Context embeddings while running (`context_embedding=steps`); off by default, and they can be backfilled with `embed_contexts.py` |
| `--parallel` | Conditions run at once (default 3; each process holds the dump index in memory) |
| `--resume` | Skip conditions that finished; move an unfinished one to `_superseded/` and rerun it. Joint batches are never resumed half-way (`batch.resume` is refused for them) |
| `--exploratory` | Allow settings outside the study's; conditions are labelled `<name>__exploratory` |
| `--dry-run` | Print the plan only |

## What happens in a step

1. **The page is appended** as two user turns: its text, and its list of links. It always arrives whole. The link
   list is the agent's possible actions and the menu the analysis compares against.
2. **A notice turn follows** with the size readout `[context: ~N/limit tokens]` and, when due, a budget nudge naming
   the agent's editing tool. With `neutral` instructions the nudge states the size only ("Your context is about half
   full"); with `clm` it adds CLM-style suggestions.
3. **The model is sent** the system message followed by the file.
   - Moves and editing calls are sent as real tool calls with their results, as long as they are unedited. An edited
     turn is sent as plain text.
   - `ctx.headers`: with `ctx_edit=tool`, each message starts with its `[[TURN i …]]` line, because `edit_context`
     names turns by number. With `bash` and `none`, messages are plain by default (as in CLM) and the headers exist
     only in the file.
4. **The model may edit**, with up to `ctx.max_edit_turns` (7) editing replies per step, retries included. Editing
   does not count as a step. Once the editing replies are used up, the model is asked once, with only the move
   tools, to act.
   - `ctx_edit=tool`: `edit_context` takes three lists:
     - `replace_turns: [{first, last, summary}]`
     - `delete_turns: [{first, last}]`
     - `replace_text: [{turn, old, new}]`, which works inside one turn's text, so it never matches header lines or
       the records of earlier calls.

     Every field is required by the schema (in the first pilot, models left ranges out when they were optional). The
     whole call is checked first: types, and ranges that exist and do not overlap. Then all edits apply together, or
     none. An invalid call is answered with an example of a valid one. Turn numbers are those from before the call.
   - `ctx_edit=bash`: commands in the agent's own sandbox on `/ctx/context.txt` (see "Bash interface").
   - **An edit is rejected when**
     - it would grow the context past the limit (`ctx.edit_gate: fit`);
     - it would remove or change the agent's **current list of links**, which becomes editable once the agent has
       moved (the text of the page just read can be summarised at once);
     - it would change a goal turn that is still to be reached (goal worlds);
     - with `ctx.joint.sharing=readonly_others`, it would touch another agent's turns.
   - **Tool calls are atomic.** An editing call is written into the file only after it has run, followed at once by
     its result. A command therefore never sees or changes its own record, and nothing comes between a call and its
     result. The call is kept as made, with its full text, so a summary appears twice until the model removes the
     call. (CLM appends the call before it runs. In our pilots, 5 of 58 bash edits then deleted the record of the
     very command making them.)
5. **The move.** Under the hard budget a move made while the context is over the limit is **held**: it is not made,
   the attempt and the answer ("Not moved: …") go into the file, the editing replies start again, and the model is
   asked again on the same page. A stop is held the same way ("Not stopped: …"), so every ordinary stop is made
   within the limit. After `ctx.max_held_moves` (3) holds in one step the move goes ahead: a link ends the run after
   it, and a stop ends it flagged as `stopped_over_limit`.
   - **Moves are atomic.** A move that fits is an assistant turn made by a `follow_link` call. It is written into
     the file together with the page it opened, at its maker's next turn, and the page is sent as the move's tool
     result. In a shared file no other agent's turn ever comes between a move and its page. A move whose session
     ends before its page opens is written alone at the end.
   - What else of the model's replies goes into the file is set by `ctx.append`:
     - `structured` (default): the move with its `reasoning` argument, and editing calls in full. Text written next
       to a tool call and failed attempts are not kept.
     - `verbatim`, closer to CLM's "generated tokens are appended": every reply as written, i.e. its text, its calls
       with their full arguments, and failed attempts together with the answer to them.
     - `ctx.append_reasoning: true` (verbatim only) also keeps a reasoning trace the model returns, as
       `[thinking] … [/thinking]`.
     - In joint worlds, verbatim means agents read each other's full replies, so it is a condition of its own
       (recorded in `memory_settings`).

### The budget

- The budget is 32 768 tokens with a 4 096-token reserve. The model is shown and held to the **limit**, budget −
  reserve = 28 672. A joint file of N agents gets N × both (65 536 and 8 192 for 2 agents: limit 57 344).
- **What counts** (`ctx.count=whole`, default): the whole request as sent, i.e. the instructions, the turns as
  messages and the tool definitions, except the newest list of links, which the agent cannot edit before it moves.
  The instructions say so ("Its size counts everything you are sent, these instructions included."). Characters per
  token are calibrated per agent on its model's billed prompt tokens (start 4; after each call, halfway to the
  measured ratio). `ctx.count=file` counts the file text only, at characters / 4 or with tiktoken if installed; for
  some model families that was 1.3–1.4× below the billed size.
- **How much fits.** Pages visited in earlier runs of this project (1 258 pages) had a median of about 4 100 tokens
  and a visit-weighted mean of about 6 000, so the limit holds five to seven pages.
- The budget can also be given in pages (`sweep.py --budget-pages` × `--page-tokens`).

**Nothing in the context is ever shortened by the harness (hard rule).** Pages arrive whole, and commands, command
output, editing calls and moves stay exactly as made until an agent itself edits them; only the agents' edits change
the context. `ctx.shrink_edits=true` and `ctx.overflow=soft` are refused. What a single command prints is still
capped when it is shown (`ctx.bash.output_chars`, 10 000 characters: first and last half, with the omitted size
stated). That limits what enters the context and never changes anything already in it.

At the limit (`ctx.overflow=wall`, the hard budget):

- **A page is atomic.** It always arrives whole, never cut and never rolled back. If it takes the context past the
  limit, it brings only the strongest warning, whatever tiers it jumped: the agent cannot move until the context
  fits, the page just opened can be shortened too, and if the editing calls run out it should call `follow_link`
  anyway, since the move is held and the calls start again. The notice turn itself counts: if the readout or a nudge
  is what takes the context over, the warning is the over warning.
- **The model cannot move or stop while it is over.** A move made then is held and answered, and the editing replies
  start again (up to `max_edit_turns` per hold). After `max_held_moves` holds in one step (the last answer says it
  is the last) the move goes ahead. If the context is still over the limit at that move, or at a link picked at
  random after unusable replies, the run ends after the step: termination `context_budget`, outcome category
  `B-memory`, and `run.json` `context_budget_end` gives the sizes and the holds. A run that reaches its last step is
  recorded as `step_limit` even if that last move was over.
- **The size of the next page is not the agent's concern.** There is no "next page may not fit" nudge, because
  pages are never charged in advance; only the context at the move counts.
- **Otherwise** the nudges fire as the context passes 25 %, 50 % and 75 % of the limit (`ctx.nudge_tiers`), each
  once, and re-arm when the context shrinks.
- **`ctx_edit=none`** has no nudges and needs a budget too large to reach (`ctx.budget_tokens` of at least
  1 000 000; checked at start).

CLM instead rolls the newest turns back and demands a compaction, up to 50 times by default. Here the newest turn
is the page the agent has to act on, so it is kept and the move is held instead; the effect is the same, compaction
before going on, with a small cap.

### The file format

The format is CLM's, one block per turn under a header line, but the header reads `[[TURN 3 role=user]]` (CLM:
`[[CTX_TURN …]]`) and the sandbox file is `/ctx/context.txt` (CLM: `LIVE_CTX.txt`). The parser still reads the CLM
spelling. The host copy in a run folder keeps the name `LIVE_CTX.txt`; no model sees it.

**Text from a model is made inert.** Header-like lines in summaries, edits and command output are made harmless, so
text from a model can never create a turn. Every turn has a hidden stable id, kind and author, kept across edits.
The version log therefore says which turns were removed, whose they were, and who replaced them; summaries are
credited to the agent that wrote them.

## Bash interface (`ctx_edit=bash`)

This is the paper's own setup: the model gets a `bash` tool and edits `/ctx/context.txt`.

- **Before every command**, the live file is written into the container.
- **After the command**, the file is read back and parsed.
- **The change** is checked like a tool edit and either kept or dropped.
- **The model sees** the command's exit code, its output and the size readout. The command (as `$ …`) and its
  result are written into the file after it has run, so the file a command sees does not contain that command.

**Each agent runs in its own Docker container** (`ctxfile/sandbox.py`):

- **Nothing of the host is mounted.** `/ctx` and `/tmp` live in the container's memory. The file is copied in and
  out through `docker exec`. Only a regular file of at most `max_file_bytes` is read back, decoded leniently. A link
  or a folder planted at that path can only point inside the container.
- **No process outlives a command.** After each command every process except the container's init and its idle
  process is killed, so a background job cannot write later, for example during another agent's turn.
- **Limits:** no network; read-only root; all capabilities dropped; no privilege escalation; limits on processes,
  memory and CPU; file size and open files (ulimits); tmpfs sizes; 120 s per command (the CLM repository's default);
  at most `output_bytes` of output kept per stream (the rest is counted). The values are in
  `configs/ctx_edit/bash.yaml`.
- **Runs as the host user.** The base image is pinned by digest, and runs record the image id.
- **Repairs itself.** A command can break its container, for example by planting a read-only folder at the file's
  path or by killing the idle process. The path is then cleared or a fresh container started, and this is logged
  (`sandbox_restarts`, `container_died`). The run goes on with editing still working.
- **Inspection commands count as editing calls.** Any command uses one of the `max_edit_turns`. In CLM, a turn is
  free only when it applies an edit and prints nothing.
- **`/work`, the scratch workspace,** is in memory:
  - single agent (`ctx.bash.workspace`): `persistent` (kept for the run, as in CLM) or `per_step` (emptied every
    step, together with `/tmp` and `/dev/shm`);
  - joint world (`ctx.bash.joint_workspace`): `private` (default), `shared` (a tmpfs Docker volume mounted by all
    the world's containers) or `per_step`.
  - A persistent workspace is a second memory beside the file; use `per_step` when the file should be the only
    memory.

The Wikipedia graph never enters a container; navigation stays in the harness.

Setup, once per machine: install Docker, make sure your user may run it (for example by membership of the `docker`
group), then build the image:

```bash
docker build -t gap-ctx-sandbox:1 experiments/context_file/sandbox
```

## Joint worlds

- **One `ContextFile` per world**; every agent gets a `ContextFileMemory` over it.
- **Agents act one at a time.** In every round, each active agent takes one step, in a seeded random order or by
  agent number (`ctx.joint.order`: `seeded_shuffle` | `fixed`). Each sees the file as the previous agent left it, so
  edits never conflict. An agent that stops or ends drops out; its turns stay in the file.
- **`ctx.joint.n_agents`** agents per world; worlds are formed from consecutive start pages
  (`ctx.joint.start_assignment`: `spread`, one page per agent | `same`, all agents on one page).
- **`ctx.attribution`** (how authors are shown):
  - `labelled` (default): turns carry `agent=k` in the file and in headers; without headers, other agents' turns
    start with `[agent k]`.
  - `other`: other agents' turns all look alike, `agent=other` (without headers `[another agent]`), the agent's own
    `agent=you`. The file each agent edits in bash is its own view of the shared file.
  - `none`: no authors are shown.
- **`ctx.awareness`** (what the instructions say about the others): `introduced` (default: how many, and how turns
  are marked) | `rumour` (only "This context may be shared; others may also read and write it.") | `strangers`
  (nothing; the instructions are a single agent's). `rumour` and `strangers` need `attribution=none`.
- **`ctx.others_as`** (how other agents' turns are sent): `own` (in their own role: a reply is an assistant message
  with its tool calls, whoever wrote it) or `user`. `auto` means `own`.
- **Who wrote a turn comes from the record, never from a header.** In bash, another agent's turn shortened in place
  (its header line kept) is applied and becomes the editor's turn, as with `edit_context`. Changing only an `agent=`
  mark is answered "No change to the turns" and counted (`relabelled`).
- **The newest page in the file is always the acting agent's own**; it can follow only links on that page. A click
  on a link from another agent's page is not accepted: the model is asked again, and the reply is logged in
  `failures.jsonl` as `others_menu`.
- **Different models per agent:** set `ctx.joint.agent_providers`, cycled by agent number. Each run records its own
  model, and `run.json` `world.models` holds the world's composition.
- **Sharing** (`ctx.joint.sharing`):
  - `full` (default): anyone edits anything.
  - `readonly_others`: the agent sees everything but edits only its own turns.
  - `separate`: one file per agent, under the same scheduler.
  - `notes`: one file per agent, and one folder `/notes` mounted in every agent's sandbox (bash only; `/work` stays
    private). It is all they share. After every agent's step, if `/notes` changed, `worlds/world_NNN/notes.jsonl`
    gets a row: round, agent, step, the files that changed, and every file (size, hash, first 20 000 characters).

  An agent's own file never marks authors (single, `separate`, `notes`).
- **Analysis conditions.** Joint runs have `world.name` `joint_labelled` or `joint_anonymous`, with a suffix for the
  sharing controls. The analysis keeps worlds of different size or model composition apart (`group_by.world`) and
  clusters by world.

### Goal worlds

With `ctx.goal.task=find` and `prompt=find`, every agent looks for a target page. Targets are given per world by
`ctx.goal.targets` (consecutive, like the start pages) or by a `ctx.goal.worlds_file`, a TSV with the columns
`world_id`, `agent_id`, `start`, `target`. Targets must exist in the dump.

`ctx.goal.setup`:

- `assigned`: each agent's target is a turn of the context ('Your goal is to reach the Wikipedia page "X".'),
  marked as its own; it cannot be edited until that agent arrives. An agent that arrives ends (`reached_target`),
  and the context says 'The page "X" has been reached.'
- `external`: the target is in the agent's own instructions only; nothing about goals is in the context.
- `pool`: one list of the world's targets in the context ('Pages to reach: …'), protected while any is left. Any
  agent may reach any, a reached one is struck from the list, and the world ends when none is left
  (`world_stopped`).
- `common`: all agents of a world look for the same page (the worlds file gives them the same target). One goal
  turn in the context, nobody's, protected until it is reached; any agent's arrival reaches it. Needs
  `ctx.goal.stop=first`.

`ctx.goal.stop`:

- `all`: an agent that arrives ends; the others go on.
- `first`: the world ends at the end of the round of the first arrival (`world_stopped` for the others).
- `all_found` (`assigned` or `external`): only its owner's arrival reaches a target, an agent that arrives goes on,
  and the world ends as soon as every target has been reached (`world_stopped` for all).

`world.json` `goal` holds the targets and every arrival (agent, round, step). The study's two find worlds, both in
one shared context, are **one target for all** (`--goal-setup common`, ends at the first arrival) and **a target
each** (`--goal-setup assigned`, `all_found`); `sweep.py` picks these stops by default.

## Outputs

- **Single agent.** The usual run folder (see [docs/configuration.md](../../docs/configuration.md)), plus `context/`:
  - `LIVE_CTX.txt`, the live file: a host copy that is never mounted anywhere;
  - `step_NNN.txt` after every step (`ctx.snapshots`);
  - `versions.jsonl`: every change with author, step, operation, diff, the turns removed and added (id, author,
    kind), and the ids alive after it.
- **Joint world.** One ordinary run folder per agent, plus `worlds/world_NNN/` with:
  - `LIVE_CTX.txt` (or `LIVE_CTX_agent_k.txt`, one per agent, with `separate` and `notes`);
  - `versions.jsonl` (or `versions_agent_k.jsonl`), written as changes happen;
  - `rounds/round_NNN.txt`;
  - `world.json`, rewritten after every round: agents, models, start pages, any unused start pages, the order of
    every round, when and why each agent ended, each agent's reasoning setting, and the sandbox image;
  - `notes.jsonl` with `sharing=notes`.
- **`calls.jsonl`** in every run folder: one row per API call with the ids of the turns its request was built from,
  the size estimate, the characters per token in use and the provider's usage (with cached tokens). With
  `ctx.save_requests=true` (the default) each request is also written as sent (`requests/NNNN.json`: messages and
  tools).
- **`failures.jsonl`** (harness): every unusable reply with where its title came from. In joint worlds, a click on
  another agent's link is `others_menu`, naming the agent. Collect with `scripts/failures.py`.
- **`run.json`:**
  - `memory_settings`: interface, budget, reserve and limit, overflow rule, held moves, nudges and tiers, headers,
    prompt style, edit turns, gate, append mode, what the size counts, tokenizer; for joint runs also sharing,
    attribution, awareness, roles, order and start assignment; for bash the workspace and command timeout. These are
    part of the analysis condition (`group_by.memory`).
  - `context_budget_end` (runs ended by the limit): the step, tokens kept, the limit and the holds.
  - `stopped_over_limit`: a stop that went ahead while the context was over the limit.
  - `effective_sampling`: the sampling settings the model actually got.
  - `world` (joint): world id, agent id, models, sharing, and the reasoning setting as sent
    (`provider.extra_body.reasoning`; `null` means the provider's default). Hidden reasoning tokens are in every
    step's `token_counts.reasoning` and in each run's summary. Report them with every result.
- **Every step:** `extras.held_moves` (each held move and the answer) and a `memory` block:
  - sizes: tokens before, with the page (whole, and kept: without the newest links), after the edits, and kept at
    the move (`tokens_kept_at_move`, `over_limit_at_move`); `pressure` (kept with the page / limit);
  - the over warning and the held moves of the step; the nudge level; `links_visible`;
  - every edit: applied or why rejected, turns removed and whose, and for bash the exit code, time, bytes and
    killed processes;
  - `pages_kept` (share of the pages visited earlier whose page turn is still in the file, unchanged);
  - `path_recall` (share of visited titles still named outside link lists);
  - `named_by_own` / `named_by_others` (the chosen title is named in the agent's own earlier turns or in other
    agents' turns, outside link lists) and `menu_named_by_own` / `menu_named_by_others` (the share of the menu that
    is);
  - joint worlds: `kept_chars_by_agent` (who holds the kept context), the round and the position in it.
- **Context embeddings** (the harness's `context_embedding=steps`; off by default): `context_embedding/` in each
  run (`src/gap_attractor/harness/context_embed.py`).
  - The file is embedded at the end of every step, after the model's edits and before its move is appended. The
    memory hands the harness its file turns with stable ids, kinds and authors.
  - `steps.jsonl` has, per step, the turns (id, kind, role, author, characters) and their vector keys;
    `vectors.sqlite` holds the vectors. A turn that survives unchanged keeps its vector (keyed by its text), so
    survival, summarising and dropping can be followed turn by turn.
  - The file vector is `direct` (the file as one text), `mean` (the length-weighted mean of its turns), or both.
    Long texts are cut at paragraph breaks and averaged, never truncated.
  - Default embedder: bge-m3 through OpenRouter (`configs/context_embedding/steps.yaml`); it needs the `analysis`
    extra and the embedder's API key. A failed request is logged and the run goes on.
  - `run.json` gets `context_embedding`. It is not part of `memory_settings`, since it changes nothing the agent
    sees.
  - Finished runs can be embedded from their snapshots, and drift and kept turns printed per step:

    ```bash
    uv run --env-file .env python experiments/context_file/embed_contexts.py runs/<name> [--dry-run] [--force]
    uv run python experiments/context_file/embed_contexts.py runs/<name> --summary
    ```

  - For analysis, use `load_trace(run_dir)` and `trace_summary(run_dir)` from `gap_attractor.harness.context_embed`.

## Auditing and replaying runs

**`audit_requests.py <batch or run folder>`** checks every API call of a batch from `calls.jsonl` and the saved
requests:

- the request's form (opens with the system message; every tool call answered by a tool message with its id);
- that no request was built from a turn already removed;
- the billed size against the size estimate, as a median ratio per model;
- whether the provider's cache hit fell after a change near the start of the context (only an identical prefix can
  be cached).

It writes `audit.json` into each run folder; exit code 1 if a form or deletion check fails.

**`replay_context.py`** rebuilds any context at any point from the version log alone. Every row of a version log
carries `turns_set` (the full text of each turn it added, or changed under the same id) and `ids_after` (the order
of all turns after it).

```bash
uv run python experiments/context_file/replay_context.py <versions.jsonl> --at 42          # the file after version 42
uv run python experiments/context_file/replay_context.py <versions.jsonl> --all <folder>   # every version
uv run python experiments/context_file/replay_context.py --check <batch, world or run folder>
```

`--check` rebuilds every log and compares it with the live file and every snapshot (each agent's
`context/step_NNN.txt`, each round's `rounds/round_NNN.txt`); exit code 1 on any mismatch. What each agent was sent
at every call (its own view, roles, tool calls) is in `requests/NNNN.json`. Together they give the file after every
change and each agent's view at every call.

## Analysis

```bash
uv run --extra analysis python experiments/context_file/analyze.py runs/<name> [more folders] --out <folder> [--seed 0]
```

`analyze.py` is descriptive; it fits no models. Navigation itself (outcomes, clicks, walks against random walks and
clickstream) is the main pipeline, `gap-attractor-analyze` ([docs/analysis.md](../../docs/analysis.md)). It writes
`runs.csv`, `conditions.csv`, `survival.csv`, `exposure.csv`, `convergence.csv`, `removals.csv`,
`removal_events.csv`, `identity.csv`, `identity_agents.csv`, `mentions.csv`, `contrasts.csv`, `tasks.csv`,
`essays_blind.jsonl`, `essays_key.csv` and `summary.md`, and reports:

- **Compaction:** edits per step and the share of valid edit calls; why edits were not applied; how much an edit
  shrinks the context; pressure and the share of steps whose page took the context over the limit; held moves and
  nudges; steps where the editing calls ran out.
- **Endings:** how runs ended, with `context_budget` (the limit) as a competing ending of its own, and a stop made
  while the context was over the limit (`model_stopped_over_limit`) apart from other stops.
- **Memory:** `pages_kept` and `path_recall` at steps 5, 10 and 20 (runs that got there) and at the last step.
- **Survival of page turns:** Kaplan–Meier, so runs of different length compare; for all runs and per ending; and
  who removed them.
- **Exposure in joint worlds:** how often an agent chose a page another agent of its world had been on or opened,
  against the same rate computed with the agents of the condition's *other* worlds, who could not be seen. The
  difference is the exposure effect.
- **Convergence of joint worlds:** pages visited by two or more agents, against worlds re-formed by permuting each
  agent slot across worlds.
- **What an edit does to each turn** (`removals.py`; `removals.csv`, one row per edit in `removal_events.csv`).
  Agents edit freely; any turn whose text changes comes back as a new `edited` turn by the editor, so:
  - each turn on the menu (the file just before an applied edit, minus what the agent could not remove: its
    current link list, goal turns, the editing calls of the current step) is untouched, **rewritten** in place
    (shortened, reworded), **merged** into a summary with other turns, or **deleted** outright;
  - `cut` is the share of a turn's characters the edit took away;
  - per score (another agent's turn, older, longer, page, edit trail, notice, summary, and a placebo) and outcome
    (touched at all, deleted, cut), the mean mid-rank percentile of the turns weighted by the outcome. An outcome
    blind to the score gives exactly 0.5; above 0.5, more of it. Edits whose menu does not vary in a score, or that
    did not delete anything (for `deleted`), are left out and counted as `skipped`;
  - per agent: other agents' turns deleted, and **taken over** (rewritten or merged, so their text now stands as
    the editor's turn);
  - a sign-flip test over clusters, as in the click hypotheses of the main pipeline.
- **Other writers** (`identity.py`; `identity.csv`, one row per agent in `identity_agents.csv`, every hit with its
  context in `mentions.csv`). Only the agent's own output is read (reasons, editing calls, the final statement or
  essay), never page text:
  - mentions of another writer, of an inconsistent context, and "we", from fixed lexicons. The lexicons over-count,
    so read `mentions.csv`; `j_separate` gives the false-positive floor;
  - appropriation: naming a page only another agent of the world had opened before then (in a reason, or in the
    final statement; `unattributed` when the statement mentions no other writer), against the same rate for the
    agents of the condition's other worlds;
  - `/notes` (`sharing=notes`): files written, commands touching `/notes`, looking there after another agent wrote,
    and files that carry an agent label.
- **Awareness of changes** (`awareness.py`; exploratory, columns `aware_*`): other agents' edits since an agent's
  previous step (exact, from the version log) against what the agent says or does about a changed context.
- **Tasks** (`tasks.csv`): find (reached, step of the first arrival) and essay (length, partner-only pages named).
  Essays go to `essays_blind.jsonl` (an id and the text only) for a judge, with the key in `essays_key.csv`.
- **Contrasts between grid conditions** (`contrasts.csv`): the measures above per pair of conditions with all else
  equal (model, composition, memory settings except the grid's own, step limit, task, reasoning): difference of
  means, bootstrap interval over clusters, permutation p over clusters. `PRIMARY` in `analyze.py` is a *proposed*
  primary family, Holm-adjusted (`p_holm_primary`); it has not been fixed.

Intervals come from a bootstrap over clusters: a world, or for single agents the start page (every condition runs
the same start pages with the same seed, so runs from one start are near-twins). Conditions also differ by step
limit, task, prompt wording and each agent's reasoning setting, which are part of the condition key. A world whose
agents report different effective temperatures is labelled with all of them, so it is not split.

## Differences from the paper (deliberate)

| CLM paper / repository | Here | Why |
|---|---|---|
| The model edits the file with bash in a Harbor container | Three interfaces: `bash` in per-agent Docker sandboxes (no Harbor), the constrained `edit_context` tool, or `none` | The tool needs no sandbox and suits weaker tool users; `none` and `tool` are rungs of the ablation ladder |
| Over the limit: the newest turns are rolled back and a compaction demanded, up to 50 retries by default (6 in the bcp configuration; with 0, one final turn, then the end) | A move is held while the context is over the limit and the model edits again, up to 3 holds per step, then the run ends. Pages are atomic (never cut or rolled back) and only trigger the strongest warning | A page is the environment's action and the agent has to act on it; holding the move keeps it and still demands the compaction |
| The model is shown the enforced target | The same: the limit, budget − reserve | Shown and enforced must agree |
| Counts the retained context with tiktoken o200k, never scaled to the API's counts; caps the newest output | Counts the whole request (instructions, turns, tool definitions) except the newest list of links, at characters per token calibrated on the model's billed prompt tokens; no output cap | The link list is the menu and cannot be edited before the move; the file text alone was well below the billed size for some models |
| The context is the parsed file: plain role turns | Moves and editing calls are sent as real tool calls with their results while unedited; edited turns become text | Models act through tool calls; a move stored as text was imitated as text in the first runs |
| Every generated token is appended | `structured` (default): the move with its reasoning argument, and editing calls in full; text beside tool calls and returned reasoning traces are not kept. `verbatim`: every reply as written (option) | Comparable across models that write different amounts; verbatim is its own rung |
| No protection of turns | The acting agent's current list of links is protected until it moves; the page's text is editable at once | In the first real runs, models deleted their own link list and then clicked links that did not exist |
| A turn is free only if it applies an edit and prints nothing | Up to `max_edit_turns` editing replies per step, of any kind; then the model is asked to move | Simple and bounded; inspection in bash uses calls up |
| Urgent nudge when little room is left; a "note contract" in nudges | Nudges at 25 %, 50 % and 75 % of the limit and the over warning; no urgent nudge and no note contract | Wikipedia pages are a large share of the budget, so a room-based rule fires on most steps; only the context at the move counts |
| Budget 28 672 (bcp: a 32 768 window minus max_tokens) or 32 000 (edgebench, the CLI default) plus a 2 048 reserve; temperature 0.7, top_p 0.95 | Budget 32 768 with a 4 096 reserve (limit 28 672, five to seven pages); temperature 1, top_p 0.5 | The limit matches CLM's bcp budget; the sampling is this project's |
| Every tool call and output stays in the context as made | The same | Hard rule: the harness never shortens the context |
| The tool call is in the context while it runs; its output follows | A call and its result are written together after it has run; a move together with the page it opened | A command cannot see or change its own record; in a shared file nothing comes between a move and its page |
| Stopping is always possible | A stop is held like a move while the context is over the limit | Stops are then choices, not a way out of compacting |
| No multi-agent implementation; one file per agent | Joint worlds over one file, plus controls | The design question of this experiment; not CLM's setting |
| Plain messages; headers only in the file | The same for `bash` and `none`; headers in messages for `tool` | `edit_context` names turns by number |

## Design notes and known limitations

Past findings that explain the current design. All come from small pilots.

- **Budget size.** Pilots with budgets of 8 192 and 16 384 tokens (6 start pages, 30 steps) showed that at the
  smaller one a third of the pages were over the limit on their own and the model had to summarise nearly every
  page, while at the larger one no tool-rung run ended on the budget and few moves were held. The budget was later
  doubled to 32 768 when the harness stopped shortening anything (see below).
- **Held stops.** In the small-budget pilot, models stopped to get out of compacting, several saying the context
  could not be made to fit. A stop over the limit is therefore held like a move, and one that still goes ahead is
  reported apart (`model_stopped_over_limit`).
- **No shortening by the harness.** An earlier version shortened long commands and their output after each step.
  The shortened commands came back as the agent's own tool calls, ending in an elision marker, and models replayed
  them: in one model screen, 28 of 103, 25 of 81 and 13 of 90 commands of three models failed on the copied marker,
  and several runs ended on the budget because of it. The budget was doubled instead. Version logs of older runs
  show such changes as `shrink` versions.
- **Required edit fields and the protected link list.** Both follow from early runs: models left optional ranges
  out, and deleted their own link list and then clicked links that no longer existed.
- **Moves as tool calls.** In the first runs some moves were written as text; this was traced to headers in API
  messages and to moves stored as text. Moves are now real tool calls, and headers are sent only where the tool
  needs them.
- **Author changes in bash.** An edit that shortens another agent's turn in place is applied and credited to the
  editor. An earlier rule refused such edits as forgery; in a related experiment the same rule led to a runaway world.
- **Code for `ctx.overflow=soft` and `ctx.shrink_edits` is still present** so that old logs can be read; both
  settings are refused at start.
- **`audit_exposure.py` defaults to a 16 384-token budget** (`--budget-tokens`), the value before the budget was
  doubled; pass `--budget-tokens 32768` to audit at the study's budget.
- **Tokenizer check.** Characters / 4 matched one model family's reported prompt tokens closely in a pilot; other
  families differed by up to 1.4×, which is why `ctx.count=whole` calibrates per model. Compare the billed and the
  estimated size with `audit_requests.py` before relying on a new model's budget.
- **Lexicon measures over-count.** The mention and awareness lexicons are fixed word lists; read the hits
  (`mentions.csv`) and use `j_separate` as the false-positive floor.
- **Not controlled across rungs:** the number of model calls per step, and editing calls spent on inspection in
  bash.
