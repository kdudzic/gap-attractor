# Gap Attractor: Wikipedia Experiment Sketch

Dated 2026-09-14. Preliminary design doc. Expect it to change.

## Definitions

- **Curiosity**: preference for depth-first instead of breadth-first exploration. The agent quickly
  nails down an interesting strand of hyperlinks and then "goes down a rabbit hole" of a certain
  semantic content cluster.
- **Attractor**: a page, topic, or cluster of topics a given agent regularly gravitates toward over
  N steps of exploration, recurring across several hyperparameter configurations and ablation
  conditions.
- **Run**: a single starting prompt followed by N jumps through a hyperlink.

## Questions ("what do we want to know?")

- **Q1**: Do models have certain "hardly predictable" (in a vague sense) stable states or interests
  they converge toward regardless of prompt, persona or task, given a relatively large context
  length? How environment-dependent is the behavior? All models have seen Wikipedia and know it
  exists.
- **Q2**: Do sufficiently frontier models have information-gathering convergent dynamics, and how
  often do they prioritize gathering information about the pieces missing from pretraining data?
  Model-specific wiki pages are generally created after the model release, so we need to
  distinguish interest towards itself from interest towards unknown information gain.

## Hypotheses ("so what?")

- **H1**: Attractors exist.
- **H2**: Attractors are topics and not pages. A given LLM agent converges on the same topic
  clusters but different specific pages across runs, so attractors live in the model's semantic
  space.
- **H3**: Attractor sets map into several "exploration styles": explorer (pure novelty-seeker, see
  H6), researcher, bored agent, no persona, etc.
- **H4**: Agents prefer pages that contain knowledge they possess relatively less of, versus agents
  gravitate toward common pages (AI, consciousness, LLMs, etc.).
- **H5**: Attractors hold across ablations (see Ablations).
- **H6**: Mathematical exploration dynamics: trajectory entropy decays over a run (settling into an
  attractor), attractor stability increases with model scale, etc. Sergei could help here the most.
- **H7**: Agent stated versus revealed preferences concerning exploration paths can be different.
- **H8**: Agents do not have a recency bias concerning hyperlink position in the pages.
- **H9**: Agents have certain model-induced biases that are present independently of the persona,
  and might drift towards these interests despite the persona being activated.
- **H10**: Models might traverse towards information they haven't seen in pretraining, for example
  pages added after the training cutoff, and be less curious about information they already "know"
  from pretraining. If a page was updated after the cutoff date, and this can be learned without
  viewing the page directly, they might be more interested in collecting the missing pieces.

## Setup

### Environment

- Local dump of the English Wikipedia with the hyperlink structure retained.
  - Articles only, current revisions, compressed (`pages-articles.xml.bz2`): about 25 GB as of
    September 2026, no media.
  - Downloadable from <https://en.wikipedia.org/wiki/Wikipedia:Database_download> and
    <https://meta.wikimedia.org/wiki/Data_dump_torrents>.
- Alternatively: dynamic Wikipedia access with a hyperlink-only graph and on-the-fly fetching of
  page contents from the internet.
- Current modelling stance: treat Wikipedia as a **hypergraph**, with topics and tags (categories)
  as hyperedge specifiers. This might change later.

### Harness

The harness is the computational infrastructure that lets us conduct experiments. Details are left
to the Harness team.

### Logger

The logger records all the important information about agent behavior during runs:

- Direct page path taken
- Agent thought process
- Graph representation of the traversal of Wikipedia's topic structure
- Numerical metrics, e.g. distance from the start node, topic diversity, traversal directions,
  average page popularity

### Prompts

- "You are free to do what you want"
- Explicit task to explore or follow its interests
- Just the contents of the starting page
- Explanation of the environment, in several variants with a varying level of detail
- Any of the above with the number of available steps specified ("You can jump through a link N
  times")
- Last prompt: ask the model at the last step why it made its choices. Open decision: do we want
  this?

### Hyperparameters

- Exploration steps (with or without an exit button)
- Temperature
- Top-p (sampling method: default, min-p, greedy)
- Reasoning effort
- Seed?

### Ablations

- Starting node (either a fixed start page or randomly sampled nodes)
- Prompt (essentially the harness and tool use description, which is already a lot of context)
- Context window on or off, i.e. whether we clear the context on each new step
- Masking echo attractor hyperlinks: will the agents find their "personal" attractors?
- Masking traces of the model itself from the contents of Wikipedia: will, e.g., GPT-5 behave
  differently if it can't get info about itself from the GPT page?
- Masking top attractors identified early: will the agents gravitate to semantically similar pages?
- Task modulation: will the agents still prefer content semantically similar to their attractor
  even when given a task?

## Why conduct both experiments (Wikipedia and TextWorld)?

- The studies act as controls to each other. If we ran only one, a reviewer might say that with a
  different definition of curiosity we wouldn't have found anything.
- We can evaluate whether attractor states in general are environment specific. Maybe exploring
  Wikipedia creates no attractor states but exploring TextWorld does, or the opposite, or both.
- If we do Wikipedia first, "attractor clues" can become a variable in the TextWorld experiment that
  would trigger the agents' curiosity regarding the attractors identified for them in the Wikipedia
  part. Would they gravitate toward them?
- One of the experiments may turn out to be impossible to conduct the way we want. Then we still
  have the second to fall back on.

## Relevant papers

- *Evaluating Large Language Models on Wikipedia Graph Navigation: Insights from the WikiGame*
  (Margiotta et al., 2025). General inspiration for navigating Wikipedia with LLMs, though the LLMs
  are given a very specific task.
- *Intrinsically motivated graph exploration using network theories of human curiosity*
  (Patankar et al., 2023). Sketches two formal models of curiosity in a graph-structured data
  exploration task.
- *LLM-WikiRace Benchmark: How Far Can LLMs Plan over Real-World Knowledge Graphs?*
  (Ziomek et al., 2026). A complete Wikipedia exploration harness for agents and a complete
  benchmark. Could be used as the control task.
- *Textual understanding boost in the WikiRace* (Ebrahimi et al., 2025). Several simpler-than-LLM
  approaches to Wikipedia navigation in defined tasks. One could be a baseline for the control task.

## Team views on the design

| Name | What is the design | What we want out of it |
|---|---|---|
| Sebastian | Agent searches through Wikipedia without any explicit directions | Understand what topics the models gravitate towards and analyze any patterns they show when exploring |
| Sachin | Agent traversing the local wiki dump; we specify some tools and rules and analyse the patterns | Understand the latent preference of the model |
| Gagana | Several tools connected to the agent; it goes through the local files (checkpoints included, to see if it actually goes into each file) and updates its knowledge base | Pages visited and lines read are logged to analyse its actions |
| Pooja | Agent searches for knowledge without us telling it what to do | See what the agent searched and possibly find a pattern per model |
| Casper | Agents exploring a static copy of the Wikipedia "graph". Curiosity and attractors measured from node-jumping dynamics | CHARACTERIZE each agent based on its activity. Are there ANY regularities in how it traverses the graph over multiple runs with modified conditions? |
| Claire | Agents given enough tools and freedom to explore Wikipedia pages freely; we observe whether there are patterns in their behaviour | Find whether there is a topic they are most interested in and visit the most |
| Yuliya | A few states (with different prompts) for a single agent, with variables such as prompt, task or no files, persona or no persona. Local Wikipedia base. Logging all agent moves and behavior | Observe the search priorities and interests, and whether there are patterns in the search strategy for particular models |

The "why is it interesting / what is the threat or value model" column is still empty for everyone.

## Skeleton experiment: no matter the budget

1. Agent is given access to Wiki pages.
2. Agent points to what it's exploring, and continues to explore.
3. Ends with giving what it pointed to.

Models:

- Claude
- GPT
- Qwen 3.7 Flash
- Kimi K3 / Deepseek 4.1 Flash

Minimum: 1 model from the list, in 1 setting (the "do whatever you want" prompt), 10 rounds total,
perhaps with different variables. 100 to 300 runs for 1 cheap model.

## Statistical expectations

Fully rigorous statistics are probably not viable at the size of the Wikipedia graph, but we might
try.
