# Methods: what the analysis compares, and why

This is the mathematical companion to [analysis.md](analysis.md), which says how to run the pipeline and which file
holds what. Here each comparison is given twice: first the intuition, then the formal definition and the property
that justifies it. Sections 1–2 set up notation; 3 defines the baselines; 4 and 5 are the two comparisons (one click,
whole runs); 6 covers stopping; 7 says what to expect; 8 lists the assumptions.

## 1. Setting

**Intuition.** An agent walks on Wikipedia: on each page it sees the page's links and picks one. We want to know
whether its choices, and the trips they add up to, differ from those of simple choosers that have no preferences
of their own. Everything is a comparison with such a chooser, made on the same pages and the same link menus the
agent had.

**Graph.** $G = (V, E)$ is the directed link graph of enwiki-20260901: $V$ the articles (redirects merged into their
targets), $(x, y) \in E$ when article $x$ links to $y$. $N^+(x)$ is the out-neighbourhood of $x$. Links are not
reciprocal in general: only 20% of the edges of the largest strongly connected component $V_c$
($|V_c| = 6\,611\,107$, $|E_c| = 224\,620\,274$) have a reverse edge.

**Runs.** A run $r$ is a path $p^r_0, p^r_1, \dots, p^r_{m_r}$ with $m_r \le 15$ moves (fewer if the agent stops).
Step $s$ of a run has a current page $x_s$, a displayed menu $M_s$ (the page's out-links as the harness resolved
them: one entry per destination article), the chosen page $y_s$ and a kind: a **link step** ($y_s \in M_s$) or a
**jump step** (a search result or intent jump, $y_s$ not necessarily in $M_s$). The menu is shown in a shuffled
order; nothing below depends on the order.

**Which steps count.** The per-click analysis uses only steps the agent decided: steps where the harness picked a
random link after unusable replies, infrastructure failures, STOP, unresolved choices and search-first variants B/D's opening search
are removed. Steps answered on a retry are kept, since the model chose the page. The whole-run analysis uses the
path as it happened (harness picks included), because it describes where the run went.

**Conditions and clusters.** Runs are grouped into conditions $c$ (variant, model, memory, intent, world, prompt
wording, stop option, sampling temperature, schema major version) and never pooled across them. Uncertainty is computed over clusters:
a cluster is one run, or one multi-agent world (agents in a world see each other's traces and are not
independent). $S_c$ is the set of analysed steps of condition $c$, $R_c$ its runs.

## 2. Pages as vectors

**Intuition.** To say that the agent "stays on topic" we need a notion of how alike two pages are. Each page gets a
vector, and the similarity of two pages is the inner product of their vectors. Three choices of vector exist
(`analysis/similarity=`), and every measure below is defined through the inner product, so it means the same thing
under each.

$$\varphi : V \to \mathbb{R}^d, \qquad \operatorname{sim}(u, v) = \langle \varphi(u), \varphi(v) \rangle .$$

### 2.1 Categories: Wikipedia as a weighted hypergraph

Wikipedia's categories are fine-grained (2.64 million) and form a directed graph of their own (parent → child,
with cycles). We roll them up to $K-1 = 29$ root topics (Arts, Sports, …, Time; list in
`configs/analysis/topics/roots.yaml`) plus an `unlabeled` bucket, $K = 30$.

- **Roll-up.** A breadth-first search runs from all roots at once along parent → child edges, to depth 7.
  Category $c$ is assigned the set $R(c) \subseteq \{1,\dots,K-1\}$ of roots that reach it at the smallest depth
  (ties kept). Category redirects are followed and hidden or maintenance categories ignored. 1.68 million categories
  are reached; the rest have $R(c) = \emptyset$.
- **Topic mix of a page.** With $\mathcal C(v)$ the categories of page $v$ and
  $\mathcal C^*(v) = \{c \in \mathcal C(v) : R(c) \neq \emptyset\}$, each such category splits one vote over its
  roots:

  $$\chi_e(v) = \frac{1}{|\mathcal C^*(v)|} \sum_{c \in \mathcal C^*(v)} \frac{\mathbf 1[e \in R(c)]}{|R(c)|}, \qquad
  \chi(v) = e_{\text{unlabeled}} \ \text{ if } \mathcal C^*(v) = \emptyset .$$

  So $\chi(v) \in \Delta^{K-1}$, the probability simplex. 8.7% of articles are unlabeled. (With
  `weighting=uniform`, $\chi$ is instead uniform on the topics the page reaches.)
- **The hypergraph.** Topics are hyperedges $H_e = \{v : \chi_e(v) > 0\}$; a page belongs to several at once. Write
  the weighted incidence matrix $\Psi \in [0,1]^{V \times K}$, $\Psi_{v e} = \chi_e(v)$. It is row-stochastic:
  $\Psi \mathbf 1 = \mathbf 1$.
- **Similarity.** $\varphi = \chi$ and
  $\operatorname{sim}(u, v) = \sum_e \chi_e(u)\chi_e(v) = \Pr[E = E']$ for independent draws $E \sim \chi(u)$,
  $E' \sim \chi(v)$: the chance that a random topic of $u$ and a random topic of $v$ coincide. Even for $u = v$ this
  is $\lVert\chi(u)\rVert_2^2 < 1$ unless $u$ has a single topic (for *Battle of Hastings*, 0.225), so absolute values
  mostly measure how mixed pages are; only differences between agent and baseline on the same steps are read.
- **Prevalence.** $\pi = \frac{1}{|V|}\sum_{v} \chi(v)$, the topic mix of a uniformly random article.

### 2.2 Embeddings

$\varphi(v) = e(v)/\lVert e(v) \rVert$ where $e(v)$ is a text embedding of the page's title and lead section (the
prose before the first heading, at most 2 000 characters; `analysis/page_text`). The default embedder is BAAI bge-m3
($d = 1024$); Qwen3-Embedding-8B is the cross-check. Similarity is the cosine. A page whose lead is empty (0.2% of
articles) is embedded by its title alone; a page with no text at all gets $\varphi = 0$ and is counted. The analogue
of $\pi$ is $\bar\varphi$, the mean of $\varphi$ over 2 000 uniformly sampled articles. There are no topics, so no
topic tables (§4.3).

### 2.3 None

$d = 0$. Only measures that need the links alone remain: the bits per click (§4.2), revisits, hub visits and link
distance (§5).

## 3. Baselines

**Intuition.** A baseline is a pretend player standing exactly where the agent stood. None is fitted to the agent's
choices, for two reasons. First, a condition has about $20 \times 15 = 300$ clicks, far too few to estimate a
choice model over a graph of millions of pages, or even the $K^2 = 900$ cells of a topic table. Second, a fitted
model would absorb the very preferences we are looking for. With fixed baselines, the only scarce sample is the
agent's runs; the baseline side can be computed exactly (one click) or simulated as often as needed (whole runs).

### 3.1 Choosers for one click

A chooser is a distribution $q_s(\cdot)$ over the menu $M_s$ that depends only on what the agent saw at step $s$.

| Chooser | $q_s(j)$ for $j \in M_s$ | Intuition |
|---|---|---|
| uniform | $1 / \lvert M_s \rvert$ | a random walk: no preference at all |
| clickstream | $\tilde c(x_s, j) \big/ \sum_{k \in M_s} \tilde c(x_s, k)$ | the average human reader |
| novelty (optional) | $\frac{1}{\lvert \mathcal E_s \rvert} \sum_{e \in \mathcal E_s} \chi_e(j) / n_e$ | seeks topics that are rare on this menu |
| jump | uniform on link steps; at a jump step the next page is a uniform random article | random walk with teleporting |

- **clickstream.** $c(x, y)$ is the number of times readers went from $x$ to $y$ by a link in September 2026 (the
  Wikimedia clickstream, the month of the dump). Wikimedia drops pairs with 10 or fewer clicks, so an unlisted pair
  had $c \in \{0, \dots, 10\}$; we set $\tilde c = c$ if listed and $\tilde c = \kappa = 5$ otherwise (robustness:
  $\kappa \in \{1, 10\}$). A page with no listed link gets the uniform chooser. 99.5% of the 21.2 million link pairs
  match our page ids.
- **novelty.** $n_e = \sum_{k \in M_s} \chi_e(k)$ and $\mathcal E_s = \{e : n_e > 0\}$: pick a topic present on the
  menu uniformly, then a link in proportion to its membership. $\sum_j q_s(j) = \frac{1}{|\mathcal E_s|}\sum_{e}
  \sum_j \chi_e(j)/n_e = 1$. Defined only with categories.
- **jump.** The expected vector after a jump is $\pi$ (categories) or $\bar\varphi$ (embeddings). The jump rate is
  reported, not modelled: the chooser describes where a jump lands, not when the agent jumps.

The chooser's **expected next vector** at step $s$ is

$$w_s = \sum_{j \in M_s} q_s(j)\,\varphi(j) \quad \text{(link step)}, \qquad w_s = \pi \text{ or } \bar\varphi \quad \text{(jump step, jump chooser)} .$$

uniform, clickstream and novelty cover link steps only; jump covers both kinds.

### 3.2 Walks for whole runs

Whole-run baselines are processes on the strongly connected core $G_c = (V_c, E_c)$, started where the agent started.
Strong connectivity makes every walk well defined (no dead ends) and lets every page reach every other.

| Walk | Transition from $x$ | Intuition |
|---|---|---|
| uniform | $P(x, y) = 1/d^+(x)$ for $y \in N^+(x)$ | the random walk |
| jump | $P_\lambda = (1 - \lambda) P + \lambda\, \mathbf 1 \rho^\top$, $\rho$ uniform on $V_c$ | random walk that teleports as often as the agent searched |
| no_revisit | uniform on $N^+(x) \setminus \{\text{pages already on the path}\}$, or on $N^+(x)$ if that set is empty | a random walk that, like the agent, knows its own path |
| clickstream | $P_c(x, y) \propto \tilde c(x, y)$ for $y \in N^+(x)$ | the average human reader, step after step |

For the jump walk, $\lambda_c = (\text{jump moves}) / (\text{moves})$ of the agent in condition $c$. It is a property
of how often the agent used search, matched by design ("same freedom to teleport"), not tuned to any statistic
being tested; with $\lambda_c = 0$ (variant A) the jump walk is the uniform walk and is left out. no_revisit is not
Markov (it depends on the path), which is the point. The clickstream walk is a first-order chain: the clickstream
records pairs, not sessions, so human paths longer than one click are approximated by chaining pairs.

## 4. One click at a time (stage `clicks`)

### 4.1 Similarity excess

**Intuition.** Does the agent move to pages more alike the current one than the chooser would, on the same menu?

**Definition.** Over the steps $S$ of a condition that the chooser covers,

$$\overline{\mathrm{sim}}_{\text{agent}} = \frac{1}{|S|}\sum_{s \in S} \langle \varphi(x_s), \varphi(y_s) \rangle, \qquad
\overline{\mathrm{sim}}_{q} = \frac{1}{|S|}\sum_{s \in S} \langle \varphi(x_s), w_s \rangle, \qquad
\Delta_{\mathrm{sim}} = \overline{\mathrm{sim}}_{\text{agent}} - \overline{\mathrm{sim}}_{q} .$$

**Why this comparison is fair.** Let $\mathcal F_{s-1}$ be everything up to the choice at step $s$ (the path so far,
the page and the menu). The chooser's term is the conditional expectation of the agent's term if the agent chose
by $q$:

$$\langle \varphi(x_s), w_s \rangle = \mathbb E_{J \sim q_s}\!\left[\langle \varphi(x_s), \varphi(J) \rangle \,\middle|\, \mathcal F_{s-1}\right] .$$

So under the null hypothesis "the agent chooses by $q$", each summand $D_s = \langle \varphi(x_s), \varphi(y_s) - w_s
\rangle$ has $\mathbb E[D_s \mid \mathcal F_{s-1}] = 0$: the partial sums form a martingale and $\mathbb
E[\Delta_{\mathrm{sim}}] = 0$, **whatever path the agent took and whatever the graph looks like**. Where the agent
went (it may sit on computing pages whose links are mostly about computing) is held fixed; only the choice among the
offered links is compared. This is why the chooser is scored on the agent's own menus and not on topic statistics
of Wikipedia at large.

With categories, $\overline{\mathrm{sim}}$ is the trace share of the topic tables (§4.3); with embeddings it is a
mean cosine. Reported as `sim_agent`, `sim_model`, `sim_excess` in `summary.csv`.

### 4.2 Bits per click

**Intuition.** How much better than a coin with $|M_s|$ sides does the chooser predict which link the agent clicked?

**Definition.** Over link steps,

$$b_q = \frac{1}{|S_{\text{link}}|} \sum_{s \in S_{\text{link}}} \left[\log_2 q_s(y_s) - \log_2 \frac{1}{|M_s|}\right]
\quad \text{(bits per click)} .$$

**Meaning.** If the agent's own choice distribution at step $s$ is $a_s$, the expectation of the summand is
$\sum_j a_s(j) \log_2 \frac{q_s(j)}{u_s(j)} = \mathrm{KL}(a_s \Vert u_s) - \mathrm{KL}(a_s \Vert q_s)$, with $u_s$ the
uniform chooser. So $b_q > 0$ means $q$ is closer to the agent than the random walk is, in Kullback–Leibler terms.
It needs no page vectors and exists under every similarity option. $b_{\text{uniform}} = 0$ by construction.
Reported as `bits_vs_uniform`.

### 4.3 Topic tables: projecting directed link flow onto the hypergraph (categories)

**Intuition.** A click joins two pages, but each page sits in several topics. Picture drawing one topic at random
from the current page and one from the next page: the click contributes to that pair of topics, with the
probability of the draw. Summed over clicks, this gives a $K \times K$ table of "from topic $e$ to topic $f$" flow.
The same is done for the chooser, with its expected next page.

**Definition.** For one step, the agent adds $\chi(x_s)\chi(y_s)^\top$ and the chooser adds $\chi(x_s)w_s^\top$:

$$F^{\text{agent}} = \sum_{s \in S} \chi(x_s)\,\chi(y_s)^\top, \qquad F^{q} = \sum_{s \in S} \chi(x_s)\, w_s^\top .$$

**As a projection.** Let $W \in \mathbb N^{V \times V}$ count the agent's traversals, $W_{xy} = \#\{s : x_s = x,
y_s = y\}$, and $W^q_{xy} = \sum_{s : x_s = x} q_s(y)$ be the chooser's expected traversals from the same pages.
Then

$$F^{\text{agent}} = \Psi^\top W \Psi, \qquad F^{q} = \Psi^\top W^q \Psi .$$

Both tables are the directed link flow pushed through the incidence matrix of the topic hypergraph. Each traversed
link $x \to y$ is a directed hyperarc from the topics of $x$ to the topics of $y$, with weight
$\chi(x)\chi(y)^\top$; the tables add these up. On link steps $W^q$ is supported on the offered links, so the
chooser only uses links that exist from that page (a jump step of the jump chooser spreads over all articles).

**Properties.**

1. *Equal row sums.* $W\mathbf 1 = W^q \mathbf 1 = \nu$, the number of steps taken from each page (since $q_s$ sums
   to one), and $\Psi\mathbf 1 = \mathbf 1$. Hence $F^{\text{agent}}\mathbf 1 = F^q \mathbf 1 = \Psi^\top \nu =: m$, the
   topic mass of the pages the agent left from. The two tables differ only in where the flow goes, so they compare
   cell by cell.
2. *Probabilistic reading.* $F/|S|$ is the joint law of $(E, E')$ when a step is drawn uniformly, $E \sim \chi(x_s)$
   and $E' \sim \chi(y_s)$ (or $w_s$) independently. $A = \operatorname{diag}(m)^{-1}F$ is the law of $E'$ given $E$;
   its diagonal is the retention of each topic.
3. *Trace.* $\operatorname{tr} F^{\text{agent}} = \sum_s \langle\chi(x_s), \chi(y_s)\rangle$, so
   $\operatorname{tr}F/|S| = \overline{\mathrm{sim}}$ of §4.1 with $\varphi = \chi$.
4. *Directed.* $F \neq F^\top$ in general: $F_{ef}$ is flow from $e$ to $f$, and column sums (inflow) differ from row
   sums (outflow). Read tables by row.
5. *Not a Markov chain over topics.* Write $W^q = \operatorname{diag}(\nu) P_q$, with $P_q(x, \cdot)$ the chooser's
   distribution from page $x$ (averaged over the agent's visits to $x$). Then
   $A^q = \operatorname{diag}(m)^{-1}\Psi^\top \operatorname{diag}(\nu) P_q \Psi$: the page-level kernel lifted to
   topics and averaged over the pages the agent happened to visit. A topic-level Markov chain would need
   $P_q\Psi = \Psi B$ for some $K \times K$ matrix $B$ (lumpability). That does not hold on Wikipedia, where pages
   of the same topic link to very different topics, so $A^k$ is **not** the $k$-step topic kernel. Long-run
   questions are answered by simulation on the page graph (§5).
6. *Null property.* As in §4.1, $\mathbb E[F^{\text{agent}} - F^q] = 0$ entrywise when the agent chooses by $q$.

**Reported quantities.** $D = (F^{\text{agent}} - F^q)/|S|$ (where the agent sends more or less flow),
$L = \log\frac{F^{\text{agent}} + 1/2}{F^q + 1/2}$ (the same as a log-ratio), $A$ for both, retention, and per-cell
bootstrap intervals. Rows with $m_e < 20$ are flagged. With about 300 steps per condition most of the 900 cells
receive little mass, so the tables are for looking at; tests use the scalar summaries.

### 4.4 Uncertainty

All per-click statistics are ratios of sums over steps. Sums are formed per cluster, clusters are resampled with
replacement (2 000 times) and the statistic recomputed, giving a 95% percentile interval. An exact two-sided sign
test across clusters is reported for $\Delta_{\mathrm{sim}}$ and $b_q$ (each cluster's own value).

## 5. Whole runs (stage `walks`)

**Intuition.** One-click comparisons hold the agent's location fixed, so they cannot say where a run *ends up*.
For that, put $B = 100$ walkers of each kind on the agent's start page, let each make as many moves as the agent did,
and compare the trips.

### 5.1 Statistics of a path

For a path $p_0, \dots, p_m$ ($m \ge 1$), with $V_{\text{hub}} \subseteq V_c$ the hub pool (pages that are not lists
or junk titles, have at least 100 out-links and are in the top 1% by in-links from content pages, i.e. 340 or more;
47 473 pages) and $d_G$ the directed shortest-path distance:

| Statistic | Definition | Needs |
|---|---|---|
| `step_similarity` | $\frac{1}{m}\sum_{t=1}^m \langle \varphi(p_{t-1}), \varphi(p_t)\rangle$ | vectors |
| `breadth` | categories: $-\sum_e \bar\chi_e \log_2 \bar\chi_e$, the entropy in bits of $\bar\chi = \frac{1}{m+1}\sum_{t=0}^m \chi(p_t)$; embeddings: $1 - \lVert \frac{1}{m+1}\sum_t \varphi(p_t) \rVert$ | vectors |
| `drift_from_start` | $1 - \langle \varphi(p_0), \varphi(p_m) \rangle$ | vectors |
| `revisit_rate` | $\frac{1}{m}\sum_{t=1}^m \mathbf 1[p_t \in \{p_0, \dots, p_{t-1}\}]$ | links |
| `hub_share` | $\frac{1}{m}\sum_{t=1}^m \mathbf 1[p_t \in V_{\text{hub}}]$ | links |
| `end_distance` | $d_G(p_0, p_m)$; only with `walks.distance=true` (one breadth-first search per start page), dropped if larger than 8 | links |

Averages over the path count repeated pages with multiplicity, so a run that loops between two pages has a narrow
breadth.

### 5.2 Paired comparison

For run $r$ with start $p^r_0$ and $m_r$ moves, and walks $Z^{r,1}, \dots, Z^{r,B}$ of one kind from the same start
with $m_r$ moves:

$$d_r = T(\text{agent path } r) - \frac{1}{B}\sum_{b=1}^B T(Z^{r,b}), \qquad
\hat\Delta = \frac{1}{|R_c|}\sum_{r \in R_c} d_r .$$

Pairing on start and length removes two large sources of variation: where runs begin, and how long they last
(which the stop option and the wording change a lot). Runs that stopped before opening a page have no path and are
counted separately.

**Interval.** Cluster bootstrap of $\hat\Delta$ as in §4.4.

**Monte Carlo test.** Null hypothesis for run $r$: the agent's path is one more draw of the walk, given the start
and the length. Then $T(\text{agent}), T(Z^{r,1}), \dots, T(Z^{r,B})$ are exchangeable. A null replicate picks
$j_r$ uniformly in $\{1, \dots, B\}$ for each run and lets that walk stand in for the agent:

$$d^0_r = T(Z^{r,j_r}) - \frac{1}{B-1}\sum_{b \neq j_r} T(Z^{r,b}), \qquad \hat\Delta^0 = \frac{1}{|R_c|}\sum_r d^0_r,
\qquad p = \frac{1 + \#\{|\hat\Delta^0| \ge |\hat\Delta|\}}{1 + n_{\text{MC}}} .$$

The test uses $n_{\text{MC}} = 2\,000$ replicates and makes no assumption about the distribution of $T$, which
matters for skewed, discrete statistics such as revisits: there a sign test is wrong, because it tests the median of
$d_r$, which is negative for a perfectly random walker (most runs have no revisit, a few have one). Using $B - 1$
walks in the null mean makes the test slightly conservative; on 80 null cases, none fell below $p = 0.05$ and 7.5%
below $0.10$.

## 6. Stopping (stage `outcomes`)

For each condition, $k$ of $n$ runs end with the agent's STOP. The stop share $k/n$ gets a 95% Wilson interval. Two
prompt wordings $a, b$ are compared on the same start pages with the exact McNemar test on the discordant pairs
(start pages that stopped under one wording only): $p = \min\!\left(1, 2\sum_{i \le \min(n_{ab}, n_{ba})}
\binom{n_{ab}+n_{ba}}{i} 2^{-(n_{ab}+n_{ba})}\right)$.

## 7. What to expect

**Intuition.** An attractor is a part of Wikipedia the agent keeps being pulled into. It would show as a consistent
departure from every baseline: more revisits, less drift from the start, a narrower breadth, consecutive pages more
alike than the menu offers. No attractor looks like an agent whose statistics sit inside the walks' range.

| Signature | Baseline (sweep start pages) | Attractor would give | Stopping sweep, 11 Oct 2026 (3 models × 4 wordings, 20 runs each) |
|---|---|---|---|
| $\Delta_{\mathrm{sim}}$ vs uniform | 0 by construction | $> 0$ | $> 0$ in every condition; with embeddings the interval excludes 0 in 12 of 12, with categories in 10 of 12; largest without a stop option, smallest under "browse" |
| $b_{\text{clickstream}}$ | — | — | 1.0 to 2.6 bits per click in every condition: the agents click like human readers far more than like a random walk |
| `revisit_rate` | uniform walk ≈ 1%, clickstream walk ≈ 5% | high | 75–88% without a stop option (two-page loops); Qwen 63% even under "browse"; DeepSeek and Haiku under "browse" 3–4% |
| `end_distance` | ≈ 3 links | small | ≈ 1 link without a stop option; ≈ 3 for DeepSeek and Haiku under "browse" |
| `breadth`, `drift_from_start` | — | narrower, smaller | narrower and closer to the start than the random walk in every condition (embeddings), Qwen most |
| `hub_share` | ≈ 37% of moves | either way | 54–68% in looping cells, 14–23% under "browse" |
| stop share | — | — | set mostly by wording: "free to do what you want" 17/20 (DeepSeek), 13/20 (Haiku), mostly on the first page; "browse" 1–2/20; Qwen rarely stops |

**Reading.** The strongest departures come from runs that cannot stop and have nothing to do; they loop. That is
evidence of a failure mode more than of a preferred region. The fairer test of an attractor is that the *same*
pages or topics pull in runs that start in different places, which needs a cross-run statistic (for example, the
concentration of visits across runs against the walks' concentration from the same starts). That is the next
statistic to add; it fits §5.2 unchanged.

## 8. Assumptions and limits

- **Same-step conditioning.** §4 compares choices given the agent's own history. It is unbiased under the null
  whatever that history is (§4.1), but it says nothing about where runs go; §5 does.
- **Graph versus menus.** Walks use the 2026-10-08 graph build; the agent saw the harness's menus. Both come from
  the same dump with the same redirect handling; small differences remain (a few agent links are not graph edges,
  agent pages outside $V_c$ count as not-a-hub and have no distance).
- **Labels and embeddings are instruments.** The category roll-up is noisy (*Spreadsheet software* lands in
  Mathematics), and an embedder has its own idea of similarity. Both affect agent and baseline alike, which
  protects against false effects but can blur real ones. Effects should agree across categories and two embedders.
- **Human data.** The clickstream is the month's average reader, browsing for every reason, seeing links in page
  order; our agents see a shuffled list. For the closest comparison, also run the agent in page order.
- **Sample size.** About 20 runs per condition: read intervals and directions, not single p-values. No correction
  for multiple comparisons is applied across the many conditions × statistics.
- **No fitted parameters.** Besides conditioning on what the agent saw (its pages and menus, §4), the only
  quantities taken from the agent are its start pages, run lengths and, for the jump walk, its jump rate. $\kappa$
  and the walk count $B$ are fixed constants.

## Symbols

| Symbol | Meaning |
|---|---|
| $G = (V, E)$, $G_c$ | directed link graph; its largest strongly connected component |
| $N^+(x)$, $d^+(x)$ | out-neighbours, out-degree |
| $x_s, y_s, M_s$ | current page, chosen page, displayed menu at step $s$ |
| $\chi(v) \in \Delta^{K-1}$ | topic mix; $\Psi$ the incidence matrix with rows $\chi(v)$; $H_e$ the hyperedge of topic $e$ |
| $\varphi(v)$ | page vector ($\chi$, a unit embedding, or empty) |
| $\pi$, $\bar\varphi$ | mean vector of a random article |
| $q_s$, $w_s$ | chooser's distribution over $M_s$; its expected next vector |
| $\tilde c$, $\kappa$ | clickstream counts with unlisted pairs set to $\kappa = 5$ |
| $W$, $W^q$, $\nu$ | agent's and chooser's link traversals; steps taken from each page |
| $F, A, D, L$, $m$ | topic tables (§4.3); topic mass of the pages left from |
| $P, P_\lambda, P_c$ | uniform, jump and clickstream walk kernels; $\lambda$ the agent's jump rate |
| $V_{\text{hub}}$ | hub pool (§5.1) |
| $T$, $d_r$, $\hat\Delta$, $B$ | path statistic, paired difference of run $r$, mean difference, walks per run |

Code: `analysis/similarity.py` (§2), `analysis/models.py` and `analysis/clickstream.py` (§3.1),
`analysis/walks.py` (§3.2, §5), `analysis/kernel.py` (§4), `analysis/inference.py` (§4.4),
`analysis/outcomes.py` (§6).
