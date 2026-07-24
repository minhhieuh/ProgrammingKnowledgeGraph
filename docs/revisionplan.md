# TOSEM Major Revision Response Plan — Programming Knowledge Graphs (PKG)

## Context

The paper (`main.tex` and its `sections/*.tex` includes) received a **Major Revision** decision from TOSEM with three reviews plus an Associate Editor meta-review (`ResponseLetter.tex`). All three referees converge on one root concern: the paper does not clearly **isolate the contribution of PKG itself** from confounds like candidate diversity in reranking, benchmark choice, and possible corpus overlap. R3 additionally challenges novelty and statistical rigor; R2 challenges benchmark realism and mechanism validation; R1 asks for stronger baselines and ablations.

This plan was built by reading `ResponseLetter.tex` in full (26 distinct review bullets across R1/R2/R3 plus the editor's summary) and cross-referencing every point against the current paper source (`sections/introduction.tex`, `related-work.tex`, `methodology.tex`, `experimental-setup.tex`, `results.tex`, `discussion.tex`, `Threats.tex`, `conclusion.tex`, `appendix.tex`, and `tables/*.tex`). All 26 bullets are accounted for below, consolidated into **20 tasks** (several bullets are duplicates across reviewers, e.g. data leakage is raised by both R2 and R3, and pruning ablation is raised by both R1 and R2).

Two things worth flagging up front, found while reading the source:
- **The pruning ablation reviewers ask for already exists in draft form.** `sections/appendix.tex` lines 37–59 contain a commented-out `table:ablation_study_human-eval` with real numbers (Block-PKG vs. Block-PKG-No-Pruning per model). This should be verified/re-run and re-enabled rather than built from scratch.
- **R3's claim that Table 7 (error analysis) doesn't reconcile with the prose does not hold against the current `.tex` source.** Recomputing StarCoder-7B's row total: `147+64+8+0+18+7 = 244`, matching the "265→244, −21" claim in `discussion.tex` line 89. This is very likely a PDF/column-wrapping rendering artifact from the submitted PDF, not a source-of-truth error — still worth a rendering pass and a cleaner table before resubmission.

---

## Artifact Recovery Status (as of 2026-07-23, per correspondence with Minh)

Before committing to full re-runs for the Experiments-track tasks, we checked whether existing run artifacts could answer reviewer asks via analysis alone, rather than regenerating results from scratch. Minh's response on artifact availability:

| Artifact | Needed for | Confirmed available? | Location |
|---|---|---|---|
| Per-item pass/fail records (all models × both benchmarks, open- and closed-source) | T15, T18, T8 | ✅ Yes — "All available for both the close-source and open-source models on HumanEval and MBPP." | *[placeholder — awaiting path from Minh]* |
| Generated candidate code per condition (incl. NoRAG and baselines) | T4, T16 | ✅ Yes — "The reranking ablation can be redone from the saved outputs, so no regeneration is needed." | *[placeholder — awaiting path from Minh]* |
| Retrieved context text per task | T7, T8 | ✅ Yes — "The actual context text used for each task is recoverable." | *[placeholder — awaiting path from Minh]* |
| Retrieval scores/rankings (top-k) | T7 | ❌ No — "we didn't save the retrieval scores or rankings... If the retrieval-quality analysis requires those, we might need to rerun the retrieval step." | — |
| PythonAlpaca corpus snapshot (as indexed) | T8, T2 | ❓ Not explicitly addressed in Minh's response — needs confirmation | *[placeholder — awaiting confirmation + path from Minh]* |
| Tutorials corpus snapshot (as indexed) | T8 | ❌ No — "The tutorial (text-side) corpus snapshot is also missing, which limits a fully precise contamination check on that corpus." | — |
| Graph construction + pruning implementation code | T2 | ❌ No — "the pruning code is missing." Blocks producing the pruning-off condition without reimplementing/recovering it. | — |
| Pruning-ON retrieval outputs | T2 | ✅ Yes — "From Iman's package, the pruning-on outputs exist." | *[placeholder — awaiting path from Minh]* |
| Prompt assembly / execution harness scripts + configs (needed so new numbers stay comparable to submitted ones) | T4, T7, T8, T15, T16, T18 | ❓ Not addressed in Minh's response | *[placeholder — awaiting confirmation + path from Minh]* |

This upgrades several tasks from "needs a full pipeline re-run" (**Experiments**) to "answerable by scripting/analysis over existing artifacts" (**Analysis**) — introduced below as a third task category alongside Experiments and Writing. Unlike Experiments, Analysis tasks require no new model inference or retrieval runs, only processing of already-generated outputs. Where an artifact gap blocks part of a task (e.g., missing pruning code, missing retrieval scores), that's called out per-task rather than forcing a clean recategorization.

---

## Task Index (26 review bullets → 20 tasks)

| # | Category | Reviewers | One-line summary | Status |
|---|----------|-----------|-------------------|--------|
| 1 | Experiments + Writing | R1-W1, R2-M6, Editor | Add/discuss stronger structure-aware & graph-RAG baselines | Not started |
| 2 | Experiments (blocked) | R1-W2, R2-M5 | Pruning ablation (draft data exists in appendix, commented out) | ❌ Blocked — pruning code missing, see Artifact Recovery Status |
| 3 | Experiments + Writing | R1-W3, R2-M3, Editor | Benchmark realism / add harder or knowledge-dependent benchmarks | Not started |
| 4 | Analysis | R1-W4, R2-M1 | Controlled reranking ablations to isolate PKG's contribution (CORE) | Not started — artifacts confirmed available |
| 5 | Experiments + Writing | R1-W5, R2-Minor3 | Qualitative case studies incl. failure cases | Not started |
| 6 | Experiments + Writing | R2-M2 | Frontier/strong-model evaluation and framing | ⏳ Writing half done, experiment half not started |
| 7 | Analysis (partial) | R2-M4 | Direct retrieval-quality analysis (not just downstream pass@1) | Not started — annotation/qualitative half analysis-ready, top-k score comparison blocked |
| 8 | Analysis (partial) | R2-M7, R3-S2, Editor | Benchmark–corpus overlap / data leakage check | Not started — per-task check analysis-ready, full corpus scan blocked on corpus snapshots |
| 9 | Writing | R2-Minor1 | Justify "knowledge graph" terminology | ✅ Done |
| 10 | Writing | R2-Minor2 | More implementation detail (configs, serialization, execution setup) | ⏳ Awaiting input from Minh |
| 11 | Writing | R2-Minor4 | Typos / capitalization pass | ✅ Done |
| 12 | Writing | R3-N1 | Contributions list inflates count (2–4 are instantiations of 1) | ✅ Done |
| 13 | Writing | R3-N2 | Thin differentiation vs. crowded field, esp. cAST | ✅ Done |
| 14 | Writing | R3-N3 | Foreground the empirical study as the core contribution | ✅ Done |
| 15 | Analysis + Writing | R3-S1, Editor | Statistical validity of pass@1 comparisons | Not started — artifacts confirmed available |
| 16 | Analysis + Writing | R3-S3 | Reranker justification / try an alternative signal | Not started — candidate re-scoring analysis-ready, harness comparability unconfirmed |
| 17 | Writing (Rebuttal note) | R3-S4 | Table 7 arithmetic — verify, explain, reformat | ✅ Done |
| 18 | Writing + Analysis | R3-S5 | Explain Func-BM25 N/A; thin closed-source analysis | Not started — analysis half artifacts confirmed available |
| 19 | Writing (production) | R3-Minor1 | "KPG" typo in Figure 4/5 legends | Not started |
| 20 | Writing (production) | R3-Minor2 | Replace unreadable scatter plots with per-topic bar charts | Not started |

---

## A. Experiments / Analysis Tasks

### T4 — Controlled reranking ablations (CORE, do first) — Category: Analysis
**Reviewers:** R1-W4, R2-M1 (both call this out as central to the paper's main claim).
Run three additional reranking conditions and report them alongside the existing `Reranked` column in `tables/humaneval.tex` / `tables/mbpp.tex`:
1. Rerank only PKG-generated candidates (Func-PKG + Block-PKG samples), same N and same execution-filter budget as the current reranker.
2. Rerank only non-PKG candidates (NoRAG + BM25 + VoyageEmb), same N/budget — isolates whether gains are from PKG or from ensembling per se.
3. No reranking, random pick among the same candidate pool — a floor baseline.
This directly answers whether headline "reranking helps" is a PKG effect or a candidate-diversity effect, per `methodology.tex` §6.2 (Solution Reranking, lines 129–150) and Eq. 11 (`eq:rerank_select`).
**Artifacts needed:** Generated candidate code for every condition (NoRAG, BM25, VoyageEmb, Func-BM25, Func-PKG, Block-PKG), per model, per benchmark.
**Artifacts confirmed:** ✅ Yes, per Minh (2026-07-23): "The reranking ablation can be redone from the saved outputs, so no regeneration is needed." All three ablation conditions above can be computed by re-scoring/re-selecting over already-saved candidates — no new model inference required.
**Location:** *[placeholder — awaiting path from Minh]*

### T2 — Pruning ablation (data partially exists) — Category: Experiments (blocked)
**Reviewers:** R1-W2, R2-M5.
Re-run (or verify) the commented-out study in `sections/appendix.tex` lines 37–59 (`table:ablation_study_human-eval`, Block-PKG vs. Block-PKG-No-Pruning). Extend it with: effect on context length (token count, reuse `table:token_budget` methodology), and 2–3 qualitative examples of pruning helping/hurting (can reuse the retrieval example already sketched in `sections/appendix.tex` lines 150–171, "Challenges in Retrieving Information from PKG"). Also run the MBPP equivalent if not already collected.
**Artifacts needed:** Pruning-ON retrieval outputs (already have these); pruning-OFF retrieval outputs (do not exist, must be generated); graph construction + pruning implementation code (needed to regenerate the pruning-OFF condition and measure context-length deltas).
**Artifacts confirmed:** ✅ Pruning-ON outputs confirmed available, per Minh (2026-07-23): "From Iman's package, the pruning-on outputs exist." ❌ Pruning implementation code confirmed **missing**: "the pruning code is missing. Therefore, I can't produce the pruning-off condition or the context-length deltas requested by Referee 2 without rerunning the graph build with pruning turned off." **This means T2 does not qualify as a pure Analysis task** — the core with/without-pruning ablation requires either recovering or reimplementing the pruning code, then rerunning graph construction for the pruning-OFF condition. Stays categorized as Experiments, now explicitly blocked pending a recovery/reimplementation decision.
**Location:** *[placeholder — awaiting path from Minh for pruning-ON outputs; pruning code itself confirmed not recoverable as-is]*

### T1 — Stronger baselines
**Reviewers:** R1-W1, R2-M6, Editor.
Given resource constraints, triage which of the cited baselines are actually implementable in the revision window: **cAST** (already cited as `zhang-etal-2025-cast`, most directly comparable — R3 flags this as the closest prior art) is the highest-priority addition since it's explicitly AST-chunking for code RAG. GraphCodeBERT and RepoCoder-style comparisons are lower cost since GraphCodeBERT embeddings are off-the-shelf. RAP-Gen/ReACC/Codegrag/repo-aware-KG baselines are heavier lifts — flag these for the Rebuttal (see R-1 below) if time-boxed out.

### T3 — Benchmark realism
**Reviewers:** R1-W3, R2-M3, Editor.
Two sub-asks, prioritize differently:
- **Contamination-robust benchmark**: add LiveCodeBench (contamination-controlled by design, directly addresses R1's "outdated benchmark" point) as a third benchmark alongside HumanEval/MBPP.
- **Knowledge-dependent tasks**: R2's sharper point is that HumanEval/MBPP are algorithmic and don't exercise the "external programming knowledge" (API/docs) that motivates PKG in `introduction.tex` lines 5–11. A smaller, targeted addition — a library/API-heavy subset (e.g., a slice of a documentation-grounded benchmark) — would do more to close this gap than a second algorithmic benchmark. Scope this as a smaller supplementary experiment rather than a full new benchmark suite, and be explicit in the Discussion about what is and isn't covered (ties to T3-Writing below).

### T7 — Retrieval-quality analysis — Category: Analysis (partial)
**Reviewers:** R2-M4.
Add a direct (not pass@1-mediated) evaluation of retrieval quality: manually annotate query-context relevance for a sample (e.g., 50–100 queries) across BM25 / VoyageEmb / PKG, report top-1 relevance agreement, and pull 3–5 qualitative examples where PKG retrieves evidence the baselines miss (and vice versa for failure cases — pairs with T5). This is new analysis, not a new experiment pipeline — the retrieved contexts are presumably already logged from existing runs.
**Artifacts needed:** Retrieved context text per task, for manual relevance annotation and qualitative examples; top-k retrieved candidates with similarity scores/rankings, for the top-k retrieval comparison sub-analysis.
**Artifacts confirmed:** ✅ Context text confirmed recoverable, per Minh (2026-07-23): "The actual context text used for each task is recoverable." ❌ Scores/rankings confirmed **not saved**: "we didn't save the retrieval scores or rankings. If the retrieval-quality analysis requires those, we might need to rerun the retrieval step." **Net effect:** the manual-annotation/qualitative-examples half of this task (the main R2-M4 ask) is Analysis-ready now; the top-k-with-scores comparison sub-analysis remains blocked pending a retrieval-only rerun (not a full pipeline rerun).
**Location:** *[placeholder — awaiting path from Minh]*

### T8 — Data leakage / corpus overlap check — Category: Analysis (partial)
**Reviewers:** R2-M7, R3-S2, Editor.
Run exact-match and near-duplicate detection (e.g., MinHash/Jaccard or embedding-similarity threshold) between PythonAlpaca + Tutorials (the PKG source corpora, `experimental-setup.tex` lines 6–9) and HumanEval/MBPP problems + canonical solutions. Report the fraction of benchmark problems with a near-duplicate in the retrieval corpus, and whether pass@1 gains correlate with duplicate presence (e.g., stratify results by duplicate vs. non-duplicate subsets). This is tractable — it's a corpus-comparison script, not a new model run.
**Artifacts needed:** (a) Retrieved context text per task, to check whether retrieved content contains near-duplicate benchmark solutions — a per-task check, not a full corpus scan. (b) Per-item pass/fail records, to stratify pass@1 gains by contamination status (ties to T3's "contamination-free subset" framing). (c) Full PythonAlpaca corpus snapshot as indexed, for an exhaustive corpus-level near-duplicate scan. (d) Full Tutorials corpus snapshot as indexed, same purpose for text-centric PKG.
**Artifacts confirmed:** ✅ (a) and (b) confirmed available (same records as T7/T15). ❌ (d) confirmed **missing**, per Minh (2026-07-23): "The tutorial (text-side) corpus snapshot is also missing, which limits a fully precise contamination check on that corpus." ❓ (c) not explicitly addressed in Minh's response — may be re-derivable from the original public PythonAlpaca source even if the exact indexed snapshot is lost, but the specific subset/version used (115,000 extracted functions) needs confirming; treat as unconfirmed until Minh replies. **Net effect:** the per-task contamination check (using retrieved context + pass/fail records) is Analysis-ready; the full corpus-level exhaustive near-duplicate scan is blocked for the tutorials corpus and unconfirmed for PythonAlpaca.
**Location:** *[placeholder — awaiting path from Minh, and confirmation on PythonAlpaca corpus snapshot status]*

### T15 — Statistical validity — Category: Analysis
**Reviewers:** R3-S1, Editor.
Run McNemar's test (paired, since both conditions are evaluated on the same fixed problem set) for the pairwise comparisons currently reported as point deltas in `tables/humaneval-prop.tex` and `tables/mbpp-prop.tex` (closed-source models, where deltas are smallest, e.g. GPT-4o 81.4%→83.4%, Claude-3-Haiku 67.2%→67.6%). Extend to the open-source tables if time allows. Report significance markers (e.g., `*` for p<0.05) directly in the result tables.
**Artifacts needed:** Per-item pass/fail records for every model × benchmark × retrieval condition (open-source and closed-source, HumanEval and MBPP) — McNemar's test is computed directly from paired per-item outcomes, nothing else needed.
**Artifacts confirmed:** ✅ Yes, per Minh (2026-07-23): "All available for both the close-source and open-source models on HumanEval and MBPP." This is a clean Analysis task — purely a statistics computation over existing per-item records, no rerun of any kind needed.
**Location:** *[placeholder — awaiting path from Minh]*

### T16 — Alternative reranker signal — Category: Analysis + Writing
**Reviewers:** R3-S3.
The current reranker (Eq. 11, `methodology.tex` lines 144–150) picks by cosine similarity between query and candidate code, and `discussion.tex` §6.4 Implication 3 (lines 272–273) already admits this correlates with surface plausibility, not correctness, particularly in math/optimization/algorithm topics. Run at least one alternative signal as an ablation — e.g., an execution-aware signal (already partially available via the `R(c)` runtime filter in `eq:rerank_select`'s pipeline) or a lightweight LLM-judge reranker — and report whether it closes the gap to the "Ideal Reranker" oracle in the math/optimization/algorithm topics specifically.
**Artifacts needed:** Generated candidate code per condition (to re-score with an alternative signal — no new generation needed, just a different selection function applied to the existing candidate pool); for full comparability with the submitted numbers, the original prompt assembly / execution harness configuration.
**Artifacts confirmed:** ✅ Candidate code confirmed available (same artifact as T4). ❓ Prompt assembly/execution harness — not addressed in Minh's response; unconfirmed whether recoverable. **Net effect:** the core "try an alternative reranker signal" work is Analysis (re-scoring saved candidates), contingent on the harness-comparability caveat above.
**Location:** *[placeholder — awaiting path from Minh]*

### T6 — Frontier-model evaluation (extend, don't rebuild)
**Reviewers:** R2-M2.
The paper already has closed-source results (`tables/humaneval-prop.tex`, `tables/mbpp-prop.tex`, `discussion.tex` §6.1) showing marginal/negative benefit for strong models — this is evidence, not a gap, but it's underdeveloped (ties to T18). If feasible, add 1–2 more recent frontier models (e.g., a newer Claude/GPT tier) to strengthen the trend rather than starting a new model class.

### T18 (experiment half) — Closed-source error/topic analysis — Category: Analysis
**Reviewers:** R3-S5.
`discussion.tex` §6.1 (lines 8–17) has no error-type or topic breakdown for closed-source models, unlike the open-source analysis in §6.3/§6.2. Add at least a topic-level breakdown for one or two closed-source models (reuse the same pipeline as Figure 3/`fig:topic_based_accuracy`) to support the "retrieval policies may need to be model-adaptive" claim on stronger footing.
**Artifacts needed:** Per-item pass/fail records for closed-source models on MBPP (and HumanEval), to compute topic-level and error-type breakdowns analogous to the existing open-source analysis.
**Artifacts confirmed:** ✅ Yes, per Minh (2026-07-23) — same confirmed per-item records as T15, explicitly covers closed-source models. This is a clean Analysis task: reuse the existing topic-mapping/error-classification pipeline over already-computed closed-source outcomes.
**Location:** *[placeholder — awaiting path from Minh]*

---

## B. Writing Tasks — locations and specific edits

### T1 (writing half) — Position stronger baselines / narrow claims where not run
**Locations:** `sections/related-work.tex` §6.2 "RAG for Code Generation" (lines 21–37, esp. the "Differences of our work" paragraph at lines 32–37); `sections/discussion.tex` §6.5 Open Gaps (lines 286–297).
**Edit:** For any baseline not empirically added (T1-experiment triage), add an explicit sentence in Related Work naming it and stating why it's out of scope for this revision (e.g., requires fine-tuning, different task setting), and add a corresponding line to Open Gaps. This converts an unaddressed gap into an acknowledged, scoped limitation — reviewers penalize silence more than a scoped absence.

### T3 (writing half) — Reframe motivation/scope re: benchmark realism
**Locations:** `sections/introduction.tex` paragraph 3 (lines 9–11, the "actionable programming knowledge is heterogeneous" motivation) and `sections/Threats.tex` (lines 6–8, already partially hedges on "Python and English tutorials").
**Edit:** Add one sentence acknowledging that HumanEval/MBPP are algorithmic and that the external-knowledge motivation is most directly tested in the (new, smaller) API/doc-heavy supplementary experiment from T3-experiment, not the primary benchmarks. This pre-empts R2's "gap between motivation and evaluation" critique even in places the new experiment doesn't fully close it.

### T5 (writing half) — Extend case studies with failure examples
**Locations:** `sections/appendix.tex` §"Examples" (lines 301–476) currently has only 2 success examples (HumanEval #159, #90, both "Failed→Passed"); `sections/discussion.tex` §6.3 (lines 83–137, error analysis).
**Edit:** Add 2–3 paired examples where PKG *hurts* (retrieves misleading context and flips a correct NoRAG answer to incorrect) to balance the current all-success framing — R1 explicitly asks for "when PKG helps or hurts." Also add one example each for the IndentationError and TypeError increases already reported in Table 7 (`discussion.tex` line 88), showing the actual retrieved snippet that triggered the failure — this directly answers R2-Minor3.

### T6 (writing half) — Sharpen closed-source framing — ✅ DONE (writing half only)
**Location:** `sections/discussion.tex` §6.1 (lines 8–17).
**Edit:** The current text already states the key finding ("marginal benefit... sometimes negative" for closed-source, high-baseline models). Added a `\revise{}`-marked sentence right after the existing paragraph making the answer to R2's implicit question ("do frontier models even need PKG?") explicit: for models near ceiling, PKG's marginal value is low/negative, and the gated/conditional retrieval strategy from Implication 1 (`\label{sec:disc-implications}`, confirmed still valid) is the recommended deployment mode. **T6's experiment half (adding 1–2 more recent frontier models, per the Task Index) is still not started** — this only closes the writing half.

### T9 — Justify "knowledge graph" terminology — ✅ DONE
**Locations:** `sections/methodology.tex` §5.1 (lines 6–19, "Programming Knowledge Graph (PKG) Construction"); optionally `sections/related-work.tex` §6.1 (lines 5–17, structure/graph-aware RAG).
**Edit:** Added a `\paragraph{Why a graph, and not flat or hierarchical chunking.}` block (wrapped in the `revised` environment) after the "Graph schema" paragraph in `methodology.tex`, making three points: (1) multiple typed entry points per artifact (`Name`/`Impl`/`Block`) vs. one opaque chunk ID, (2) branch-pruning as a graph traversal operation (`parent` edges over the induced DAG) that a flat chunk list can't express, (3) one typed-graph formalism uniformly covering both AST-derived code containment and JSON path hierarchy. Deliberately did not cite specific equation numbers for the pruning step, since those equations in `methodology.tex` (the `G_{v^*}^{-u}` / `G_{\text{pruned}}` pair) currently have no `\label` — referenced the paragraph name instead to avoid a dangling/incorrect `\ref`.

### T10 — Add missing implementation details — ⏳ AWAITING INPUT FROM MINH
**Locations:** `sections/experimental-setup.tex` (lines 1–20, currently covers retrieval approaches, dataset stats, models, metric, benchmarks but not vector index config or runtime-check details); `sections/methodology.tex` §5.1.4 (lines 64–71, embedding/storage — mentions Neo4j + vector index but not index type/params); prompts already exist per-model in `sections/appendix.tex` (lines 176–275).
**Edit:** Add a short paragraph to `experimental-setup.tex` specifying: vector index type/parameters (e.g., HNSW config in Neo4j 5.20.0), JSON-extraction prompt/validation procedure (referenced in `methodology.tex` line 45 but not shown, unlike the code-generation prompts which are in the appendix), and the sandboxing setup used for the runtime-sanity filter `R(c)` (`methodology.tex` line 140).
**Status:** Skipped for now — needs real values from Minh (vector index config, JSON-extraction prompt, sandbox setup) rather than fabricated specifics. Resume once that input arrives.

### T11 — Typo / capitalization pass — ✅ DONE
**Confirmed locations (exact, verified by grep):**
- `sections/results.tex:31` — "retreival" → "retrieval" (in the Ideal Reranker baseline description). Fixed.
- `sections/experimental-setup.tex:9` — "pkg contains 288,583 path-value nodes..." → capitalize to "PKG contains..." (mid-sentence lowercase, exactly the inconsistency R2 flagged). Fixed.
**Edit:** Also grep the full source for any other `pkg` (lowercase, not part of `\label`/`\ref` macros) before resubmission — the two above are the ones with reader-visible impact.
**Verification:** Re-ran a broader grep for `\bpkg\b` and common misspellings across `sections/*.tex`. All other lowercase `pkg` hits are inside `\label{app:pkg-schema}` / `\ref{sec:pkg-generation}` macros (not reader-visible) or inside commented-out (`%`) dead text, so left untouched. No other typos found.

### T12 — Restructure the contributions list — ✅ DONE
**Location:** `sections/introduction.tex` lines 42–61 (the 5-item enumerate).
**Edit:** Reviewer R3 is right that items 2–4 (granularity comparison, JSON/DAG construction, pruning) are sub-components of item 1 (the PKG representation), not independent contributions. Restructured to 3 items: (1) PKG representation with the granularity/pruning mechanisms folded in as its defining internal design choices, (2) the reranking mechanism, (3) a new, elevated empirical-study contribution (previously only mentioned in intro prose, not listed as a contribution). Whole `enumerate` block wrapped in the `revised` environment. Verified no other section numerically references the old 5-item structure (grepped for "contribution"/"novel" across `sections/*.tex`) — the prose paragraphs above the list already describe each piece individually and are unaffected. Abstract/conclusion alignment with the elevated empirical-study framing is deliberately left to T14.

### T13 — Strengthen differentiation from cAST and the crowded field — ✅ DONE
**Location:** `sections/related-work.tex` lines 30–37 (esp. the "Differences of our work with the existing literature" paragraph, lines 32–37, which currently differentiates from `wang2024coderag`, `zhou2022docprompting`, and `zhang2023repocoder` but never explicitly contrasts with cAST, StructRAG, KG²RAG, GRAG, KGCompass, or Prometheus even though cAST is cited in `introduction.tex` lines 13/18).
**Edit:** Added a `revised`-wrapped paragraph after the RepoCoder differentiation (before "Despite these advancements...") explicitly naming cAST (confirmed citation key `zhang-etal-2025-cast` exists in `references.bib`, was previously cited only in the introduction, never in Related Work) and stating the concrete three-axis difference: cAST does AST-based chunking for retrieval units alone, whereas PKG (a) unifies code-centric and text-centric knowledge in one graph schema, (b) supports two selectable retrieval granularities with structural pruning over the induced DAG, and (c) couples retrieval with post-generation reranking across RAG/non-RAG candidates — framed as no single cited system (also ties back to StructRAG, KG²RAG, GRAG, KGCompass, Prometheus already discussed earlier in the section) combining all three. Consistent with T14's framing of the empirical study as the complementary contribution.

### T14 — Foreground the empirical study — ✅ DONE
**Locations:** Abstract (`main.tex` lines 210–214); `sections/introduction.tex` contributions list (post-T12 restructuring); `sections/conclusion.tex` (lines 6–10).
**Edit:** R3 explicitly suggests this reframing is the paper's strongest, most defensible angle: "the empirical study of when and why structure-aware retrieval helps or hurts." Contributions list already elevated in T12. Added a `\revise{}`-marked sentence to the abstract (`main.tex`, after the pass@1 numbers) naming the empirical study explicitly, and extended the opening sentence of `conclusion.tex` line 6 with a `\revise{}`-marked clause stating the same, so the framing is stated up front rather than only implied by the topic/error findings later in the paragraph (lines 8–9, left unchanged since they already support this framing).

### T17 — Table 7 verification and reformatting (Rebuttal-flavored writing task) — ✅ DONE
**Location:** `sections/discussion.tex` lines 91–115 (`table:error_analysis`) and surrounding prose lines 87–89.
**Finding:** Verified arithmetic is internally consistent in the current `.tex` source (StarCoder-7B: 265→244 matches row sums; other rows check out similarly). This is very likely a PDF column-wrap/multirow rendering artifact in the submitted PDF, not a source error.
**Edit:** Replaced the nested-`tabular`-inside-`scalebox` two-line header (the likely visual-alignment culprit) with a `\multicolumn` + `\cmidrule` grouped header (one spanning model-name row, one Base/+Block-PKG sub-header row). Added a printed **Total** row (265/244, 357/269, 208/249 — recomputed independently from the six visible rows and confirmed to match the prose in line 88–89 exactly) so a reader no longer has to sum manually. Updated the caption to explain Base/+Block-PKG and the Total row. Added a short `\revise{}`-marked clause in the prose (line 89) pointing to the new Total row. No data values changed.
**Color-marking gotcha (verified by test-compiling with MiKTeX):** wrapping the float in the outer `\begin{revised}...\end{revised}` environment did **not** color the table — `color`/`xcolor` is documented as unreliable across page/box boundaries, and floats (`table`, `table*`, `figure`) can ship out on a different page than where an outer `\color` was set, breaking the color push/pop pairing. Fixed by putting `\color{RevisionColor}` *inside* the float itself (right after `\centering`, before `\caption`) instead of wrapping it from outside. Confirmed fixed by recompiling and visually inspecting page 19 of the output PDF. **Any future table/figure color-marking must set `\color{RevisionColor}` inside the float, not via the `revised` environment wrapped around it.**

### T18 (writing half) — Explain Func-BM25 N/A
**Locations:** `tables/humaneval-prop.tex` line 12 (comment shows `N/A` for `Func-BM25` across all closed-source rows) and `tables/mbpp-prop.tex` line 12 (same); `sections/discussion.tex` §6.1.
**Edit:** Add a table footnote or one sentence in `discussion.tex` §6.1 explaining why Func-BM25 wasn't run for proprietary models (most likely: API-based closed models weren't paired with the function-extraction BM25 pipeline for cost/scope reasons — confirm actual reason with the team before writing).

### T19 — Fix "KPG" typo in Figure 4/5 legends
**Locations:** Figures referenced from `sections/discussion.tex` lines 42–48 (`fig:topic_based_accuracy_codellama`, image `images/appendix/codellama7b.png`) and lines 50–61 (`fig:topic_based_accuracy_deepseek`, image `images/appendix/deepseek.png`).
**Note:** These are pre-rendered PNGs; the typo is baked into the image, not the `.tex` legend/caption text (the `.tex` captions already correctly say "PKG"). This must be fixed in the original plotting code (not present in this repo) and the images re-exported and replaced in `images/appendix/`.

### T20 — Replace scatter plots with per-topic bar charts
**Locations:** `sections/appendix.tex` figures at lines 193–199, 207–213, 237–243, 252–258, 283–289, 292–298 (the 2D-embedding scatter plots for CodeLlama-7B, StarCoder2-7B, DeepSeek-Coder-7B, each NoRAG/reranked pair — R3's "Figs 7-12").
**Note:** Same caveat as T19 — these are external plotting outputs. Since the underlying data is per-problem pass/fail by topic, a per-topic bar chart (correct vs. incorrect count per topic, comparable to `fig:topic_based_accuracy` at `results.tex` line 172–178) is a straightforward re-plot of already-collected data, not a new experiment. Update the six figures and their surrounding prose (`appendix.tex` lines 202, 246, 279–281) to reference bar charts instead of scatter plots.

---

## C. Rebuttal Angles (where full compliance isn't feasible in the revision window)

### R-1: Baselines not empirically added (residual of T1)
If time-boxes out RAP-Gen/ReACC/Codegrag/full repo-aware-KG comparisons: rebut by noting these target different task settings (patch generation, completion, repository-level repair) rather than the standalone-function generation setting evaluated here (HumanEval/MBPP), so a fair head-to-head would require re-implementing them for a task they weren't designed for — cite this as the reason for scoping the comparison to retrieval-representation baselines (BM25, dense, and now cAST) plus an explicit "out of scope" acknowledgment (already added to Related Work per T1-writing).

### R-2: Full new "realistic" benchmark suite (residual of T3)
If a full LiveCodeBench/Fea-bench-scale addition isn't feasible: rebut by presenting the smaller API/doc-heavy supplementary slice (T3-experiment) as a targeted, resource-appropriate test of the specific "external knowledge" mechanism under dispute, rather than a full benchmark migration, and commit to the fuller evaluation as future work (already a natural fit for `discussion.tex` §6.5 Open Gaps, lines 292–297).

### R-3: Full retrieval-relevance annotation at scale (residual of T7)
If only a small annotated sample is feasible: rebut by framing the sample as a targeted mechanism check (consistent with how the paper already does topic-level and error-type breakdowns rather than exhaustive per-item analysis) and note the sample size/methodology transparently rather than claiming full coverage.

### R-4: Table 7 "inconsistency" (T17)
Directly rebut R3-S4 with the verified arithmetic: state in the response letter that the reported source table is internally consistent (show the row-sum check), and that the issue was very likely a PDF-rendering artifact in the review copy, now fixed by [reformatting per T17].

### R-5: Statistical claim in Threats.tex (T15)
Do not rebut this one — R3 is factually correct that "deterministic generation ⇒ no statistical test needed" is a non sequitur (determinism only guarantees reproducibility of the same run, not that a delta reflects a systematic effect). This must be fixed as a Writing correction (`experimental-setup.tex` line 17, `Threats.tex` line 10) plus the T15 experiment, not rebutted.

---

## Verification (coverage check)

All 26 individual review bullets (R1: 5 weaknesses; R2: 7 major + 4 minor; R3: 3 novelty + 5 soundness + 2 minor) map to a task above — see the Task Index table's "Reviewers" column, cross-checked bullet-by-bullet against `ResponseLetter.tex` during planning. No review point was dropped; duplicates (data leakage: R2-M7/R3-S2; pruning ablation: R1-W2/R2-M5; stronger baselines: R1-W1/R2-M6; case studies: R1-W5/R2-Minor3) are consolidated into single tasks addressing both.
