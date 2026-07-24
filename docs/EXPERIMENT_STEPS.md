# PKG Experiment — Step by Step

End-to-end pipeline as actually implemented in this repo. Two tracks share stages 1–4:

- **Track A (open-source, `ICLR_PKG/`)** — local models on A100, `code_generation.py`.
- **Track B (API models, root)** — Claude/GPT via `src/experiments/experiment_runner.py`.

Stages 1–3 run **once** (offline, build retrieval assets). Stages 4–7 run **per model × benchmark × condition**.

---

## Stage 1 — Build the corpora

| Source | Path | Size |
|--------|------|------|
| PythonAlpaca (code-centric, ~143K QA pairs → 115K functions) | `ICLR_PKG/dataset/python_alpaca.csv` | 281 MB |
| Extracted Python functions | `ICLR_PKG/dataset/python_codes.csv` | 40 MB |
| Tutorials (text-centric, 76.6K tutorials) | **MISSING** | — |

Code path: `knowledge_programming_graph.py: main()` reads `./datasets/python_alpaca.csv`, applies
`FunctionAnalyzer.extract_python_code` then `.get_function_blocks` to pull function blocks from the `output` column.

---

## Stage 2 — Build the PKG (graph construction)

Driver: `ICLR_PKG/knowledge_programming_graph.py` (also `src/core/`). Loops the corpus in chunks of 1000.

Per function:
1. `analyzer.get_function_name(code)` — function name.
2. `analyzer.remove_docstring_from_function(code)` — strip docstrings.
3. `analyzer.get_code_blocks(pure_code)` — extract blocks (`if`/`for`/`with`/`try`) + block metadata.
4. `enhancer.generate_docstring(code)` — LLM-generated docstring enrichment.
5. `analyzer.extract_relations(block_info)` — sibling/parent structure.
6. Embed everything: `get_embeddings_from_dict(...)` (VoyageCode2 in the paper).
7. `GraphMacker.generate_nodes(...)` — 3 node types: `func_name`, `implementation`, `code_block`.
8. `GraphMacker.generate_relations(...)` — structural edges: func_name→implementation (`implementation`), implementation→first block (`child`), parent block→child block (`child`).
9. `GraphMacker.create_semantic_relations(...)` — similarity edges where cosine > **0.8**.
10. `save_nodes()` / `save_relatios()` → JSON.

Then load nodes+relations into **Neo4j 5.20.0** with APOC (`neo4j_graph.ipynb`) and build the vector index.

Reported scale: PythonAlpaca PKG = 425,058 nodes / 434,518 relations; Tutorials PKG = 288,583 nodes / 287,936 relations.

---

## Stage 3 — Retrieve context per benchmark task (offline, cached)

For each benchmark task: embed query → vector search the PKG → take best node → branch-prune → append to prompt. Broken down:

### 3.1 What's in the graph to search

From Stage 2, every corpus function is shredded into nodes, each with its own embedding: `func_name`, `implementation` (whole body), and one `code_block` per `if`/`for`/`with`/`try`. Blocks link parent→child, so a single function forms a small tree — implementation on top, its blocks beneath, nested blocks below those.

### 3.2 Embed the query

The benchmark task prompt (e.g. the HumanEval docstring) goes through Voyage-Code-2 → one vector.

### 3.3 Vector search → `v_best`

Cosine similarity between query vector and every node vector in the Neo4j vector index. Highest scorer = `v_best`. Which nodes are searched depends on condition:

- **Func-PKG** searches only `implementation` nodes → best match is an entire function
- **Block-PKG** searches only `code_block` nodes → best match is a single block

### 3.4 Why pruning is needed

`v_best` is a real function written for a *different* purpose — closest match, not a perfect one. It carries extra logic irrelevant to the query. Passing all of it wastes context and can mislead the model into copying the irrelevant parts.

### 3.5 Branch pruning

Treat `v_best` as a tree/DAG of its child blocks, then:

1. Generate candidate versions, each with one branch removed
2. Embed each pruned candidate
3. Cosine-compare each against the **query** embedding
4. Keep the highest-scoring version → `n_pruned`

Intuition: if removing a branch makes the remaining code *more* similar to the request, that branch was noise.

Docs' own example: query counts "boring" sentences; retrieved function counts both "boring" and "exciting" sentences; pruning drops the "exciting" branch.

### 3.6 Augment

`n_pruned` content is appended to the original query → `q_augmented`. Track A wrapper is literally:

```
helper code 1:
<pruned code>
End of helper section.
```

That combined prompt goes to the generator. This is exactly what got cached into `augmented_problems/*.jsonl`.

> **Caveat:** 3.4–3.5 are the paper's *specification* (`docs/Implementation.md`), **not** verified against code — the pruning implementation isn't in this repo. Only post-pruning output text is on disk. So the exact candidate strategy (one branch at a time? all subsets? greedy iterative?) can't be confirmed from what's here.

### 3.7 Cached outputs

Results cached to `augmented_problems/` — **retrieval never re-runs during generation**:

| Condition | File | What it searches |
|-----------|------|------------------|
| `voyage_func` (Func-PKG) | `{benchmark}_function_wise_relevant_context.jsonl` | `v_impl` nodes — whole function |
| `voyage_block` (Block-PKG) | `{benchmark}_blockwise_relevant_context.jsonl` | `v_block` nodes — code blocks |
| `bm25` | `bm25_relevant_context_{benchmark}.jsonl` | sparse baseline (`rank_bm25`) |
| `voyage_emb` | `voyage_code_relevant_context_{benchmark}.jsonl` | plain dense retrieval (Voyage-Code-2), no graph |
| `no_rag` | — | no context |

Schema: `{task_id, problem}`. **Only the retained context text is stored — no top-k list, no scores.**

> ⚠️ Pruning code is **not in the repo**. Stage-3 outputs are post-pruning, so pruning-off cannot be reproduced without reimplementing it.

---

## Stage 4 — Generate code

Fixed decoding both tracks: **greedy, temperature 0, max_new_tokens 512, 1 sample per task** (pass@1).

### Track A — open-source
```bash
python code_generation.py \
  --model_type {codellama_7b|codellama_13b|codellama_34b|starcoder2_7b|llama3_8b|deepseekcoder_7b} \
  --dest_path <out>.jsonl
  # augmentation: voyage_func | voyage_block | bm25 | no_rag
```
Flow: `load_model()` (optional 4-bit NF4 quant) → `make_augmented_data()` / `make_augmented_bm25_data()` wraps context as `"helper code 1:\n...\nEnd of helper section."` → per-model prompt (`codellama_prompt` / `starcoder_prompt` / `llama3_prompt` / `deepseek_prompt`) → generate → `extract_python_code()` pulls `[PYTHON]...[/PYTHON]` or ``` fences → `write_jsonl`.

### Track B — API models
```bash
python -m src.experiments.experiment_runner \
  --model-name gpt-4o-mini --model-type openai \
  --benchmark mbpp --augmentation-types voyage_emb --verbose
```
Key flags: `--model-type {anthropic,openai}`, `--benchmark {humaneval,mbpp}`, `--augmentation-types` (any of the 5), `--temperature 0.0`, `--max-tokens 512`, `--disable-reranking`, `--output-dir experiment_results`. Runs 28 worker processes. Writes per-run `config.json`, `<cond>_results.jsonl`, `metrics.json`, `detailed_metrics/`, and `prompt_logs/<cond>/<task>_prompt.json` (full prompt + response + token cost).

Output schema both tracks: `{task_id, completion}`.

---

## Stage 5 — Evaluate (pass/fail per item)

- HumanEval: `run_humaneval_evaluation.py` → `evaluate_functional_correctness`
- MBPP: `mbpp_eval.py` → `MBPP.human_eval.evaluation.evaluate_functional_correctness`

Produces `<file>.jsonl_results.jsonl` = `{task_id, completion, result, passed}` — **the per-item record**. MBPP also emits `*_individual_results.json` with `assertion_results`.

---

## Stage 6 — Re-rank across conditions

`reranker.py` — picks one final answer from the 4 candidates (norag, bm25, bwrag, fwrag) for each task.

```bash
python reranker.py --evaluation {human_eval|mbpp} \
  --norag_path <> --bwrag_path <> --fwrag_path <> --bm25rag_path <> \
  --output_path <>
```

1. **AST filter** — `FirstFunctionExtractor` / `remove_comments_and_docstrings` drop syntactically broken candidates.
2. **Runtime filter** — execute under `timeout_handler`, drop erroring candidates.
3. **Semantic pick** — embed query + each candidate with `voyage-code-2`, `cosine_similarity`, take argmax (`rerank_one_solution`).

`count_correct_answers()` also computes the **`ideal`** figure = task counted correct if *any* of the 4 conditions passed — the oracle upper bound.

---

## Stage 7 — Aggregate

`summarize_humaneval_results.py`, `summarize_mbpp_results.py`, `csv_to_latex_converter.py` → `*_results_summary.csv`, `*_pass_rates_only.csv`, `*_latex_table.tex`.

---

## Run matrix

- **Models:** CodeLlama-7B/13B/34B, StarCoder2-7B, DeepSeek-Coder-7B, Llama-3.1-8B (Track A); claude-3-haiku, claude-sonnet-4, gpt-4o, gpt-4o-mini (Track B).
- **Benchmarks:** HumanEval, MBPP.
- **Conditions:** no_rag, bm25, voyage_emb, voyage_block (Block-PKG), voyage_func (Func-PKG), reranked.
- **Metric:** pass@1, greedy, 1 sample/task.

---

## Reproduction blockers

| Stage | Blocker |
|-------|---------|
| 1 | Tutorials corpus missing — text-centric / JSON-PKG track not reproducible |
| 2–3 | Branch-pruning code absent — cannot run pruning-off or measure context-length delta |
| 3 | No retrieval scores/top-k logged — score-based retrieval-quality analysis needs re-retrieval |
| 4 | 1 sample per task — no within-condition multi-sample oracle (cross-condition `ideal` is available) |

---

## Security note

`ICLR_PKG/reranker.py:133` contains a **hardcoded VoyageAI API key** in `init_voyageai_embedder()`. Rotate that key and move it to an environment variable before this folder is shared or published.
