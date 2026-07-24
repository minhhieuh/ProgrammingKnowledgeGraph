# PKG Artifact Inventory

Maps the replication package to the 4 revision asks (per-item pass/fail, candidate code, retrieved contexts, corpus + graph/pruning code). Paths are the **post-reorg** layout. The per-run result trees ship inside `results_data.zip` — `unzip` it to get the `results/` paths below.

Two logical tracks, now unified in one tree:
- **API models** (Claude, GPT) — pipeline `src/`, results `results/api_models/`.
- **Open-source models** (StarCoder2, CodeLlama, DeepSeek-Coder, Llama-3) — pipeline `opensource_pipeline/`, results `results/opensource/`.

---

## Where things live

| Path | Contents |
|------|----------|
| `results/api_models/<model>_humaneval_<ts>/` | Clean HumanEval set, 4 API models. Per condition: `*_results.jsonl` (candidate code: `task_id, completion`) + `*_results.jsonl_results.jsonl` (verdict: `+result, passed`). 6 conditions: `no_rag, bm25, voyage_emb, voyage_block, voyage_func, reranked`. |
| `results/api_models/<model>_mbpp_<ts>/` | Same, MBPP. Also `*_individual_results.json` (`task_id, generated_code, assertion_results`). |
| `results/opensource/<model>/` | **Open-source HumanEval results.** Models: `starcoder`, `codellama7b/13b/34b`, `deepseek`, `llama3`. Conditions: norag, bm25, block_level (bwrag), function_level (fwrag), graph (PKG). Verdict files `*.jsonl_results.jsonl` = `task_id, completion, result, passed`. |
| `results/opensource/MBPP/<model>/` | Same for MBPP (llama3, starcoder7b, deepseekcoder, codellama7b/13b). |
| `results/summary/` | Aggregate tables: `*_results_summary.csv`, `*_pass_rates_only.csv`, `*_latex_table.tex`, `humaneval_evaluation_report.json`. |
| `data/augmented_problems/*.jsonl` | Retrieved context per task per method (bm25 / voyage_code / blockwise / functionwise), fields `task_id, problem` — **API pipeline**. Text only, no scores, no top-k split. |
| `opensource_pipeline/augmented_problems/*.jsonl` | Same, **open-source pipeline** (slightly different content/naming). |
| `data/corpora/python_alpaca.csv` | **PythonAlpaca corpus snapshot, 268 MB (~5.4M rows).** Code-centric retrieval corpus. Git LFS. |
| `data/corpora/python_codes.csv` | 40 MB, `instruction,input,output`. Secondary code corpus. Git LFS. |
| `src/core/knowledge_programming_graph.py`, `opensource_pipeline/knowledge_programming_graph.py` | Graph construction (`GraphMacker`, `Node`, `Relation`). Reads `data/corpora/python_alpaca.csv`. **No pruning.** |
| `src/core/reranker.py`, `opensource_pipeline/reranker.py` | Reranking (`rerank_one_solution`, `count_correct_answers`). Key now via `VOYAGE_API_KEY` env var. |
| `src/core/{code_generation,function_analyzer,function_enhancer}.py`, `opensource_pipeline/*.py` | Generation + AST analysis/enhancement. |
| `opensource_pipeline/neo4j_graph.ipynb` | Graph build / Neo4j notebook. |
| `scripts/models/download.py`, `opensource_pipeline/models/download.py` | HF download script: StarCoder2-7B, CodeLlama-7/13B, DeepSeek-Coder-7B, Llama-3.1-8B. **Weights not shipped.** |
| `data/benchmarks/mbpp_test.jsonl`, `MBPP/` | Benchmark data + vendored eval harness (`from MBPP.human_eval…`). |
| `data/augmented_data.json` | 500 MBPP embeddings (not the corpus). |
| `docs/Implementation.md`, `docs/experiments.md` | **Pruning described here** (branch/DAG pruning) — spec only, no code. |

**API models with data:** `claude-3-haiku-20240307`, `claude-sonnet-4-20250514`, `gpt-4o`, `gpt-4o-mini`.

**Coverage note:** the open-source condition set is **uneven** — some models miss a condition (e.g. codellama13b/deepseek lack block_level). Needs a per-cell completeness check.

> **Removed in cleanup:** the raw `experiment_results/` dump (44 dirs, incl. `prompt_logs/<cond>/<task>_prompt.json` that held the full prompt with retrieved context inline) was deleted as redundant. Retrieved-context text now comes only from the `augmented_problems/` files below.

---

## Status vs. the 4 asks

| Ask | Status | Where |
|-----|--------|-------|
| 1. Per-item pass/fail | **Have** (API + open-source) | `results/api_models/*_results.jsonl_results.jsonl`, `*_individual_results.json`; `results/opensource/` |
| 2. Candidate code per condition | **Have** (incl. NoRAG + baselines) | same files (`completion` / `generated_code`). One generation per task, no multi-sample. |
| 3. Retrieved contexts | **Partial** — text yes, **scores/top-k no** | `data/augmented_problems/*.jsonl`, `opensource_pipeline/augmented_problems/*.jsonl` |
| 4a. PythonAlpaca corpus | **Have** | `data/corpora/python_alpaca.csv` |
| 4b. Graph construction code | **Have** | `src/core/knowledge_programming_graph.py`, `opensource_pipeline/knowledge_programming_graph.py` |
| 4c. Pruning code | **MISSING** | described in `docs/` only; zero in any `.py`/notebook. Pruning-on results exist; pruning-off + context-length deltas not runnable without it. |
| 4d. Tutorial (text-side) corpus | **MISSING** | not present anywhere |
| Open-source raw model weights | download script only | `scripts/models/download.py` |
