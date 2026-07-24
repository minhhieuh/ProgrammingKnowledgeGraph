# PKG Artifact Inventory

Two data sources:
- **Root folder** = newer re-run, **closed-source API models only** (Claude, GPT).
- **`ICLR_PKG/`** = original package, **open-source models + PythonAlpaca corpus + graph code**.

---

## Root folder — API-model experiments

| Path | Contents |
|------|----------|
| `humaneval_results/<model>_humaneval_<ts>/` | Clean HumanEval set, 4 API models. Per condition: `*_results.jsonl` (candidate code: `task_id, completion`) + `*_results.jsonl_results.jsonl` (verdict: `+result, passed`). 6 conditions: `no_rag, bm25, voyage_emb, voyage_block, voyage_func, reranked`. |
| `mbpp_results/<model>_mbpp_<ts>/` | Same, MBPP. Also `*_individual_results.json` (`task_id, generated_code, assertion_results`). |
| `mbpp_results__/` | Duplicate/newer MBPP run variant. |
| `experiment_results/` (44 dirs) | Full per-run logs. Each: `config.json`, `<cond>_results.jsonl`, `metrics.json`, `summary_report.md`, `detailed_metrics/`, and `prompt_logs/<cond>/<task>_prompt.json`. |
| `experiment_results/<run>/prompt_logs/.../*_prompt.json` | Full prompt sent per task: `system_prompt, user_prompt` (**retrieved context embedded inline**), `response`, token `metrics`. No retrieval scores. |
| `src/data/augmented_problems/*.jsonl` | Retrieved context per task per method (bm25 / voyage_code / blockwise / functionwise), fields `task_id, problem`. **Text only, no scores, no top-k split.** |
| `src/core/knowledge_programming_graph.py` | Graph construction (`GraphMacker`, `Node`, `Relation`). References `./datasets/python_alpaca.csv` (absent in root). No pruning. |
| `src/core/reranker.py` | Reranking (`rerank_one_solution`, `count_correct_answers`). |
| `src/core/code_generation.py`, `function_analyzer.py`, `function_enhancer.py` | Generation + AST analysis/enhancement. |
| `models/download.py` | HF download script: StarCoder2-7B, CodeLlama-7/13B, DeepSeek-Coder-7B, Llama-3.1-8B. **No result folders for these here.** |
| `*_results_summary.csv`, `*_pass_rates_only.csv`, `*_latex_table.tex` | Aggregate tables. |
| `augmented_data.json` | 500 MBPP embeddings (not the corpus). |
| `mbpp_test.jsonl`, `MBPP/`, `humaneval_evaluation_report.json` | Benchmark data + eval harness. |
| `docs/Implementation.md`, `docs/experiments.md` | **Pruning described here** (branch/DAG pruning) — spec only, no code. |

**Models with data (root):** `claude-3-haiku-20240307`, `claude-sonnet-4-20250514`, `gpt-4o`, `gpt-4o-mini`.

---

## `ICLR_PKG/` — original package (open-source models + corpus)

| Path | Contents |
|------|----------|
| `dataset/python_alpaca.csv` | **PythonAlpaca corpus snapshot, 281 MB (~5.4M rows).** Code-centric retrieval corpus. Recovered. |
| `dataset/python_codes.csv` | 40 MB, `instruction,input,output`. Secondary code corpus. |
| `results/<model>/` | **Open-source HumanEval results.** Models: `starcoder`, `codellama7b/13b/34b`, `deepseek`, `llama3`. Conditions: norag, bm25, block_level (bwrag), function_level (fwrag), graph (PKG). Verdict files `*.jsonl_results.jsonl` = `task_id, completion, result, passed`. |
| `results/MBPP/<model>/` | Same for MBPP (llama3, starcoder7b, deepseekcoder, codellama7b/13b). |
| `augmented_problems/*.jsonl` | Retrieved contexts (same schema as root, text only, no scores). |
| `knowledge_programming_graph.py`, `reranker.py`, `code_generation.py`, `function_analyzer.py`, `function_enhancer.py`, `prompt_utils.py` | Graph + retrieval + generation code. |
| `neo4j_graph.ipynb` | Graph build / Neo4j notebook. |
| `models/download.py` | Same open-model download script. |

**Coverage note:** open-source condition set is **uneven** — some models miss a condition (e.g. codellama13b/deepseek lack block_level in listing). Needs per-cell completeness check.

---

| Ask | Status | Where |
|-----|--------|-------|
| 1. Per-item pass/fail | **Have** (API + open-source) | root `*_results.jsonl_results.jsonl`, `*_individual_results.json`; `ICLR_PKG/results/` |
| 2. Candidate code per condition | **Have** (incl. NoRAG + baselines) | same files (`completion` / `generated_code`). One generation per task, no multi-sample. |
| 3. Retrieved contexts | **Partial** — text yes, **scores/top-k no** | `augmented_problems/*.jsonl`, `prompt_logs/*` |
| 4a. PythonAlpaca corpus | **Have** | `ICLR_PKG/dataset/python_alpaca.csv` |
| 4b. Graph construction code | **Have** | `knowledge_programming_graph.py` (+ ICLR_PKG) |
| 4c. Pruning code | **MISSING** | described in `docs/` only; zero in any `.py`/notebook. Pruning-on results exist; pruning-off + context-length deltas not runnable without it. |
| 4d. Tutorial (text-side) corpus | **MISSING** | not present anywhere |
| Open-source raw model weights | download script only | `models/download.py` |
