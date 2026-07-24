# Context-Augmented Code Generation Using Programming Knowledge Graphs (PKG)

Replication package for the PKG framework — Retrieval-Augmented Generation for code, built on a Programming Knowledge Graph.
Accompanies ["Context-Augmented Code Generation Using Programming Knowledge Graphs"](https://arxiv.org/pdf/2410.18251).

Two experiment tracks share the same data/graph stages:
- **API models** (Claude, GPT) — driven by `src/` (`experiment_runner`).
- **Open-source models** (StarCoder2, CodeLlama, DeepSeek-Coder, Llama-3) — driven by `opensource_pipeline/`.

## Repository structure

```
├── README.md
├── requirements.txt / requirements_experiment.txt / environment.yml
├── .env.example              # copy to .env, add your keys
├── docs/                     # writeups + this package's audit
│   ├── Implementation.md, experiments.md
│   ├── EXPERIMENT_STEPS.md       # full step-by-step pipeline
│   ├── ARTIFACT_INVENTORY.md     # what data exists / is missing
│   └── images/
├── data/
│   ├── corpora/              # python_alpaca.csv, python_codes.csv (retrieval corpus)
│   ├── benchmarks/           # mbpp_test.jsonl, model_pricing.csv
│   └── augmented_problems/   # cached retrieved contexts (API pipeline)
├── MBPP/                     # vendored MBPP eval harness (human_eval package)
├── src/                      # API-model pipeline (Python package `src.*`)
│   ├── core/  experiments/  data/  scripts/  notebooks/
├── opensource_pipeline/      # open-source model pipeline (flat scripts, run from this dir)
│   ├── knowledge_programming_graph.py, code_generation.py, reranker.py, ...
│   └── augmented_problems/   # cached retrieved contexts (open-source pipeline)
├── analysis/                 # evaluation + aggregation scripts
│   ├── summarize_humaneval_results.py, summarize_mbpp_results.py
│   ├── mbpp_eval.py, run_humaneval_evaluation.py, csv_to_latex_converter.py
├── results/
│   ├── api_models/           # humaneval_results/, mbpp_results/ (per-item pass/fail + code)
│   ├── opensource/           # open-source model results
│   └── summary/              # aggregate CSVs, LaTeX tables, reports
├── scripts/                  # run.sh, setup.sh, run_experiment.py, models/
└── logs/                     # evaluation run logs
```

## Quick start

### Prerequisites
- Python 3.9+
- Neo4j + APOC + Graph Data Science (for building/serving the PKG)

### Install
```bash
conda env create -f environment.yml && conda activate pkg-experiments
# or
pip install -r requirements.txt
```

### Keys
```bash
cp .env.example .env      # then fill in ANTHROPIC / VOYAGE / OPENAI keys
```

## Reproducing the results

All commands run from the repo root.

**1. Generate — API models**
```bash
python -m src.experiments.experiment_runner \
  --model-name gpt-4o-mini --model-type openai \
  --benchmark mbpp --augmentation-types no_rag voyage_func voyage_block bm25 voyage_emb --verbose
# writes to experiment_results/<run>/ ; curated runs live in results/api_models/
```

**1b. Generate — open-source models**
```bash
cd opensource_pipeline
python code_generation.py --model_type codellama_7b --dest_path <out>.jsonl
```

**2. Aggregate to the paper tables** (reads `results/api_models/`, writes `results/summary/`)
```bash
python analysis/summarize_humaneval_results.py
python analysis/summarize_mbpp_results.py
python analysis/csv_to_latex_converter.py
```

## License
MIT — see LICENSE.
