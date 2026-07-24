# Modern Experiment Runner for PKG Framework

This experiment runner allows you to reproduce the experiments from the "Context-Augmented Code Generation Using Programming Knowledge Graphs" paper using Anthropic Claude and OpenAI GPT models, while preserving all experimental settings.

## Features

- **Anthropic Claude & OpenAI GPT Support**: Support for Claude Sonnet/Haiku and GPT-4/GPT-3.5-turbo models
- **Preserved Experimental Settings**: Maintains all settings from the original paper (temperature=0, max_tokens=512, pass@1 evaluation)
- **Complete Retrieval Pipeline**: Supports all augmentation types (No RAG, Block-PKG, Func-PKG, BM25)
- **Re-ranking Mechanism**: Implements the 3-step re-ranking process with AST analysis and semantic similarity
- **Comprehensive Evaluation**: Pass@1 metrics with syntactic and runtime validation
- **Flexible Configuration**: Easy-to-use command-line interface with extensive options
- **Rich Output**: Detailed results, metrics, and summary reports

## Installation

### Option 1: Conda Environment (Recommended)

The easiest way to set up the environment is using conda:

1. **Run the automated setup script:**
```bash
./setup_conda_env.sh
```

2. **Or create the environment manually:**
```bash
# Create environment from YAML file
conda env create -f environment.yml

# Activate the environment
conda activate pkg-experiments
```

3. **Set up API keys:**
```bash
export ANTHROPIC_API_KEY="your-anthropic-key"
export OPENAI_API_KEY="your-openai-key"
export VOYAGE_API_KEY="your-voyage-key"  # For re-ranking
```

### Option 2: Pip Installation

If you prefer to use pip or don't have conda:

1. **Install dependencies:**
```bash
pip install -r requirements_experiment.txt
```

2. **Set up API keys:**

Create a `.env` file or set environment variables:
```bash
export ANTHROPIC_API_KEY="your-anthropic-key"
export OPENAI_API_KEY="your-openai-key"
export VOYAGE_API_KEY="your-voyage-key"  # For re-ranking
```

## Quick Start

### Running with Claude Sonnet (recommended)

```bash
# Make sure the conda environment is activated
conda activate pkg-experiments

python experiment_runner.py \
    --model-name claude-3-sonnet-20240229 \
    --model-type anthropic \
    --benchmark humaneval \
    --verbose
```

### Running with GPT-4

```bash
python experiment_runner.py \
    --model-name gpt-4 \
    --model-type openai \
    --benchmark humaneval \
    --verbose
```

### Running with GPT-3.5-turbo

```bash
python experiment_runner.py \
    --model-name gpt-3.5-turbo \
    --model-type openai \
    --benchmark humaneval \
    --verbose
```

### Running with specific augmentation types only

```bash
python experiment_runner.py \
    --model-name claude-3-sonnet-20240229 \
    --model-type anthropic \
    --augmentation-types no_rag voyage_func \
    --benchmark humaneval
```

## Command Line Options

### Required Arguments
- `--model-name`: Model name (e.g., `claude-3-sonnet-20240229`, `gpt-4`, `gpt-3.5-turbo`)
- `--model-type`: Provider type (`anthropic` or `openai`)

### Optional Arguments

**Experiment Configuration:**
- `--benchmark`: Benchmark to use (`humaneval`, `mbpp`) [default: humaneval]
- `--augmentation-types`: List of augmentation types to test [default: all]
  - `no_rag`: No retrieval augmentation
  - `voyage_func`: Function-wise retrieval using Voyage embeddings
  - `voyage_block`: Block-wise retrieval using Voyage embeddings
  - `bm25`: BM25 sparse retrieval
- `--disable-reranking`: Disable the re-ranking mechanism

**Model Parameters (preserve paper settings):**
- `--temperature`: Generation temperature [default: 0.0 for greedy decoding]
- `--max-tokens`: Maximum tokens to generate [default: 512]
- `--timeout`: API call timeout in seconds [default: 30]

**Output Configuration:**
- `--output-dir`: Directory to save results [default: experiment_results]
- `--experiment-name`: Custom experiment name [default: auto-generated]
- `--verbose`: Enable detailed logging

## Usage Examples

### 1. Complete Experiment with Claude Sonnet

```bash
conda activate pkg-experiments

python experiment_runner.py \
    --model-name claude-3-sonnet-20240229 \
    --model-type anthropic \
    --benchmark humaneval \
    --experiment-name "claude_sonnet_humaneval_full" \
    --verbose
```

This will run all four augmentation types (no_rag, voyage_func, voyage_block, bm25) and perform re-ranking.

### 2. Quick Test with Limited Augmentation Types

```bash
python experiment_runner.py \
    --model-name claude-3-sonnet-20240229 \
    --model-type anthropic \
    --augmentation-types no_rag voyage_func \
    --benchmark humaneval \
    --experiment-name "claude_quick_test"
```

### 3. MBPP Benchmark Experiment

```bash
python experiment_runner.py \
    --model-name gpt-4 \
    --model-type openai \
    --benchmark mbpp \
    --experiment-name "gpt4_mbpp_experiment"
```

### 4. GPT-3.5-turbo Experiment

```bash
python experiment_runner.py \
    --model-name gpt-3.5-turbo \
    --model-type openai \
    --benchmark humaneval \
    --experiment-name "gpt35_humaneval"
```

### 5. Disable Re-ranking

```bash
python experiment_runner.py \
    --model-name claude-3-sonnet-20240229 \
    --model-type anthropic \
    --disable-reranking \
    --experiment-name "claude_no_reranking"
```

### 6. Custom Output Directory

```bash
python experiment_runner.py \
    --model-name claude-3-sonnet-20240229 \
    --model-type anthropic \
    --output-dir ./my_experiments \
    --experiment-name "claude_custom_location"
```

## Output Structure

After running an experiment, you'll find the following files in your output directory:

```
experiment_results/
└── claude-3-sonnet-20240229_humaneval_1703123456/
    ├── config.json                    # Experiment configuration
    ├── metrics.json                   # Numerical results
    ├── summary_report.md              # Human-readable summary
    ├── no_rag_results.jsonl           # Individual results per augmentation type
    ├── voyage_func_results.jsonl
    ├── voyage_block_results.jsonl
    ├── bm25_results.jsonl
    └── reranked_results.jsonl         # Re-ranked solutions (if enabled)
```

### Key Output Files

1. **`summary_report.md`**: Human-readable summary with pass@1 scores and configuration
2. **`metrics.json`**: Detailed metrics for each augmentation type
3. **`*_results.jsonl`**: Complete results including generated code, test outcomes, and context
4. **`reranked_results.jsonl`**: Best solutions selected by the re-ranking mechanism

## Understanding the Results

### Metrics Explained

- **Pass@1**: Percentage of problems where the first generated solution passes all tests
- **Syntax Accuracy**: Percentage of generated solutions that are syntactically valid Python
- **Passed/Total**: Number of problems solved out of total problems

### Example Output

```
============================================================
EXPERIMENT SUMMARY
============================================================
Model: claude-3-sonnet-20240229
Benchmark: humaneval
------------------------------------------------------------
no_rag         : 65.2% (107/164)
voyage_func    : 72.0% (118/164)
voyage_block   : 69.5% (114/164)
bm25           : 67.1% (110/164)
reranked       : 75.6% (124/164)
------------------------------------------------------------
Results saved to: experiment_results/claude-3-sonnet-20240229_humaneval_1703123456
============================================================
```

## Supported Models

### Anthropic Models
- `claude-3-sonnet-20240229` (recommended)
- `claude-3-haiku-20240307`
- `claude-3-opus-20240229`

### OpenAI Models
- `gpt-4` (recommended)
- `gpt-4-turbo`
- `gpt-3.5-turbo`

## Experimental Settings (Paper Compliance)

The experiment runner preserves all key settings from the original paper:

- **Temperature**: 0.0 (greedy decoding)
- **Max Tokens**: 512
- **Evaluation Metric**: Pass@1
- **Retrieval Methods**: Block-PKG, Func-PKG, BM25, No RAG
- **Re-ranking**: 3-step process (AST analysis → Runtime execution → Semantic similarity)
- **Benchmarks**: HumanEval and MBPP

## Environment Management

### Conda Commands

```bash
# Activate environment
conda activate pkg-experiments

# Deactivate environment
conda deactivate

# List installed packages
conda list

# Update environment from YAML
conda env update -f environment.yml

# Remove environment
conda env remove -n pkg-experiments

# Export current environment
conda env export > environment_backup.yml
```

## Troubleshooting

### Common Issues

1. **Conda Environment Not Found**
   ```
   CondaEnvironmentError: cannot locate environment: pkg-experiments
   ```
   Solution: Run `./setup_conda_env.sh` or `conda env create -f environment.yml`

2. **API Key Not Found**
   ```
   Error: API key not provided. Set ANTHROPIC_API_KEY environment variable
   ```
   Solution: Set the appropriate environment variable or use `--api-key`

3. **Missing Augmented Data**
   ```
   Warning: Augmented data file not found: augmented_problems/humaneval_function_wise_relevant_context.jsonl
   ```
   Solution: Ensure the `augmented_problems/` directory contains the required JSONL files

4. **Import Error for LLM Provider**
   ```
   ImportError: anthropic package not installed
   ```
   Solution: Activate the conda environment: `conda activate pkg-experiments`

5. **MBPP Dataset Not Found**
   ```
   Error: MBPP dataset not found. Please ensure mbpp.csv is available.
   ```
   Solution: Download and place the MBPP dataset as `mbpp.csv` in the project root

### Performance Tips

1. **Always activate the conda environment** before running experiments
2. **Use `--verbose` for debugging** but remove it for production runs
3. **Start with fewer augmentation types** (`--augmentation-types no_rag voyage_func`) for quick testing
4. **Set appropriate timeouts** based on your model's response time
5. **Monitor API usage** to avoid rate limits

## Extending the Framework

### Adding New Models

To add support for new models from existing providers:

1. Simply use the new model name with the appropriate provider type
2. For example, if Anthropic releases a new model, just use it with `--model-type anthropic`

### Custom Evaluation Metrics

You can extend the `calculate_metrics()` function to include additional metrics like:
- Pass@k for k > 1
- Code quality metrics
- Execution time analysis
- Memory usage tracking

## Citation

If you use this experiment runner in your research, please cite the original paper:

```bibtex
@article{pkg2024,
  title={Context-Augmented Code Generation Using Programming Knowledge Graphs},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## License

This project follows the same license as the original PKG framework.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the original paper and implementation
3. Open an issue in the repository with detailed information about your setup and error messages 