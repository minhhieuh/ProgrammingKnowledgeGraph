# Context-Augmented Code Generation Using Programming Knowledge Graphs

This repository accompanies the research paper ["Context-Augmented Code Generation Using Programming Knowledge Graphs"](https://arxiv.org/pdf/2410.18251) and provides the implementation for a novel framework leveraging a Programming Knowledge Graph (PKG) to enhance code generation using Retrieval-Augmented Generation (RAG).

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Neo4j, APOC and Graph Data Science for knowledge graph management

### Installation
```bash
# Using conda (recommended)
conda env create -f config/environment.yml
conda activate pkg-experiments

# Or using pip
pip install -r config/requirements.txt
```

### Basic Usage
```bash
# Run a basic experiment with Claude
python src/experiments/experiment_runner.py --model claude-3-haiku-20240307 --benchmark humaneval --retrieval_method block_pkg

# Generate PKG from code
python src/core/knowledge_programming_graph.py
```

## 📁 Repository Structure

```
├── src/                    # Source code
│   ├── core/              # Core PKG implementation
│   ├── experiments/       # Experiment runners and utilities
│   ├── data/             # Datasets and data processing
│   ├── notebooks/        # Jupyter notebooks
│   └── scripts/          # Utility scripts
├── docs/                 # Documentation
├── config/              # Configuration files
├── web/                 # Web interfaces
├── models/              # Model files
├── images/              # Documentation images
└── experiment_results/  # Experiment outputs
```

## 📖 Documentation

- **[Implementation Details](docs/Implementation.md)** - Technical implementation guide
- **[Experiments Guide](docs/experiments.md)** - Experimental methodology
- **[Experiment Runner](docs/EXPERIMENT_RUNNER_README.md)** - Modern experiment runner
- **[Setup Guide](docs/CONDA_SETUP_GUIDE.md)** - Detailed setup instructions

## 🎯 Key Features

- **PKG-Based Retrieval**: Semantic representation at fine granularity levels
- **Tree Pruning**: Improved retrieval precision by pruning irrelevant branches
- **Re-ranking Solutions**: Reducing hallucination in generated code
- **FIM Enhancement**: Automatic code augmentation with metadata

## 📊 Results

- Up to **20% improvement** in pass@1 accuracy
- **34% better** performance than state-of-the-art on MBPP benchmark

## 🔬 Citation

```bibtex
@article{saberi2024context,
  title={Context-Augmented Code Generation Using Programming Knowledge Graphs},
  author={Iman Saberi and Fatemeh Fard},
  year={2024},
  journal={arXiv preprint arXiv:2410.18251}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details. 