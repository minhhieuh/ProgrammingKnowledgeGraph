# Conda Environment Setup Guide

This guide will help you set up a conda environment for the PKG Experiments framework.

## 🚀 Quick Setup (Recommended)

### 1. Automated Setup
Run the setup script to create everything automatically:

```bash
./setup_conda_env.sh
```

### 2. Activate Environment
```bash
conda activate pkg-experiments
```

### 3. Set API Keys
```bash
export ANTHROPIC_API_KEY="your-anthropic-key"
export OPENAI_API_KEY="your-openai-key" 
export VOYAGE_API_KEY="your-voyage-key"  # Optional, for re-ranking
```

### 4. Test Installation
```bash
python experiment_runner.py --help
```

## 📋 Manual Setup

If you prefer to set up manually:

### 1. Create Environment
```bash
conda env create -f environment.yml
```

### 2. Activate Environment
```bash
conda activate pkg-experiments
```

### 3. Verify Installation
```bash
conda list | grep -E "(anthropic|openai|human-eval)"
```

## 🔧 Daily Usage

### Quick Activation
You can source the activation script for convenience:
```bash
source activate_env.sh
```

This will:
- Activate the conda environment
- Check your API key setup
- Show you next steps

### Manual Activation
```bash
conda activate pkg-experiments
```

## 📦 Environment Details

The conda environment includes:

### Core Dependencies
- Python 3.9
- NumPy 1.26.4
- Pandas 2.2.2
- tqdm (progress bars)

### PKG-Specific Packages
- human-eval==1.0.3
- voyageai==0.2.3
- astor==0.8.1

### LLM Providers
- anthropic>=0.25.0
- openai>=1.0.0

### Development Tools (Optional)
- Jupyter notebook
- pytest, black, flake8

## 🛠️ Environment Management

### Common Commands
```bash
# Activate environment
conda activate pkg-experiments

# Deactivate environment
conda deactivate

# List packages
conda list

# Update environment
conda env update -f environment.yml

# Remove environment
conda env remove -n pkg-experiments

# Export current state
conda env export > my_environment.yml
```

## 🔍 Troubleshooting

### Environment Not Found
```bash
# If you get "environment not found" error
conda env list  # Check if pkg-experiments exists
./setup_conda_env.sh  # Recreate if needed
```

### Package Import Errors
```bash
# Make sure environment is activated
conda activate pkg-experiments

# Check if packages are installed
conda list | grep anthropic
conda list | grep openai
```

### API Key Issues
```bash
# Check if keys are set
echo $ANTHROPIC_API_KEY
echo $OPENAI_API_KEY

# Set keys in current session
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
```

## 🎯 Next Steps

Once your environment is set up:

1. **Run a quick test:**
   ```bash
   python experiment_runner.py \
       --model-name claude-3-sonnet-20240229 \
       --model-type anthropic \
       --augmentation-types no_rag \
       --verbose
   ```

2. **Try the interactive example:**
   ```bash
   python example_claude_experiment.py
   ```

3. **Read the full documentation:**
   ```bash
   cat EXPERIMENT_RUNNER_README.md
   ```

## 📝 Environment File Contents

The `environment.yml` file contains:

```yaml
name: pkg-experiments
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - numpy=1.26.4
  - pandas=2.2.2
  - tqdm>=4.64.0
  - jupyter
  - ipykernel
  - pytest>=7.0.0
  - black>=22.0.0
  - flake8>=5.0.0
  - pip
  - pip:
    - human-eval==1.0.3
    - voyageai==0.2.3
    - astor==0.8.1
    - anthropic>=0.25.0
    - openai>=1.0.0
```

This ensures consistent package versions across different systems and makes the environment reproducible. 