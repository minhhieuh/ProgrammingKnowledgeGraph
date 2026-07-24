#!/bin/bash

# Setup script for PKG Experiments Conda Environment
# This script creates a conda environment and sets up the PKG experiment runner

set -e  # Exit on any error

echo "🚀 Setting up PKG Experiments Conda Environment"
echo "================================================"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed or not in PATH"
    echo "   Please install Anaconda or Miniconda first:"
    echo "   https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✅ Conda found: $(conda --version)"

# Check if environment.yml exists
if [ ! -f "config/environment.yml" ]; then
    echo "❌ config/environment.yml not found"
    echo "   Please make sure you're running this script from the project root directory"
    exit 1
fi

echo "📋 Creating conda environment from config/environment.yml..."

# Create the environment
conda env create -f config/environment.yml

echo "✅ Environment created successfully!"

# Provide activation instructions
echo ""
echo "🎯 Next Steps:"
echo "=============="
echo "1. Activate the environment:"
echo "   conda activate pkg-experiments"
echo ""
echo "2. Set up your API keys:"
echo "   export ANTHROPIC_API_KEY='your-anthropic-key'"
echo "   export OPENAI_API_KEY='your-openai-key'"
echo "   export VOYAGE_API_KEY='your-voyage-key'"
echo ""
echo "3. Run a test experiment:"
echo "   python experiment_runner.py --model-name claude-3-sonnet-20240229 --model-type anthropic --augmentation-types no_rag --verbose"
echo ""
echo "4. Or run the interactive example:"
echo "   python example_claude_experiment.py"
echo ""
echo "📖 For more details, see EXPERIMENT_RUNNER_README.md"
echo ""
echo "🎉 Setup complete! Happy experimenting!" 