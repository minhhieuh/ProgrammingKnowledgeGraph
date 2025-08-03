#!/bin/bash

# Activation script for PKG Experiments
# Source this file to activate the environment and set up common variables
# Usage: source activate_env.sh

echo "🔧 Activating PKG Experiments Environment"
echo "========================================="

# Activate conda environment
if command -v conda &> /dev/null; then
    echo "📦 Activating conda environment 'pkg-experiments'..."
    conda activate pkg-experiments
    
    if [[ "$CONDA_DEFAULT_ENV" == "pkg-experiments" ]]; then
        echo "✅ Environment activated successfully!"
    else
        echo "❌ Failed to activate environment. Make sure it exists:"
        echo "   ./setup_conda_env.sh"
        return 1
    fi
else
    echo "❌ Conda not found. Please install conda or use pip installation."
    return 1
fi

# Check for API keys and provide guidance
echo ""
echo "🔑 Checking API Keys:"
echo "===================="

if [[ -n "$ANTHROPIC_API_KEY" ]]; then
    echo "✅ ANTHROPIC_API_KEY is set"
else
    echo "⚠️  ANTHROPIC_API_KEY not set"
    echo "   Set it with: export ANTHROPIC_API_KEY='your-key'"
fi

if [[ -n "$OPENAI_API_KEY" ]]; then
    echo "✅ OPENAI_API_KEY is set"
else
    echo "⚠️  OPENAI_API_KEY not set"
    echo "   Set it with: export OPENAI_API_KEY='your-key'"
fi

if [[ -n "$VOYAGE_API_KEY" ]]; then
    echo "✅ VOYAGE_API_KEY is set (for re-ranking)"
else
    echo "⚠️  VOYAGE_API_KEY not set (optional, for re-ranking)"
    echo "   Set it with: export VOYAGE_API_KEY='your-key'"
fi

echo ""
echo "🚀 Ready to run experiments!"
echo "============================="
echo "Try: python experiment_runner.py --help"
echo "Or:  python example_claude_experiment.py" 