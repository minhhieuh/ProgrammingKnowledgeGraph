#!/bin/bash

# Convenience setup script for PKG experiments
# This script runs the actual setup script from the scripts directory

echo "🚀 PKG Framework Setup"
echo "====================="

# Check if we're in the right directory
if [ ! -f "environment.yml" ]; then
    echo "❌ Please run this script from the project root directory"
    exit 1
fi

# Run the actual setup script
bash src/scripts/setup_conda_env.sh

echo ""
echo "🎉 Setup complete! You can now:"
echo "   1. Activate the environment: source src/scripts/activate_env.sh"
echo "   2. Run experiments: python run_experiment.py --help"
echo "   3. Check the documentation in docs/" 