#!/usr/bin/env python3
"""
Convenience script to run PKG experiments from the root directory.
This script handles the import paths and provides a simple interface.
"""

import sys
import os
from pathlib import Path

# Add repo root to Python path so `import src` resolves regardless of cwd
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

if __name__ == "__main__":
    from src.experiments.experiment_runner import main
    main() 