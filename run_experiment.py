#!/usr/bin/env python3
"""
Convenience script to run PKG experiments from the root directory.
This script handles the import paths and provides a simple interface.
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

if __name__ == "__main__":
    from src.experiments.experiment_runner import main
    main() 