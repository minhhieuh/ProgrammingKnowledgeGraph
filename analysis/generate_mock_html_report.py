#!/usr/bin/env python3
"""
Convenience script to generate HTML reports from mock prompt test results.
This script handles the import paths and provides a simple interface.
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

if __name__ == "__main__":
    from src.experiments.generate_mock_html_report import main
    main() 