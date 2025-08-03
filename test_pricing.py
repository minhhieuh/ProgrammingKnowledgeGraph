#!/usr/bin/env python3
"""
Simple test script to verify model pricing loading from CSV
"""
import os
import pandas as pd
from typing import Dict

def load_model_pricing(csv_filename: str = "model_pricing.csv") -> Dict[str, Dict[str, float]]:
    """Load model pricing information from CSV file"""
    # Try multiple possible paths for the CSV file
    possible_paths = [
        csv_filename,  # Current directory
        f"../../{csv_filename}",  # From src/experiments/ to root
        f"../{csv_filename}",  # From src/ to root
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), csv_filename)  # Absolute path to root
    ]
    
    pricing = {}
    for csv_path in possible_paths:
        try:
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                for _, row in df.iterrows():
                    pricing[row['model_name']] = {
                        'input': float(row['input_price_per_mtok']),
                        'output': float(row['output_price_per_mtok'])
                    }
                print(f"✅ Loaded pricing for {len(pricing)} models from {csv_path}")
                return pricing
        except Exception as e:
            print(f"Failed to load from {csv_path}: {e}")
            continue
    
    # If no file found, use fallback
    print(f"⚠️  Pricing file {csv_filename} not found in any expected location")
    return {}

if __name__ == "__main__":
    pricing = load_model_pricing()
    print('\nLoaded pricing for models:')
    for model, costs in pricing.items():
        print(f'  {model}: input=${costs["input"]}/1M, output=${costs["output"]}/1M')
    print(f'\nTotal models: {len(pricing)}') 