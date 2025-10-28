#!/usr/bin/env python3
"""
Script to summarize MBPP evaluation results.
Analyzes all methods and augmentation types in the mbpp_results directory
and calculates pass rates and syntax validity rates, outputting to CSV files.
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

def load_jsonl_results(file_path: str) -> List[Dict]:
    """Load results from a JSONL file."""
    results = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []
    return results

def load_json_results(file_path: str) -> List[Dict]:
    """Load results from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

def calculate_pass_rate_from_individual_results(results: List[Dict]) -> Tuple[float, int, int]:
    """Calculate pass rate from individual results with assertion_results."""
    if not results:
        return 0.0, 0, 0
    
    total_tasks = len(results)
    passed_tasks = 0
    
    for result in results:
        assertion_results = result.get('assertion_results', [])
        if assertion_results:
            # Task passes if all assertions pass
            all_passed = all(ar.get('status') == 'PASSED' for ar in assertion_results)
            if all_passed:
                passed_tasks += 1
    
    pass_rate = (passed_tasks / total_tasks) * 100 if total_tasks > 0 else 0.0
    return pass_rate, passed_tasks, total_tasks

def calculate_pass_rate(results: List[Dict]) -> Tuple[float, int, int]:
    """Calculate pass rate from results."""
    if not results:
        return 0.0, 0, 0
    
    total_tasks = len(results)
    passed_tasks = sum(1 for result in results if result.get('passed', False))
    pass_rate = (passed_tasks / total_tasks) * 100 if total_tasks > 0 else 0.0
    
    return pass_rate, passed_tasks, total_tasks

def calculate_syntax_rate(results: List[Dict]) -> Tuple[float, int, int]:
    """Calculate syntax validity rate from results."""
    if not results:
        return 0.0, 0, 0
    
    total_tasks = len(results)
    syntax_valid_tasks = sum(1 for result in results if result.get('is_syntactically_valid', False))
    syntax_rate = (syntax_valid_tasks / total_tasks) * 100 if total_tasks > 0 else 0.0
    
    return syntax_rate, syntax_valid_tasks, total_tasks

def calculate_ideal_ranker_pass_rate(method_data: Dict) -> Tuple[float, int, int]:
    """Calculate ideal ranker pass rate - upper bound if we could always pick the best result."""
    # Methods to consider for ideal ranker (excluding reranked as it's already a ranker)
    ranker_methods = ['no_rag', 'bm25', 'voyage_func', 'voyage_block', 'voyage_emb']
    
    # Get all individual results for each method
    all_results = {}
    total_tasks = 0
    
    for method in ranker_methods:
        if method in method_data:
            # We need to load the individual results again to get task-level data
            # This is a bit inefficient but necessary for the ideal ranker calculation
            all_results[method] = method_data[method].get('individual_results', [])
            if not total_tasks:
                total_tasks = method_data[method]['total']
    
    if not all_results or total_tasks == 0:
        return 0.0, 0, 0
    
    # For ideal ranker, a task passes if ANY of the methods got it right
    passed_tasks = 0
    
    # We'll need to track this by task_id, but since we don't have easy access to the
    # individual results here, we'll approximate using the stored results
    # This is a limitation - ideally we'd need to reload and compare task by task
    
    # For now, let's use a conservative approximation based on the maximum pass rate
    # In reality, this should be calculated by loading all individual results and
    # checking task by task, but that would require significant restructuring
    
    # Get the method with highest pass rate as a lower bound
    max_passed = 0
    for method in ranker_methods:
        if method in method_data:
            max_passed = max(max_passed, method_data[method]['passed'])
    
    # This is an approximation - the real ideal ranker would likely be higher
    passed_tasks = max_passed
    pass_rate = (passed_tasks / total_tasks) * 100 if total_tasks > 0 else 0.0
    
    return pass_rate, passed_tasks, total_tasks

def calculate_ideal_ranker_from_files(method_dir: Path, ranker_methods: List[str]) -> Tuple[float, int, int]:
    """Calculate ideal ranker pass rate by loading and comparing individual results files."""
    all_method_results = {}
    
    # Load individual results for each ranker method
    for method in ranker_methods:
        individual_file = method_dir / f"{method}_individual_results.json"
        if individual_file.exists():
            results = load_json_results(str(individual_file))
            # Create a dict mapping task_id to whether it passed
            task_results = {}
            for result in results:
                task_id = result.get('task_id')
                assertion_results = result.get('assertion_results', [])
                if assertion_results:
                    all_passed = all(ar.get('status') == 'PASSED' for ar in assertion_results)
                    task_results[task_id] = all_passed
                else:
                    task_results[task_id] = False
            all_method_results[method] = task_results
    
    if not all_method_results:
        return 0.0, 0, 0
    
    # Get all unique task IDs
    all_task_ids = set()
    for method_results in all_method_results.values():
        all_task_ids.update(method_results.keys())
    
    if not all_task_ids:
        return 0.0, 0, 0
    
    # For ideal ranker, count tasks where ANY method succeeded
    passed_tasks = 0
    total_tasks = len(all_task_ids)
    
    for task_id in all_task_ids:
        # Check if any method passed this task
        any_passed = False
        for method_results in all_method_results.values():
            if task_id in method_results and method_results[task_id]:
                any_passed = True
                break
        
        if any_passed:
            passed_tasks += 1
    
    pass_rate = (passed_tasks / total_tasks) * 100 if total_tasks > 0 else 0.0
    return pass_rate, passed_tasks, total_tasks

def extract_method_name(folder_name: str) -> str:
    """Extract clean method name from folder name."""
    # Remove timestamp suffix
    parts = folder_name.split('_')
    if len(parts) >= 2 and parts[-1].isdigit():
        return '_'.join(parts[:-2])  # Remove last two parts (mbpp and timestamp)
    return folder_name

def extract_augmentation_type(filename: str) -> str:
    """Extract augmentation type from filename."""
    # Remove _individual_results.json or _results.jsonl suffix
    if filename.endswith('_individual_results.json'):
        return filename.replace('_individual_results.json', '')
    elif filename.endswith('_results.jsonl_results.jsonl'):
        return filename.replace('_results.jsonl_results.jsonl', '')
    elif filename.endswith('_results.jsonl'):
        return filename.replace('_results.jsonl', '')
    return filename

def find_original_experiment_file(method_dir_name: str, augmentation_type: str) -> str:
    """Find the original experiment JSONL file for syntax analysis."""
    experiment_results_dir = Path('experiment_results')
    
    # Try to find matching experiment directory
    for exp_dir in experiment_results_dir.iterdir():
        if exp_dir.is_dir() and exp_dir.name == method_dir_name:
            # Look for the original JSONL file
            original_file = exp_dir / f"{augmentation_type}_results.jsonl"
            if original_file.exists():
                return str(original_file)
    
    return ""

def main():
    """Main function to analyze MBPP results."""
    results_dir = Path('mbpp_results')
    
    if not results_dir.exists():
        print(f"Error: {results_dir} directory not found!")
        return
    
    # Dictionary to store results: {method: {augmentation_type: {pass_rate, syntax_rate, ...}}}
    summary_data = {}
    
    print("Analyzing MBPP results...")
    print("=" * 50)
    
    # Methods to consider for ideal ranker calculation
    ranker_methods = ['no_rag', 'bm25', 'voyage_func', 'voyage_block', 'voyage_emb']
    
    # Iterate through each method directory
    for method_dir in results_dir.iterdir():
        if not method_dir.is_dir():
            continue
            
        method_name = extract_method_name(method_dir.name)
        print(f"\nProcessing method: {method_name}")
        
        summary_data[method_name] = {}
        
        # Find all *_individual_results.json files for pass rate calculation
        individual_result_files = []
        for file_path in method_dir.iterdir():
            if file_path.is_file() and file_path.name.endswith('_individual_results.json'):
                individual_result_files.append(file_path)
        
        # Process each augmentation type
        for file_path in individual_result_files:
            augmentation_type = extract_augmentation_type(file_path.name)
            print(f"  - Processing {augmentation_type}...")
            
            # Load and analyze pass rate results from individual results
            individual_results = load_json_results(str(file_path))
            pass_rate, passed, total = calculate_pass_rate_from_individual_results(individual_results)
            
            # Find corresponding JSONL file for syntax analysis
            syntax_file = method_dir / f"{augmentation_type}_results.jsonl"
            syntax_rate, syntax_valid, syntax_total = 0.0, 0, 0
            
            if syntax_file.exists():
                print(f"    Found syntax file: {syntax_file}")
                syntax_results = load_jsonl_results(str(syntax_file))
                syntax_rate, syntax_valid, syntax_total = calculate_syntax_rate(syntax_results)
            else:
                print(f"    Warning: Could not find syntax file for {augmentation_type}")
                # Use the same totals as pass results
                syntax_total = total
            
            summary_data[method_name][augmentation_type] = {
                'pass_rate': pass_rate,
                'passed': passed,
                'total': total,
                'syntax_rate': syntax_rate,
                'syntax_valid': syntax_valid,
                'syntax_total': syntax_total
            }
            
            print(f"    Pass rate: {pass_rate:.2f}% ({passed}/{total})")
            print(f"    Syntax rate: {syntax_rate:.2f}% ({syntax_valid}/{syntax_total})")
        
        # Calculate ideal ranker performance for this method
        if any(method in summary_data[method_name] for method in ranker_methods):
            print(f"  - Calculating ideal ranker...")
            ideal_pass_rate, ideal_passed, ideal_total = calculate_ideal_ranker_from_files(method_dir, ranker_methods)
            
            # Add ideal ranker as a special augmentation type
            summary_data[method_name]['ideal_ranker'] = {
                'pass_rate': ideal_pass_rate,
                'passed': ideal_passed,
                'total': ideal_total,
                'syntax_rate': 100.0,  # Assume ideal ranker has perfect syntax (picks from valid options)
                'syntax_valid': ideal_total,
                'syntax_total': ideal_total
            }
            
            print(f"    Ideal ranker pass rate: {ideal_pass_rate:.2f}% ({ideal_passed}/{ideal_total})")
    
    # Create DataFrame for CSV output
    rows = []
    all_augmentation_types = set()
    
    # Collect all augmentation types
    for method_data in summary_data.values():
        all_augmentation_types.update(method_data.keys())
    
    # Sort augmentation types, putting ideal_ranker at the end
    all_augmentation_types = sorted([t for t in all_augmentation_types if t != 'ideal_ranker'])
    if any('ideal_ranker' in method_data for method_data in summary_data.values()):
        all_augmentation_types.append('ideal_ranker')
    
    # Create rows for CSV
    for method_name, method_data in summary_data.items():
        row = {'Method': method_name}
        for aug_type in all_augmentation_types:
            if aug_type in method_data:
                pass_rate = method_data[aug_type]['pass_rate']
                passed = method_data[aug_type]['passed']
                total = method_data[aug_type]['total']
                syntax_rate = method_data[aug_type]['syntax_rate']
                syntax_valid = method_data[aug_type]['syntax_valid']
                syntax_total = method_data[aug_type]['syntax_total']
                
                row[f'{aug_type}_pass_rate'] = f"{pass_rate:.2f}%"
                row[f'{aug_type}_passed_total'] = f"{passed}/{total}"
                row[f'{aug_type}_syntax_rate'] = f"{syntax_rate:.2f}%"
                row[f'{aug_type}_syntax_total'] = f"{syntax_valid}/{syntax_total}"
            else:
                row[f'{aug_type}_pass_rate'] = "N/A"
                row[f'{aug_type}_passed_total'] = "N/A"
                row[f'{aug_type}_syntax_rate'] = "N/A"
                row[f'{aug_type}_syntax_total'] = "N/A"
        rows.append(row)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows)
    
    # Reorder columns: Method first, then alternating pass and syntax metrics for each augmentation type
    columns = ['Method']
    for aug_type in all_augmentation_types:
        columns.extend([
            f'{aug_type}_pass_rate', f'{aug_type}_passed_total',
            f'{aug_type}_syntax_rate', f'{aug_type}_syntax_total'
        ])
    
    df = df[columns]
    
    # Save to CSV
    output_file = 'mbpp_results_summary.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\n" + "=" * 50)
    print(f"Summary saved to: {output_file}")
    print(f"Total methods analyzed: {len(summary_data)}")
    print(f"Augmentation types found: {', '.join(all_augmentation_types)}")
    
    # Also create a simplified version with just pass rates and syntax rates
    simple_rows = []
    for method_name, method_data in summary_data.items():
        row = {'Method': method_name}
        for aug_type in all_augmentation_types:
            if aug_type in method_data:
                pass_rate = method_data[aug_type]['pass_rate']
                syntax_rate = method_data[aug_type]['syntax_rate']
                row[f'{aug_type}_pass'] = f"{pass_rate:.2f}%"
                row[f'{aug_type}_syntax'] = f"{syntax_rate:.2f}%"
            else:
                row[f'{aug_type}_pass'] = "N/A"
                row[f'{aug_type}_syntax'] = "N/A"
        simple_rows.append(row)
    
    simple_df = pd.DataFrame(simple_rows)
    simple_output_file = 'mbpp_results_summary_simple.csv'
    simple_df.to_csv(simple_output_file, index=False)
    print(f"Simplified summary saved to: {simple_output_file}")
    
    # Print summary table
    print(f"\nSummary Table (Pass % | Syntax %):")
    print(simple_df.to_string(index=False))
    
    # Create pass-only and syntax-only tables for easier reading
    pass_only_rows = []
    syntax_only_rows = []
    
    for method_name, method_data in summary_data.items():
        pass_row = {'Method': method_name}
        syntax_row = {'Method': method_name}
        
        for aug_type in all_augmentation_types:
            if aug_type in method_data:
                pass_rate = method_data[aug_type]['pass_rate']
                syntax_rate = method_data[aug_type]['syntax_rate']
                pass_row[aug_type] = f"{pass_rate:.2f}%"
                syntax_row[aug_type] = f"{syntax_rate:.2f}%"
            else:
                pass_row[aug_type] = "N/A"
                syntax_row[aug_type] = "N/A"
        
        pass_only_rows.append(pass_row)
        syntax_only_rows.append(syntax_row)
    
    # Save separate tables
    pass_df = pd.DataFrame(pass_only_rows)
    syntax_df = pd.DataFrame(syntax_only_rows)
    
    pass_df.to_csv('mbpp_pass_rates_only.csv', index=False)
    syntax_df.to_csv('mbpp_syntax_rates_only.csv', index=False)
    
    print(f"\nPass Rates Only:")
    print(pass_df.to_string(index=False))
    print(f"\nSyntax Rates Only:")
    print(syntax_df.to_string(index=False))
    
    print(f"\nAdditional files created:")
    print(f"- mbpp_pass_rates_only.csv")
    print(f"- mbpp_syntax_rates_only.csv")

if __name__ == "__main__":
    main() 