#!/usr/bin/env python3
"""
Script to run HumanEval evaluation for all experiment result folders.
First converts all experiment results to HumanEval format, then evaluates them
and generates a comprehensive JSON report.
"""

import os
import subprocess
import json
import tempfile
import shutil
import re
from pathlib import Path
from datetime import datetime

def convert_experiment_results():
    """
    Convert all experiment results to HumanEval format using the conversion script.
    Only converts folders where all JSONL files have more than 100 rows.
    """
    print("Converting experiment results to HumanEval format...")
    
    experiment_results_dir = "experiment_results"
    humaneval_results_dir = "humaneval_results"
    
    if not os.path.exists(experiment_results_dir):
        print(f"Directory {experiment_results_dir} does not exist!")
        return False
    
    # Get all experiment result folders
    experiment_folders = []
    for item in os.listdir(experiment_results_dir):
        item_path = os.path.join(experiment_results_dir, item)
        if os.path.isdir(item_path) and "humaneval" in item.lower():
            experiment_folders.append(item)
    
    print(f"Found {len(experiment_folders)} experiment folders to check:")
    for folder in experiment_folders:
        print(f"  - {folder}")
    
    # Check and convert each experiment folder
    converted_count = 0
    for folder in experiment_folders:
        if not "humaneval" in folder:
            continue
        
        print(f"\nChecking {folder}...")
        
        root = os.path.join(experiment_results_dir, folder)
        dest = os.path.join(humaneval_results_dir, folder)
        
        # Get all JSONL files in the experiment folder
        jsonl_files = [f for f in os.listdir(root) if f.endswith(".jsonl")]
        
        if not jsonl_files:
            print(f"  No JSONL files found in {folder} - skipping")
            continue
        
        print(f"  Found {len(jsonl_files)} JSONL files, checking row counts...")
        
        # Check if all JSONL files have more than 100 rows
        all_files_valid = True
        file_row_counts = {}
        
        for file in jsonl_files:
            try:
                file_path = os.path.join(root, file)
                row_count = 0
                with open(file_path, "r") as f:
                    for line in f:
                        if line.strip():  # Count non-empty lines
                            row_count += 1
                
                file_row_counts[file] = row_count
                print(f"    {file}: {row_count} rows")
                
                if row_count <= 100:
                    all_files_valid = False
                    print(f"    ✗ {file} has only {row_count} rows (≤100) - folder will be skipped")
                    
                if "dummy_function" in file:
                    all_files_valid = False
                    print(f"    ✗ {file} has only {row_count} rows (≤100) - folder will be skipped")
                
            except Exception as e:
                print(f"    ✗ Error reading {file}: {str(e)} - folder will be skipped")
                all_files_valid = False
                break
        
        if not all_files_valid:
            print(f"  Skipping {folder} - not all JSONL files have >100 rows")
            continue
        
        print(f"  ✓ All JSONL files have >100 rows - converting {folder}...")
        
        # Remove existing destination folder if it exists
        if os.path.exists(dest):
            print(f"    Removing existing folder: {dest}")
            shutil.rmtree(dest)
        
        # Create destination directory
        os.makedirs(dest, exist_ok=True)
        
        print(f"    Converting {len(jsonl_files)} JSONL files...")
        
        # Convert all JSONL files in the experiment folder
        conversion_success = True
        for file in jsonl_files:
            try:
                new_jsonl = []
                with open(os.path.join(root, file), "r") as f:
                    for line in f:
                        if line.strip():  # Skip empty lines
                            data = json.loads(line)
                            new_jsonl.append({
                                "task_id": data["task_id"],
                                "completion": data["generated_code"]
                            })
                
                with open(os.path.join(dest, file), "w") as f:
                    for line in new_jsonl:
                        f.write(json.dumps(line) + "\n")
                
                print(f"      ✓ Converted {file} ({len(new_jsonl)} entries)")
                
            except Exception as e:
                print(f"      ✗ Error converting {file}: {str(e)}")
                conversion_success = False
        
        if conversion_success:
            converted_count += 1
            print(f"    ✓ Successfully converted {folder}")
        else:
            print(f"    ✗ Failed to convert some files in {folder}")
    
    print(f"\nConversion completed!")
    print(f"Successfully converted {converted_count} out of {len(experiment_folders)} folders")
    return converted_count > 0

def run_humaneval_evaluation(jsonl_file_path):
    """
    Run evaluate_functional_correctness on a JSONL file and extract pass@1 percentage.
    
    Args:
        jsonl_file_path (str): Path to the JSONL file to evaluate
        
    Returns:
        float: Pass@1 percentage (0-100)
    """
    import re  # Move re import to the top of the function
    
    try:
        # Create a temporary directory to capture the results
        with tempfile.TemporaryDirectory() as temp_dir:
            # Change to the directory containing the JSONL file
            original_dir = os.getcwd()
            jsonl_dir = os.path.dirname(jsonl_file_path)
            jsonl_filename = os.path.basename(jsonl_file_path)
            
            os.chdir(jsonl_dir)
            
            # Run the evaluation command
            result = subprocess.run(
                ['evaluate_functional_correctness', jsonl_filename],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            # Change back to original directory
            os.chdir(original_dir)
            
            if result.returncode != 0:
                print(f"Error evaluating {jsonl_filename}: {result.stderr}")
                return None
            
            # Parse the output to extract pass@1 percentage
            output_lines = result.stdout.strip().split('\n')
            
            # First, try to find dictionary format with numpy types
            for line in output_lines:
                line = line.strip()
                if 'pass@1' in line.lower() and ('{' in line or 'np.float' in line):
                    try:
                        # Handle numpy float format: {'pass@1': np.float64(0.0)}
                        if 'np.float' in line:
                            # Extract the numeric value from np.float64(value)
                            match = re.search(r'np\.float\d*\(([0-9.]+)\)', line)
                            if match:
                                decimal_value = float(match.group(1))
                                return decimal_value * 100  # Convert to percentage
                        
                        # Handle regular dictionary format
                        if line.startswith('{') and line.endswith('}'):
                            # Replace single quotes with double quotes and handle numpy types
                            json_line = line.replace("'", '"')
                            # Remove numpy type references
                            json_line = re.sub(r'np\.float\d*\(([0-9.]+)\)', r'\1', json_line)
                            result_dict = json.loads(json_line)
                            if 'pass@1' in result_dict:
                                decimal_value = float(result_dict['pass@1'])
                                return decimal_value * 100  # Convert to percentage
                    except (json.JSONDecodeError, ValueError) as e:
                        print(f"Error parsing line '{line}': {e}")
                        continue
            
            # Then try other formats
            for line in output_lines:
                if 'pass@1' in line.lower():
                    # Format 1: "pass@1: 0.85 (85.0%)"
                    percentage_match = re.search(r'\((\d+\.?\d*)\s*%\)', line)
                    if percentage_match:
                        return float(percentage_match.group(1))
                    
                    # Format 2: "pass@1: 0.85"
                    decimal_match = re.search(r'pass@1\s*:\s*(\d+\.?\d*)', line, re.IGNORECASE)
                    if decimal_match:
                        decimal_value = float(decimal_match.group(1))
                        # If it's already a percentage (>1), return as is, otherwise convert
                        return decimal_value if decimal_value > 1 else decimal_value * 100
                    
                    # Format 3: Just look for any number followed by %
                    percent_match = re.search(r'(\d+\.?\d*)\s*%', line)
                    if percent_match:
                        return float(percent_match.group(1))
            
            # If we can't parse the output, try to find results file
            # results_file = jsonl_filename.replace('.jsonl', '_results.jsonl')
            results_file = jsonl_filename
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    results = [json.loads(line) for line in f]
                    passed = sum(1 for result in results if result.get('passed', False))
                    total = len(results)
                    return (passed / total) * 100 if total > 0 else 0
            
            print(f"Could not parse pass@1 from output for {jsonl_filename}")
            print(f"Output: {result.stdout}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"Timeout evaluating {jsonl_filename}")
        return None
    except Exception as e:
        print(f"Error evaluating {jsonl_filename}: {str(e)}")
        return None

def main():
    """
    Main function to convert experiment results and evaluate all HumanEval folders.
    """
    print("=== HumanEval Evaluation Pipeline ===\n")
    
    # Step 1: Convert all experiment results (only those with >100 rows in all JSONL files)
    conversion_successful = convert_experiment_results()
    if not conversion_successful:
        print("No experiment results were converted (no folders met the >100 rows requirement). Checking existing humaneval_results...")
    
    # Step 2: Find all folders in humaneval_results
    humaneval_results_dir = "humaneval_results"
    
    if not os.path.exists(humaneval_results_dir):
        print(f"Directory {humaneval_results_dir} does not exist!")
        return
    
    # Get all folders in humaneval_results
    result_folders = []
    for item in os.listdir(humaneval_results_dir):
        item_path = os.path.join(humaneval_results_dir, item)
        if os.path.isdir(item_path):
            result_folders.append(item)
    
    if not result_folders:
        print(f"No folders found in {humaneval_results_dir}")
        if not conversion_successful:
            print("No experiment results were converted and no existing results found.")
            print("Make sure your experiment result folders contain JSONL files with >100 rows each.")
        return
    
    print(f"\n=== Evaluating {len(result_folders)} result folders ===")
    for folder in result_folders:
        print(f"  - {folder}")
    
    # Step 3: Evaluate each folder
    all_results = []
    
    for folder in result_folders:
        print(f"\n--- Processing folder: {folder} ---")
        folder_path = os.path.join(humaneval_results_dir, folder)
        
        # Find all JSONL files in the folder
        jsonl_files = []
        for file in os.listdir(folder_path):
            if file.endswith('.jsonl'):
                jsonl_files.append(os.path.join(folder_path, file))
        
        if not jsonl_files:
            print(f"No JSONL files found in {folder}")
            continue
        
        print(f"Found {len(jsonl_files)} JSONL files to evaluate:")
        for file in jsonl_files:
            print(f"  - {os.path.basename(file)}")
        
        # Evaluate each JSONL file in the folder
        folder_results = []
        for jsonl_file in jsonl_files:
            print(f"\nEvaluating {os.path.basename(jsonl_file)}...")
            
            pass_at_1 = run_humaneval_evaluation(jsonl_file)
            
            if pass_at_1 is not None:
                method_name = os.path.basename(jsonl_file).replace('_results.jsonl', '')
                result = {
                    'experiment_folder': folder,
                    'method': method_name,
                    'file': os.path.basename(jsonl_file),
                    'pass_at_1_percentage': pass_at_1
                }
                folder_results.append(result)
                all_results.append(result)
                print(f"  ✓ Pass@1: {pass_at_1:.2f}%")
            else:
                print(f"  ✗ Failed to evaluate {os.path.basename(jsonl_file)}")
        
        # Print folder summary
        if folder_results:
            print(f"\n{folder} Summary:")
            print("Method\t\t\tPass@1 %")
            print("-" * 40)
            for result in sorted(folder_results, key=lambda x: x['pass_at_1_percentage'], reverse=True):
                print(f"{result['method']:<20}\t{result['pass_at_1_percentage']:.2f}%")
    
    # Step 4: Generate comprehensive JSON report
    if all_results:
        # Create overall report
        report = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'total_experiments': len(result_folders),
            'total_evaluations': len(all_results),
            'results': all_results,
            'summary_by_experiment': {},
            'summary_by_method': {},
            'best_performers': {
                'by_experiment': {},
                'by_method': {}
            }
        }
        
        # Group results by experiment folder
        for result in all_results:
            exp_folder = result['experiment_folder']
            if exp_folder not in report['summary_by_experiment']:
                report['summary_by_experiment'][exp_folder] = []
            report['summary_by_experiment'][exp_folder].append({
                'method': result['method'],
                'pass_at_1_percentage': result['pass_at_1_percentage']
            })
        
        # Group results by method
        for result in all_results:
            method = result['method']
            if method not in report['summary_by_method']:
                report['summary_by_method'][method] = []
            report['summary_by_method'][method].append({
                'experiment_folder': result['experiment_folder'],
                'pass_at_1_percentage': result['pass_at_1_percentage']
            })
        
        # Find best performers
        for exp_folder, methods in report['summary_by_experiment'].items():
            best_method = max(methods, key=lambda x: x['pass_at_1_percentage'])
            report['best_performers']['by_experiment'][exp_folder] = best_method
        
        for method, experiments in report['summary_by_method'].items():
            avg_score = sum(exp['pass_at_1_percentage'] for exp in experiments) / len(experiments)
            report['best_performers']['by_method'][method] = {
                'average_pass_at_1': avg_score,
                'experiment_count': len(experiments),
                'experiments': experiments
            }
        
        # Save JSON report
        json_filename = "humaneval_evaluation_report.json"
        with open(json_filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n=== OVERALL RESULTS ===")
        print(f"Comprehensive report saved to {json_filename}")
        print(f"Total experiments evaluated: {len(result_folders)}")
        print(f"Total method evaluations: {len(all_results)}")
        
        print("\n=== BEST PERFORMERS BY EXPERIMENT ===")
        for exp_folder, best_method in report['best_performers']['by_experiment'].items():
            print(f"{exp_folder}:")
            print(f"  Best: {best_method['method']} ({best_method['pass_at_1_percentage']:.2f}%)")
        
        print("\n=== AVERAGE PERFORMANCE BY METHOD ===")
        method_averages = [(method, data['average_pass_at_1']) 
                          for method, data in report['best_performers']['by_method'].items()]
        method_averages.sort(key=lambda x: x[1], reverse=True)
        
        for method, avg_score in method_averages:
            exp_count = report['best_performers']['by_method'][method]['experiment_count']
            print(f"{method:<20}\t{avg_score:.2f}% (across {exp_count} experiments)")
        
    else:
        print("No successful evaluations completed.")

if __name__ == "__main__":
    main() 