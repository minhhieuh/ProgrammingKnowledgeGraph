from MBPP.human_eval.evaluation import evaluate_functional_correctness
import os
import glob
import json
import logging
from datetime import datetime
import traceback
import sys
import io
import contextlib

# Setup logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = "evaluation_logs"
os.makedirs(log_dir, exist_ok=True)

# Configure main logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f'mbpp_evaluation_{timestamp}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create detailed evaluation logger
detailed_logger = logging.getLogger('detailed_evaluation')
detailed_handler = logging.FileHandler(os.path.join(log_dir, f'mbpp_detailed_evaluation_{timestamp}.log'))
detailed_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
detailed_logger.addHandler(detailed_handler)
detailed_logger.setLevel(logging.INFO)

def evaluate_individual_assertions(generated_code, test_assertions, task_id, detailed_logger):
    """
    Evaluate each test assertion individually and log the results
    """
    detailed_logger.info(f"INDIVIDUAL ASSERTION EVALUATION FOR TASK {task_id}:")
    
    results = []
    for i, assertion in enumerate(test_assertions, 1):
        detailed_logger.info(f"  Testing assertion {i}: {assertion}")
        
        # Create the complete test code
        test_code = generated_code + "\n" + assertion
        
        try:
            # Capture stdout/stderr
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            
            # Execute the test
            exec_globals = {}
            exec(test_code, exec_globals)
            
            # If we get here, the assertion passed
            stdout_output = stdout_capture.getvalue()
            stderr_output = stderr_capture.getvalue()
            
            detailed_logger.info(f"    ✓ PASSED")
            if stdout_output.strip():
                detailed_logger.info(f"    STDOUT: {stdout_output.strip()}")
            if stderr_output.strip():
                detailed_logger.info(f"    STDERR: {stderr_output.strip()}")
                
            results.append({
                'assertion': assertion,
                'status': 'PASSED',
                'error': None,
                'stdout': stdout_output,
                'stderr': stderr_output
            })
            
        except AssertionError as e:
            detailed_logger.info(f"    ✗ FAILED: AssertionError - {str(e)}")
            results.append({
                'assertion': assertion,
                'status': 'FAILED',
                'error': f"AssertionError: {str(e)}",
                'stdout': stdout_capture.getvalue(),
                'stderr': stderr_capture.getvalue()
            })
            
        except Exception as e:
            detailed_logger.info(f"    ✗ ERROR: {type(e).__name__} - {str(e)}")
            results.append({
                'assertion': assertion,
                'status': 'ERROR',
                'error': f"{type(e).__name__}: {str(e)}",
                'stdout': stdout_capture.getvalue(),
                'stderr': stderr_capture.getvalue()
            })
            
        finally:
            # Restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    
    # Summary for this task
    passed = sum(1 for r in results if r['status'] == 'PASSED')
    total = len(results)
    detailed_logger.info(f"  SUMMARY: {passed}/{total} assertions passed")
    
    return results

exp_folder = "experiment_results/gpt-4o-mini_mbpp_1754249928"
run_name = os.path.basename(exp_folder)

model_name = run_name.split("_")[0]

dest = "mbpp_results"
out_folder = os.path.join(dest, run_name)

if not os.path.exists(out_folder):
    os.makedirs(out_folder, exist_ok=True)

logger.info(f"Starting MBPP evaluation for {run_name}")
logger.info(f"Model: {model_name}")
logger.info(f"Output folder: {out_folder}")

# Load MBPP problems for detailed logging
mbpp_problems = {}
with open("mbpp_test.jsonl", "r") as f:
    for line in f:
        problem = json.loads(line)
        mbpp_problems[problem["task_id"]] = problem
    
# find every jsonl file in the exp_folder
jsonl_files = glob.glob(os.path.join(exp_folder, "*.jsonl"))

logger.info(f"Found {len(jsonl_files)} JSONL files to process")

# Store evaluation results for summary
evaluation_results = {}
detailed_results = []


for jsonl_file in jsonl_files:
    file_name = os.path.basename(jsonl_file)
    augmentation_type = file_name.replace('_results.jsonl', '')
    
    logger.info(f"Processing {jsonl_file}")
    print(f"Processing {jsonl_file}")
    
    out_file = os.path.join(out_folder, os.path.basename(jsonl_file))
    new_jsonl = []
    
    # Count original entries and log detailed information
    original_count = 0
    detailed_logger.info(f"\n{'='*80}")
    detailed_logger.info(f"PROCESSING FILE: {file_name} (Augmentation: {augmentation_type})")
    detailed_logger.info(f"{'='*80}")
    
    individual_results = []  # Store individual assertion results
    
    for line in open(jsonl_file):
        original_count += 1
        data = json.loads(line)
        task_id = int(data["task_id"].split("/")[-1])
        
        new_line = {
            "task_id": task_id,
            "generation": data["generated_code"],
            "is_syntactically_valid": data["is_syntactically_valid"],
        }
        new_jsonl.append(new_line)
        
        # Log detailed information for each instance
        detailed_logger.info(f"\n{'-'*60}")
        detailed_logger.info(f"TASK ID: {task_id}")
        detailed_logger.info(f"PROMPT: {mbpp_problems.get(task_id, {}).get('prompt', 'Unknown')}")
        detailed_logger.info(f"SYNTACTICALLY VALID: {data['is_syntactically_valid']}")
        detailed_logger.info(f"GENERATED CODE:")
        detailed_logger.info(f"{data['generated_code']}")
        
        # Log the test assertions this code will be evaluated against
        if task_id in mbpp_problems:
            detailed_logger.info(f"TEST ASSERTIONS:")
            for i, test in enumerate(mbpp_problems[task_id]["test"], 1):
                detailed_logger.info(f"  {i}. {test}")
            
            # Log the complete test code that will be executed
            test_code = data["generated_code"] + "\n" + "\n".join(mbpp_problems[task_id]["test"])
            detailed_logger.info(f"COMPLETE TEST CODE TO BE EXECUTED:")
            detailed_logger.info(f"{test_code}")
            
            # Run individual assertion evaluation
            if data["is_syntactically_valid"]:
                assertion_results = evaluate_individual_assertions(
                    data["generated_code"], 
                    mbpp_problems[task_id]["test"], 
                    task_id, 
                    detailed_logger
                )
                individual_results.append({
                    'task_id': task_id,
                    'generated_code': data["generated_code"],
                    'assertion_results': assertion_results
                })
            else:
                detailed_logger.info(f"SKIPPING INDIVIDUAL EVALUATION - Code is not syntactically valid")
                individual_results.append({
                    'task_id': task_id,
                    'generated_code': data["generated_code"],
                    'assertion_results': []
                })
        else:
            detailed_logger.info(f"WARNING: No test data found for task_id {task_id}")
        
        detailed_logger.info(f"{'-'*60}")
        
    # sort by task_id
    new_jsonl.sort(key=lambda x: int(x["task_id"]))
    
    logger.info(f"Processed {original_count} entries from {file_name}")
        
    with open(out_file, "w") as f:
        for line in new_jsonl:
            f.write(json.dumps(line) + "\n")
            
    logger.info(f"Wrote processed file: {out_file}")
    print(f"Wrote {out_file}")
    
    # Save individual results to a separate file
    individual_results_file = os.path.join(out_folder, f"{augmentation_type}_individual_results.json")
    with open(individual_results_file, "w") as f:
        json.dump(individual_results, f, indent=2)
    logger.info(f"Wrote individual results: {individual_results_file}")
        
    # Run evaluation with detailed logging
    logger.info(f"Starting evaluation for {augmentation_type}")
    results = evaluate_functional_correctness(
        input_file=out_file,
        problem_file="mbpp_test.jsonl",
        tmp_dir="./temp",
        out_dir=out_folder,
        language='python',
        is_mbpp=True,  # Critical!
        k=[1],
        timeout=20.0,
        n_workers=28
    )
    
    # Store results
    evaluation_results[augmentation_type] = results
    detailed_results.append({
        'augmentation_type': augmentation_type,
        'file_name': file_name,
        'original_count': original_count,
        'processed_count': len(new_jsonl),
        'results': results,
        'individual_results': individual_results
    })
    
    logger.info(f"Evaluation completed for {augmentation_type}")
    logger.info(f"Results: {results}")
    print(f"Results for {augmentation_type}: {results}")
    
    detailed_logger.info(f"\n{'='*80}")
    detailed_logger.info(f"EVALUATION RESULTS FOR {augmentation_type}: {results}")
    detailed_logger.info(f"{'='*80}\n")

# Log final summary
detailed_logger.info(f"\n{'='*80}")
detailed_logger.info(f"FINAL SUMMARY")
detailed_logger.info(f"{'='*80}")
for aug_type, results in evaluation_results.items():
    detailed_logger.info(f"{aug_type}: {results}")
detailed_logger.info(f"{'='*80}")