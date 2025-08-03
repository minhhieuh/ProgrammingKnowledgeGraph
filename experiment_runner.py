#!/usr/bin/env python3
"""
Modern Experiment Runner for Context-Augmented Code Generation using Programming Knowledge Graphs

This module provides a flexible framework for running experiments with Anthropic Claude and OpenAI GPT models,
while preserving all experimental settings from the original paper.

Key features:
- Support for Anthropic Claude and OpenAI GPT models
- Configurable retrieval methods (Block-PKG, Func-PKG, BM25, No RAG)
- Re-ranking mechanism with AST analysis and runtime execution
- Pass@1 evaluation with greedy decoding (temperature=0)
- Support for both HumanEval and MBPP benchmarks
- Comprehensive logging and result tracking
"""

import os
import json
import time
import ast
import signal
import argparse
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

# LLM Framework imports
try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

# Evaluation imports
from human_eval.data import write_jsonl, read_problems
import voyageai

# Local imports
from reranker import rerank_one_solution, cosine_similarity, remove_comments_and_docstrings
from prompt_utils import *


@dataclass
class ExperimentConfig:
    """Configuration for experiment parameters"""
    model_name: str
    model_type: str  # 'openai' or 'anthropic'
    api_key: Optional[str] = None
    temperature: float = 0.0  # Greedy decoding as per paper
    max_tokens: int = 512  # As specified in paper
    timeout: int = 30  # Timeout for API calls
    
    # Retrieval settings
    augmentation_types: List[str] = None  # ['no_rag', 'voyage_func', 'voyage_block', 'bm25']
    
    # Evaluation settings
    benchmark: str = 'humaneval'  # 'humaneval' or 'mbpp'
    enable_reranking: bool = True
    
    # Output settings
    output_dir: str = 'experiment_results'
    experiment_name: str = None
    
    def __post_init__(self):
        if self.augmentation_types is None:
            self.augmentation_types = ['no_rag', 'voyage_func', 'voyage_block', 'bm25']
        if self.experiment_name is None:
            self.experiment_name = f"{self.model_name}_{self.benchmark}_{int(time.time())}"


class LLMProvider:
    """Base class for LLM providers"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.setup_client()
    
    def setup_client(self):
        """Setup the LLM client"""
        raise NotImplementedError
    
    def generate(self, prompt: str) -> str:
        """Generate code from prompt"""
        raise NotImplementedError


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider"""
    
    def setup_client(self):
        if not anthropic:
            raise ImportError("anthropic package not installed. Install with: pip install anthropic")
        self.client = anthropic.Anthropic(api_key=self.config.api_key)
    
    def generate(self, prompt: str) -> str:
        try:
            response = self.client.messages.create(
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logging.error(f"Anthropic API error: {e}")
            return ""


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider"""
    
    def setup_client(self):
        if not openai:
            raise ImportError("openai package not installed. Install with: pip install openai")
        self.client = openai.OpenAI(api_key=self.config.api_key)
    
    def generate(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"OpenAI API error: {e}")
            return ""


def get_provider(config: ExperimentConfig) -> LLMProvider:
    """Factory function to get the appropriate LLM provider"""
    providers = {
        'anthropic': AnthropicProvider,
        'openai': OpenAIProvider
    }
    
    if config.model_type not in providers:
        raise ValueError(f"Unsupported model type: {config.model_type}. Supported types: {list(providers.keys())}")
    
    return providers[config.model_type](config)


def create_prompt(problem: str, model_type: str, augmented_data: Optional[str] = None) -> str:
    """Create appropriate prompt based on model type and augmented data"""
    
    if augmented_data:
        prompt = f"""You are an expert Python programmer. Solve the following problem:

{problem}

The following code might be helpful as reference:
{augmented_data}

If the helper code is useful, integrate its logic directly into your solution. Otherwise, ignore it.

Requirements:
- Write executable Python code
- Include all necessary imports
- Ensure the solution is self-contained
- Write your solution between [PYTHON] and [/PYTHON] tags

Your solution:"""
    else:
        prompt = f"""You are an expert Python programmer. Solve the following problem:

{problem}

Requirements:
- Write executable Python code
- Include all necessary imports
- Ensure the solution is self-contained
- Write your solution between [PYTHON] and [/PYTHON] tags

Your solution:"""
    
    return prompt


def extract_python_code(text: str) -> str:
    """Extract Python code from generated text"""
    import re
    
    # First try to find code in [PYTHON] tags
    python_match = re.search(r'\[PYTHON\](.*?)\[/PYTHON\]', text, re.DOTALL)
    if python_match:
        return python_match.group(1).strip()
    
    # Try to find code in ```python blocks
    python_block = re.search(r'```python\n(.*?)```', text, re.DOTALL)
    if python_block:
        return python_block.group(1).strip()
    
    # Try to find code in ``` blocks
    code_block = re.search(r'```\n(.*?)```', text, re.DOTALL)
    if code_block:
        return code_block.group(1).strip()
    
    # If no specific markers found, try to extract function definitions
    lines = text.split('\n')
    code_lines = []
    in_function = False
    
    for line in lines:
        if line.strip().startswith('def '):
            in_function = True
            code_lines.append(line)
        elif in_function:
            if line.strip() == '' or line.startswith(' ') or line.startswith('\t'):
                code_lines.append(line)
            else:
                break
    
    if code_lines:
        return '\n'.join(code_lines)
    
    # Last resort: return the entire text
    return text.strip()


def load_augmented_data(augmentation_type: str, benchmark: str) -> Dict[str, Any]:
    """Load augmented data based on type and benchmark"""
    file_mapping = {
        'voyage_func': f'augmented_problems/{benchmark}_function_wise_relevant_context.jsonl',
        'voyage_block': f'augmented_problems/{benchmark}_blockwise_relevant_context.jsonl',
        'bm25': f'augmented_problems/bm25_relevant_context_{benchmark}.jsonl'
    }
    
    if augmentation_type not in file_mapping:
        return {}
    
    file_path = file_mapping[augmentation_type]
    if not os.path.exists(file_path):
        logging.warning(f"Augmented data file not found: {file_path}")
        return {}
    
    augmented_data = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                task_id = data['task_id']
                
                # Format the augmented content
                if augmentation_type == 'bm25':
                    # BM25 has different structure
                    content = data.get('content', '')
                else:
                    # Voyage embeddings have similarity scores and content
                    problems = data.get('problem', [])
                    if problems:
                        # Take top 3 most relevant pieces
                        top_content = []
                        for score, content in problems[:3]:
                            top_content.append(f"# Relevance: {score:.3f}\n{content}")
                        content = '\n\n'.join(top_content)
                    else:
                        content = ''
                
                augmented_data[task_id] = content
    except Exception as e:
        logging.error(f"Error loading augmented data from {file_path}: {e}")
        return {}
    
    return augmented_data


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException


def evaluate_solution(code: str, task_id: str, problems: Dict) -> bool:
    """Evaluate if a solution passes the test cases"""
    try:
        # Set timeout for execution
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(10)  # 10 second timeout
        
        # Get the test cases
        problem = problems[task_id]
        test_code = problem['test']
        
        # Combine solution with test
        full_code = f"{code}\n\n{test_code}"
        
        # Execute the code
        exec_globals = {}
        exec(full_code, exec_globals)
        
        signal.alarm(0)  # Cancel timeout
        return True
        
    except TimeoutException:
        logging.warning(f"Timeout during execution for {task_id}")
        return False
    except Exception as e:
        logging.debug(f"Execution failed for {task_id}: {e}")
        return False
    finally:
        signal.alarm(0)  # Ensure timeout is cancelled


def is_syntactically_valid(code: str) -> bool:
    """Check if code is syntactically valid"""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def run_single_experiment(config: ExperimentConfig, augmentation_type: str) -> List[Dict]:
    """Run experiment for a single augmentation type"""
    logging.info(f"Running experiment: {config.model_name} with {augmentation_type}")
    
    # Load problems
    if config.benchmark == 'humaneval':
        problems = read_problems()
    elif config.benchmark == 'mbpp':
        problems = load_mbpp_problems()
    else:
        raise ValueError(f"Unknown benchmark: {config.benchmark}")
    
    # Load augmented data if needed
    augmented_data = {}
    if augmentation_type != 'no_rag':
        augmented_data = load_augmented_data(augmentation_type, config.benchmark)
    
    # Initialize LLM provider
    provider = get_provider(config)
    
    results = []
    
    for task_id, problem_data in tqdm(problems.items(), desc=f"Processing {augmentation_type}"):
        problem_prompt = problem_data['prompt']
        
        # Get augmented context if available
        context = augmented_data.get(task_id, None) if augmentation_type != 'no_rag' else None
        
        # Create prompt
        prompt = create_prompt(problem_prompt, config.model_type, context)
        
        # Generate solution
        try:
            raw_output = provider.generate(prompt)
            generated_code = extract_python_code(raw_output)
            
            # Evaluate solution
            is_valid = is_syntactically_valid(generated_code)
            passed = False
            
            if is_valid:
                passed = evaluate_solution(generated_code, task_id, problems)
            
            result = {
                'task_id': task_id,
                'prompt': problem_prompt,
                'augmentation_type': augmentation_type,
                'raw_output': raw_output,
                'generated_code': generated_code,
                'is_syntactically_valid': is_valid,
                'passed': passed,
                'augmented_context': context if context else ""
            }
            
            results.append(result)
            logging.debug(f"Completed {task_id}: {'PASSED' if passed else 'FAILED'}")
            
        except Exception as e:
            logging.error(f"Error processing {task_id}: {e}")
            result = {
                'task_id': task_id,
                'prompt': problem_prompt,
                'augmentation_type': augmentation_type,
                'raw_output': "",
                'generated_code': "",
                'is_syntactically_valid': False,
                'passed': False,
                'error': str(e),
                'augmented_context': context if context else ""
            }
            results.append(result)
    
    return results


def load_mbpp_problems() -> Dict:
    """Load MBPP problems - implement based on your MBPP data format"""
    try:
        mbpp_df = pd.read_csv("mbpp.csv")
        problems = {}
        for idx, row in mbpp_df.iterrows():
            task_id = f"MBPP/{idx}"
            problems[task_id] = {
                'prompt': row['text'],
                'test': row.get('test_list', ''),
                'solution': row.get('code', '')
            }
        return problems
    except FileNotFoundError:
        logging.error("MBPP dataset not found. Please ensure mbpp.csv is available.")
        return {}


def perform_reranking(all_results: Dict[str, List[Dict]], config: ExperimentConfig) -> List[Dict]:
    """Perform re-ranking as described in the paper"""
    if not config.enable_reranking:
        return []
    
    logging.info("Performing solution re-ranking...")
    
    # Initialize embedder for semantic similarity
    try:
        voyageai.api_key = os.getenv('VOYAGE_API_KEY')  # Set your Voyage API key
        vo = voyageai.Client()
    except:
        logging.warning("VoyageAI not available for re-ranking")
        return []
    
    # Load problems for queries
    if config.benchmark == 'humaneval':
        problems = read_problems()
    else:
        problems = load_mbpp_problems()
    
    reranked_results = []
    
    for task_id in problems.keys():
        # Collect all solutions for this task
        candidates = []
        for aug_type, results in all_results.items():
            task_result = next((r for r in results if r['task_id'] == task_id), None)
            if task_result:
                candidates.append(task_result)
        
        if not candidates:
            continue
        
        query = problems[task_id]['prompt']
        
        try:
            # Use the reranking function from the original implementation
            best_solution = rerank_one_solution(query, candidates, vo)
            reranked_results.append(best_solution)
        except Exception as e:
            logging.error(f"Error during re-ranking for {task_id}: {e}")
            # Fall back to first valid solution
            valid_candidates = [c for c in candidates if c.get('passed', False)]
            if valid_candidates:
                reranked_results.append(valid_candidates[0])
            elif candidates:
                reranked_results.append(candidates[0])
    
    return reranked_results


def calculate_metrics(results: List[Dict]) -> Dict[str, float]:
    """Calculate evaluation metrics"""
    total = len(results)
    if total == 0:
        return {}
    
    passed = sum(1 for r in results if r.get('passed', False))
    syntactically_valid = sum(1 for r in results if r.get('is_syntactically_valid', False))
    
    return {
        'pass@1': passed / total,
        'syntax_accuracy': syntactically_valid / total,
        'total_problems': total,
        'passed_problems': passed,
        'syntactically_valid_problems': syntactically_valid
    }


def save_results(config: ExperimentConfig, all_results: Dict[str, List[Dict]], 
                reranked_results: List[Dict], metrics: Dict[str, Dict[str, float]]):
    """Save experiment results"""
    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(output_dir / 'config.json', 'w') as f:
        config_dict = asdict(config)
        # Remove non-serializable items
        if 'api_key' in config_dict:
            config_dict['api_key'] = '***hidden***'
        json.dump(config_dict, f, indent=2)
    
    # Save individual results
    for aug_type, results in all_results.items():
        output_file = output_dir / f'{aug_type}_results.jsonl'
        write_jsonl(str(output_file), results)
    
    # Save reranked results
    if reranked_results:
        write_jsonl(str(output_dir / 'reranked_results.jsonl'), reranked_results)
    
    # Save metrics
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Create summary report
    create_summary_report(output_dir, config, metrics)


def create_summary_report(output_dir: Path, config: ExperimentConfig, metrics: Dict[str, Dict[str, float]]):
    """Create a summary report of the experiment"""
    report_path = output_dir / 'summary_report.md'
    
    with open(report_path, 'w') as f:
        f.write(f"# Experiment Summary Report\n\n")
        f.write(f"**Model**: {config.model_name}\n")
        f.write(f"**Model Type**: {config.model_type}\n")
        f.write(f"**Benchmark**: {config.benchmark}\n")
        f.write(f"**Temperature**: {config.temperature}\n")
        f.write(f"**Max Tokens**: {config.max_tokens}\n")
        f.write(f"**Re-ranking Enabled**: {config.enable_reranking}\n\n")
        
        f.write("## Results by Augmentation Type\n\n")
        f.write("| Augmentation Type | Pass@1 | Syntax Accuracy | Passed/Total |\n")
        f.write("|-------------------|--------|-----------------|-------------|\n")
        
        for aug_type, metric in metrics.items():
            if aug_type != 'reranked':
                pass_at_1 = metric.get('pass@1', 0) * 100
                syntax_acc = metric.get('syntax_accuracy', 0) * 100
                passed = metric.get('passed_problems', 0)
                total = metric.get('total_problems', 0)
                f.write(f"| {aug_type} | {pass_at_1:.1f}% | {syntax_acc:.1f}% | {passed}/{total} |\n")
        
        if 'reranked' in metrics:
            metric = metrics['reranked']
            pass_at_1 = metric.get('pass@1', 0) * 100
            syntax_acc = metric.get('syntax_accuracy', 0) * 100
            passed = metric.get('passed_problems', 0)
            total = metric.get('total_problems', 0)
            f.write(f"| **Reranked** | **{pass_at_1:.1f}%** | **{syntax_acc:.1f}%** | **{passed}/{total}** |\n")
        
        f.write(f"\n## Experiment Configuration\n\n")
        f.write(f"```json\n")
        config_dict = asdict(config)
        if 'api_key' in config_dict:
            config_dict['api_key'] = '***hidden***'
        f.write(json.dumps(config_dict, indent=2))
        f.write(f"\n```\n")


def main():
    parser = argparse.ArgumentParser(description='Run PKG experiments with Claude and GPT models')
    
    # Model configuration
    parser.add_argument('--model-name', required=True, 
                       help='Model name (e.g., claude-3-sonnet-20240229, gpt-4, gpt-3.5-turbo)')
    parser.add_argument('--model-type', required=True, 
                       choices=['anthropic', 'openai'],
                       help='Type of model provider')
    parser.add_argument('--api-key', 
                       help='API key for the model (or set as environment variable)')
    
    # Experiment configuration
    parser.add_argument('--benchmark', choices=['humaneval', 'mbpp'], default='humaneval',
                       help='Benchmark to use for evaluation')
    parser.add_argument('--augmentation-types', nargs='+', 
                       choices=['no_rag', 'voyage_func', 'voyage_block', 'bm25'],
                       default=['no_rag', 'voyage_func', 'voyage_block', 'bm25'],
                       help='Augmentation types to test')
    parser.add_argument('--disable-reranking', action='store_true',
                       help='Disable solution re-ranking')
    
    # Output configuration
    parser.add_argument('--output-dir', default='experiment_results',
                       help='Directory to save results')
    parser.add_argument('--experiment-name',
                       help='Name for this experiment (auto-generated if not provided)')
    
    # Technical parameters
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='Temperature for generation (0.0 for greedy decoding)')
    parser.add_argument('--max-tokens', type=int, default=512,
                       help='Maximum tokens to generate')
    parser.add_argument('--timeout', type=int, default=30,
                       help='Timeout for API calls')
    
    # Logging
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Get API key from environment if not provided
    api_key = args.api_key
    if not api_key:
        env_vars = {
            'anthropic': 'ANTHROPIC_API_KEY',
            'openai': 'OPENAI_API_KEY'
        }
        if args.model_type in env_vars:
            api_key = os.getenv(env_vars[args.model_type])
            if not api_key:
                logging.error(f"API key not provided. Set {env_vars[args.model_type]} environment variable or use --api-key")
                return
    
    # Create configuration
    config = ExperimentConfig(
        model_name=args.model_name,
        model_type=args.model_type,
        api_key=api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        augmentation_types=args.augmentation_types,
        benchmark=args.benchmark,
        enable_reranking=not args.disable_reranking,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name
    )
    
    logging.info(f"Starting experiment: {config.experiment_name}")
    logging.info(f"Model: {config.model_name} ({config.model_type})")
    logging.info(f"Benchmark: {config.benchmark}")
    logging.info(f"Augmentation types: {config.augmentation_types}")
    
    # Run experiments for each augmentation type
    all_results = {}
    metrics = {}
    
    for aug_type in config.augmentation_types:
        try:
            results = run_single_experiment(config, aug_type)
            all_results[aug_type] = results
            metrics[aug_type] = calculate_metrics(results)
            
            logging.info(f"Completed {aug_type}: Pass@1 = {metrics[aug_type]['pass@1']:.3f}")
            
        except Exception as e:
            logging.error(f"Error in {aug_type} experiment: {e}")
            continue
    
    # Perform re-ranking if enabled
    reranked_results = []
    if config.enable_reranking and len(all_results) > 1:
        try:
            reranked_results = perform_reranking(all_results, config)
            if reranked_results:
                metrics['reranked'] = calculate_metrics(reranked_results)
                logging.info(f"Re-ranking completed: Pass@1 = {metrics['reranked']['pass@1']:.3f}")
        except Exception as e:
            logging.error(f"Error during re-ranking: {e}")
    
    # Save all results
    save_results(config, all_results, reranked_results, metrics)
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Model: {config.model_name}")
    print(f"Benchmark: {config.benchmark}")
    print("-"*60)
    
    for aug_type, metric in metrics.items():
        pass_at_1 = metric.get('pass@1', 0) * 100
        passed = metric.get('passed_problems', 0)
        total = metric.get('total_problems', 0)
        print(f"{aug_type:15s}: {pass_at_1:5.1f}% ({passed}/{total})")
    
    print("-"*60)
    print(f"Results saved to: {Path(config.output_dir) / config.experiment_name}")
    print("="*60)


if __name__ == "__main__":
    main() 