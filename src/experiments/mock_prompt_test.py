#!/usr/bin/env python3
"""
Mock Prompt Test for PKG Experiments

This script generates all prompts for different configurations and augmentation types
without making actual LLM API calls. It's designed to verify that prompting is correct
and test all instances across all datasets and configurations.

Features:
- Tests all augmentation types: no_rag, voyage_func, voyage_block, bm25
- Tests both benchmarks: humaneval and mbpp
- Tests multiple model configurations
- Outputs detailed prompt information for verification
- Calculates input tokens and costs like real experiments
- No API costs incurred
"""

import os
import json
import logging
import argparse
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
import pandas as pd

# Token counting imports
try:
    import tiktoken
except ImportError:
    tiktoken = None

def load_model_pricing() -> Dict[str, Dict[str, float]]:
    """Load model pricing from CSV file"""
    pricing = {}
    
    # Try different possible paths for the CSV file
    csv_paths = [
        "model_pricing.csv",
        "../model_pricing.csv", 
        "../../model_pricing.csv"
    ]
    
    for csv_path in csv_paths:
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                for _, row in df.iterrows():
                    pricing[row['model_name']] = {
                        "input": float(row['input_price_per_mtok']),
                        "output": float(row['output_price_per_mtok'])
                    }
                logging.info(f"Loaded pricing for {len(pricing)} models from {csv_path}")
                return pricing
            except Exception as e:
                logging.warning(f"Error loading pricing from {csv_path}: {e}")
                continue
    
    # Fallback to hardcoded pricing if CSV not found
    logging.warning("Could not load pricing from CSV, using fallback pricing")
    return {
        # Anthropic Claude models
        "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
        "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
        "claude-3-7-sonnet-20250219": {"input": 3.00, "output": 15.00},
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3-5-sonnet-20240620": {"input": 3.00, "output": 15.00},
        "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
        "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
        
        # OpenAI GPT models
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-4": {"input": 30.00, "output": 60.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "gpt-3.5-turbo-0125": {"input": 0.50, "output": 1.50},
    }

# Load pricing at module level
MODEL_PRICING = load_model_pricing()

# Define prompt templates directly to avoid import issues
SYSTEM_PROMPT_TEMPLATE = "You are an expert Python programmer. Your task is to solve programming problems by writing clean, executable Python code."

USER_PROMPT_TEMPLATE_WITH_CONTEXT = """Solve the following problem:

{problem}

The following code might be helpful as reference:
{context}

If the helper code is useful, integrate its logic directly into your solution. Otherwise, ignore it.

Requirements:
- Write executable Python code
- Include all necessary imports
- Ensure the solution is self-contained
- Write your solution between [PYTHON] and [/PYTHON] tags

Your solution:"""

USER_PROMPT_TEMPLATE_NO_CONTEXT = """Solve the following problem:

{problem}

Requirements:
- Write executable Python code
- Include all necessary imports
- Ensure the solution is self-contained
- Write your solution between [PYTHON] and [/PYTHON] tags

Your solution:"""

FULL_PROMPT_TEMPLATE_WITH_CONTEXT = """You are an expert Python programmer. Solve the following problem:

{problem}

The following code might be helpful as reference:
{context}

If the helper code is useful, integrate its logic directly into your solution. Otherwise, ignore it.

Requirements:
- Write executable Python code
- Include all necessary imports
- Ensure the solution is self-contained
- Write your solution between [PYTHON] and [/PYTHON] tags

Your solution:"""

FULL_PROMPT_TEMPLATE_NO_CONTEXT = """You are an expert Python programmer. Solve the following problem:

{problem}

Requirements:
- Write executable Python code
- Include all necessary imports
- Ensure the solution is self-contained
- Write your solution between [PYTHON] and [/PYTHON] tags

Your solution:"""


def estimate_tokens_anthropic(text: str) -> int:
    """Estimate token count for Anthropic models (roughly 4 chars per token)"""
    return max(1, len(text) // 4)


def count_tokens_openai(text: str, model: str) -> int:
    """Count tokens for OpenAI models using tiktoken if available"""
    if not tiktoken:
        # Fallback estimation: roughly 4 characters per token
        return max(1, len(text) // 4)
    
    try:
        # Map model names to tiktoken encodings
        encoding_map = {
            "gpt-4": "cl100k_base",
            "gpt-4o": "o200k_base",
            "gpt-4o-mini": "o200k_base",
            "gpt-4-turbo": "cl100k_base",
            "gpt-3.5-turbo": "cl100k_base",
        }
        
        encoding_name = encoding_map.get(model, "cl100k_base")
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except Exception:
        # Fallback to estimation
        return max(1, len(text) // 4)


def calculate_input_cost(input_tokens: int, model_name: str) -> float:
    """Calculate cost for input tokens only (since we're not generating output)"""
    pricing = MODEL_PRICING.get(model_name, {"input": 0.0, "output": 0.0})
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    return input_cost


def count_tokens_for_model(text: str, model_name: str, model_type: str) -> int:
    """Count tokens for the appropriate model type"""
    if model_type == "anthropic":
        return estimate_tokens_anthropic(text)
    elif model_type == "openai":
        return count_tokens_openai(text, model_name)
    else:
        # Fallback estimation
        return max(1, len(text) // 4)


@dataclass
class MockExperimentConfig:
    """Configuration for mock experiment parameters"""
    model_name: str
    model_type: str  # 'openai' or 'anthropic'
    temperature: float = 0.0
    max_tokens: int = 512
    augmentation_types: List[str] = None
    benchmark: str = 'humaneval'  # 'humaneval' or 'mbpp'
    output_dir: str = 'mock_prompt_test_results'
    experiment_name: str = None
    
    def __post_init__(self):
        if self.augmentation_types is None:
            self.augmentation_types = ['no_rag', 'voyage_func', 'voyage_block', 'bm25']
        if self.experiment_name is None:
            self.experiment_name = f"mock_{self.model_name}_{self.benchmark}"


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def load_humaneval_problems() -> Dict:
    """Load HumanEval problems without heavy dependencies"""
    try:
        # Try to import and use human_eval
        from human_eval.data import read_problems
        return read_problems()
    except ImportError:
        logging.error("human_eval package not available. Please install it or provide HumanEval data.")
        # Fallback: create a small sample problem for testing
        return {
            "HumanEval/0": {
                "prompt": "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n",
                "test": "check(has_close_elements)",
                "solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n"
            }
        }


def load_mbpp_problems() -> Dict:
    """Load MBPP problems from CSV file"""
    try:
        # First try the path from the experiment runner
        mbpp_path = "src/data/mbpp.csv"
        if not os.path.exists(mbpp_path):
            # Try root directory
            mbpp_path = "mbpp.csv"
        
        if not os.path.exists(mbpp_path):
            logging.error("MBPP dataset not found. Please ensure mbpp.csv is available.")
            return {}
            
        mbpp_df = pd.read_csv(mbpp_path)
        problems = {}
        for idx, row in mbpp_df.iterrows():
            task_id = f"MBPP/{idx}"
            problems[task_id] = {
                'prompt': row['text'],
                'test': row.get('test_list', ''),
                'solution': row.get('code', '')
            }
        logging.info(f"Loaded {len(problems)} MBPP problems")
        return problems
    except Exception as e:
        logging.error(f"Error loading MBPP dataset: {e}")
        return {}


def load_augmented_data(augmentation_type: str, benchmark: str) -> Dict[str, Any]:
    """Load augmented data based on type and benchmark"""
    file_mapping = {
        'voyage_func': f'src/data/augmented_problems/{benchmark}_function_wise_relevant_context.jsonl',
        'voyage_block': f'src/data/augmented_problems/{benchmark}_blockwise_relevant_context.jsonl',
        'bm25': f'src/data/augmented_problems/bm25_relevant_context_{benchmark}.jsonl'
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
                    # BM25 has different structure - can have either 'content' or 'problem' key
                    content = data.get('content', '')
                    if not content:
                        # Some BM25 files use 'problem' key instead of 'content'
                        content = data.get('problem', '')
                    # Handle case where content is a list of strings
                    if isinstance(content, list):
                        content = '\n\n'.join(content)
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
                
        logging.info(f"Loaded augmented data for {len(augmented_data)} problems from {file_path}")
    except Exception as e:
        logging.error(f"Error loading augmented data from {file_path}: {e}")
        return {}
    
    return augmented_data


def create_full_prompt(problem: str, augmented_data: Optional[str] = None) -> str:
    """Create full prompt using templates"""
    if augmented_data:
        return FULL_PROMPT_TEMPLATE_WITH_CONTEXT.format(
            problem=problem,
            context=augmented_data
        )
    else:
        return FULL_PROMPT_TEMPLATE_NO_CONTEXT.format(problem=problem)


def create_system_user_prompts(problem: str, augmented_data: Optional[str] = None) -> tuple[str, str]:
    """Create separate system and user prompts"""
    system_prompt = SYSTEM_PROMPT_TEMPLATE
    
    if augmented_data:
        user_prompt = USER_PROMPT_TEMPLATE_WITH_CONTEXT.format(
            problem=problem,
            context=augmented_data
        )
    else:
        user_prompt = USER_PROMPT_TEMPLATE_NO_CONTEXT.format(problem=problem)
    
    return system_prompt, user_prompt


def mock_generate_response(prompt: str, config: MockExperimentConfig) -> dict:
    """Mock LLM response generation - returns empty response with prompt details and token/cost info"""
    # Calculate input tokens
    input_tokens = count_tokens_for_model(prompt, config.model_name, config.model_type)
    
    # Calculate input cost
    input_cost = calculate_input_cost(input_tokens, config.model_name)
    
    return {
        'prompt': prompt,
        'response': '',  # Empty response for mock
        'model_name': config.model_name,
        'model_type': config.model_type,
        'temperature': config.temperature,
        'max_tokens': config.max_tokens,
        'prompt_length': len(prompt),
        'prompt_word_count': len(prompt.split()),
        'input_tokens': input_tokens,
        'input_cost': input_cost,
        'timestamp': datetime.now().isoformat()
    }


def run_mock_experiment(config: MockExperimentConfig, augmentation_type: str) -> List[Dict]:
    """Run mock experiment for a single augmentation type"""
    logging.info(f"🧪 Running MOCK experiment: {config.model_name} with {augmentation_type}")
    
    # Load problems
    if config.benchmark == 'humaneval':
        problems = load_humaneval_problems()
    elif config.benchmark == 'mbpp':
        problems = load_mbpp_problems()
    else:
        raise ValueError(f"Unknown benchmark: {config.benchmark}")
    
    if not problems:
        logging.error(f"No problems loaded for benchmark: {config.benchmark}")
        return []
    
    # Load augmented data if needed
    augmented_data = {}
    if augmentation_type != 'no_rag':
        augmented_data = load_augmented_data(augmentation_type, config.benchmark)
    
    results = []
    total_problems = len(problems)
    
    logging.info(f"🚀 Starting MOCK {augmentation_type} experiment with {total_problems} problems")
    
    for i, (task_id, problem_data) in enumerate(problems.items(), 1):
        problem_prompt = problem_data['prompt']
        
        # Get augmented context if available
        context = augmented_data.get(task_id, None) if augmentation_type != 'no_rag' else None
        
        # Log prompt creation details
        logging.info(f"\n🔄 Processing {task_id} ({i}/{total_problems})")
        logging.info(f"📝 Problem: {problem_prompt[:100]}...")
        logging.info(f"🔧 Augmentation Type: {augmentation_type}")
        
        if context:
            logging.info(f"📚 Augmented Context Length: {len(context)} chars")
            logging.info(f"📚 Augmented Context Preview: {context[:150]}...")
        else:
            logging.info(f"📚 No augmented context (using {augmentation_type})")
        
        # Create prompts
        full_prompt = create_full_prompt(problem_prompt, context)
        system_prompt, user_prompt = create_system_user_prompts(problem_prompt, context)
        
        logging.info(f"📋 Created full prompt length: {len(full_prompt)} chars")
        logging.info(f"📋 System prompt length: {len(system_prompt)} chars")
        logging.info(f"📋 User prompt length: {len(user_prompt)} chars")
        
        # Mock generation with token and cost calculation
        mock_response = mock_generate_response(full_prompt, config)
        
        logging.info(f"🔢 Input tokens: {mock_response['input_tokens']:,}")
        logging.info(f"💰 Input cost: ${mock_response['input_cost']:.6f}")
        
        # Store result
        result = {
            'task_id': task_id,
            'augmentation_type': augmentation_type,
            'problem': problem_prompt,
            'context': context,
            'full_prompt': full_prompt,
            'system_prompt': system_prompt,
            'user_prompt': user_prompt,
            'mock_response': mock_response,
            'prompt_stats': {
                'full_prompt_length': len(full_prompt),
                'full_prompt_words': len(full_prompt.split()),
                'system_prompt_length': len(system_prompt),
                'user_prompt_length': len(user_prompt),
                'context_length': len(context) if context else 0,
                'has_context': context is not None,
                'input_tokens': mock_response['input_tokens'],
                'input_cost': mock_response['input_cost']
            }
        }
        
        results.append(result)
        
        # Log every 10th problem for progress tracking
        if i % 10 == 0:
            logging.info(f"✅ Processed {i}/{total_problems} problems for {augmentation_type}")
    
    logging.info(f"🎉 Completed MOCK {augmentation_type} experiment with {len(results)} results")
    return results


def save_mock_results(output_dir: Path, config: MockExperimentConfig, all_results: Dict[str, List[Dict]]):
    """Save mock experiment results"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results for each augmentation type
    for aug_type, results in all_results.items():
        # Save full results
        results_file = output_dir / f"mock_{aug_type}_detailed_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save prompt-only results (lighter file)
        prompts_file = output_dir / f"mock_{aug_type}_prompts_only.json"
        prompts_only = []
        for result in results:
            prompts_only.append({
                'task_id': result['task_id'],
                'augmentation_type': result['augmentation_type'],
                'full_prompt': result['full_prompt'],
                'system_prompt': result['system_prompt'],
                'user_prompt': result['user_prompt'],
                'prompt_stats': result['prompt_stats']
            })
        
        with open(prompts_file, 'w') as f:
            json.dump(prompts_only, f, indent=2)
        
        logging.info(f"💾 Saved {aug_type} results: {results_file}")
        logging.info(f"💾 Saved {aug_type} prompts: {prompts_file}")
    
    # Create summary report
    summary_file = output_dir / "mock_experiment_summary.json"
    summary = {
        'experiment_name': config.experiment_name,
        'model_name': config.model_name,
        'model_type': config.model_type,
        'benchmark': config.benchmark,
        'augmentation_types': config.augmentation_types,
        'timestamp': datetime.now().isoformat(),
        'results_summary': {}
    }
    
    total_tokens = 0
    total_cost = 0.0
    
    for aug_type, results in all_results.items():
        if results:  # Only process if we have results
            aug_tokens = sum(r['prompt_stats']['input_tokens'] for r in results)
            aug_cost = sum(r['prompt_stats']['input_cost'] for r in results)
            
            total_tokens += aug_tokens
            total_cost += aug_cost
            
            summary['results_summary'][aug_type] = {
                'total_problems': len(results),
                'avg_full_prompt_length': sum(r['prompt_stats']['full_prompt_length'] for r in results) / len(results),
                'avg_full_prompt_words': sum(r['prompt_stats']['full_prompt_words'] for r in results) / len(results),
                'problems_with_context': sum(1 for r in results if r['prompt_stats']['has_context']),
                'problems_without_context': sum(1 for r in results if not r['prompt_stats']['has_context']),
                'avg_context_length': sum(r['prompt_stats']['context_length'] for r in results if r['prompt_stats']['has_context']) / max(1, sum(1 for r in results if r['prompt_stats']['has_context'])),
                'total_input_tokens': aug_tokens,
                'total_input_cost': aug_cost,
                'avg_input_tokens_per_problem': aug_tokens / len(results),
                'avg_input_cost_per_problem': aug_cost / len(results)
            }
    
    # Add overall totals
    summary['overall_totals'] = {
        'total_input_tokens': total_tokens,
        'total_input_cost': total_cost,
        'total_problems': sum(len(results) for results in all_results.values())
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logging.info(f"📊 Saved summary report: {summary_file}")


def create_prompt_analysis_report(output_dir: Path, all_results: Dict[str, List[Dict]]):
    """Create a detailed analysis report of all prompts"""
    report_file = output_dir / "prompt_analysis_report.md"
    
    # Calculate overall totals
    total_tokens = 0
    total_cost = 0.0
    total_problems = 0
    
    for results in all_results.values():
        if results:
            total_tokens += sum(r['prompt_stats']['input_tokens'] for r in results)
            total_cost += sum(r['prompt_stats']['input_cost'] for r in results)
            total_problems += len(results)
    
    with open(report_file, 'w') as f:
        f.write("# Mock Prompt Test Analysis Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Add overall summary
        f.write("## Overall Summary\n\n")
        f.write(f"- **Total problems processed**: {total_problems:,}\n")
        f.write(f"- **Total input tokens**: {total_tokens:,}\n")
        f.write(f"- **Total input cost**: ${total_cost:.6f}\n")
        f.write(f"- **Average tokens per problem**: {total_tokens / max(1, total_problems):.1f}\n")
        f.write(f"- **Average cost per problem**: ${total_cost / max(1, total_problems):.6f}\n\n")
        f.write("---\n\n")
        
        for aug_type, results in all_results.items():
            f.write(f"## {aug_type.upper()} Augmentation Type\n\n")
            f.write(f"- Total problems: {len(results)}\n")
            
            if results:
                avg_prompt_len = sum(r['prompt_stats']['full_prompt_length'] for r in results) / len(results)
                total_aug_tokens = sum(r['prompt_stats']['input_tokens'] for r in results)
                total_aug_cost = sum(r['prompt_stats']['input_cost'] for r in results)
                
                f.write(f"- Average prompt length: {avg_prompt_len:.1f} characters\n")
                f.write(f"- **Total input tokens**: {total_aug_tokens:,}\n")
                f.write(f"- **Total input cost**: ${total_aug_cost:.6f}\n")
                f.write(f"- **Average tokens per problem**: {total_aug_tokens / len(results):.1f}\n")
                f.write(f"- **Average cost per problem**: ${total_aug_cost / len(results):.6f}\n")
                
                with_context = sum(1 for r in results if r['prompt_stats']['has_context'])
                f.write(f"- Problems with context: {with_context}\n")
                f.write(f"- Problems without context: {len(results) - with_context}\n")
                
                if with_context > 0:
                    avg_context_len = sum(r['prompt_stats']['context_length'] for r in results if r['prompt_stats']['has_context']) / with_context
                    f.write(f"- Average context length: {avg_context_len:.1f} characters\n")
                
                # Show first example prompt
                f.write(f"\n### Example Prompt for {aug_type}:\n\n")
                f.write("```\n")
                f.write(results[0]['full_prompt'][:1000])
                if len(results[0]['full_prompt']) > 1000:
                    f.write("\n... (truncated)")
                f.write("\n```\n\n")
            
            f.write("---\n\n")
    
    logging.info(f"📝 Created analysis report: {report_file}")


def main():
    """Main function to run mock prompt tests"""
    parser = argparse.ArgumentParser(description='Run mock prompt tests for PKG experiments')
    
    # Model configuration
    parser.add_argument('--model-name', default='claude-sonnet-4-20250514',
                       help='Model name for mock test (default: claude-sonnet-4-20250514)')
    parser.add_argument('--model-type', choices=['anthropic', 'openai'], default='anthropic',
                       help='Type of model provider')
    
    # Experiment configuration
    parser.add_argument('--benchmark', choices=['humaneval', 'mbpp'], default='humaneval',
                       help='Benchmark to use for testing')
    parser.add_argument('--augmentation-types', nargs='+', 
                       choices=['no_rag', 'voyage_func', 'voyage_block', 'bm25'],
                       default=['no_rag', 'voyage_func', 'voyage_block', 'bm25'],
                       help='Augmentation types to test')
    
    # Output configuration
    parser.add_argument('--output-dir', default='mock_prompt_test_results',
                       help='Directory to save mock test results')
    parser.add_argument('--experiment-name',
                       help='Name for this mock experiment (auto-generated if not provided)')
    
    # Test all benchmarks and augmentation types
    parser.add_argument('--test-all', action='store_true',
                       help='Test all combinations of benchmarks and augmentation types')
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Determine test configurations
    test_configs = []
    
    if args.test_all:
        # Test all combinations
        benchmarks = ['humaneval', 'mbpp']
        model_configs = [
            ('claude-sonnet-4-20250514', 'anthropic'),
            # ('gpt-4', 'openai'),
            # ('gpt-3.5-turbo', 'openai')
        ]
        
        for benchmark in benchmarks:
            for model_name, model_type in model_configs:
                config = MockExperimentConfig(
                    model_name=model_name,
                    model_type=model_type,
                    benchmark=benchmark,
                    augmentation_types=args.augmentation_types,
                    output_dir=args.output_dir
                )
                test_configs.append(config)
    else:
        # Single configuration
        config = MockExperimentConfig(
            model_name=args.model_name,
            model_type=args.model_type,
            benchmark=args.benchmark,
            augmentation_types=args.augmentation_types,
            output_dir=args.output_dir,
            experiment_name=args.experiment_name
        )
        test_configs.append(config)
    
    # Run all test configurations
    for config in test_configs:
        logging.info(f"🎯 Starting mock experiment: {config.experiment_name}")
        logging.info(f"📊 Model: {config.model_name} ({config.model_type})")
        logging.info(f"📊 Benchmark: {config.benchmark}")
        logging.info(f"📊 Augmentation types: {config.augmentation_types}")
        
        # Create output directory for this config
        config_output_dir = Path(config.output_dir) / config.experiment_name
        
        # Run mock experiments for each augmentation type
        all_results = {}
        
        for aug_type in config.augmentation_types:
            try:
                results = run_mock_experiment(config, aug_type)
                all_results[aug_type] = results
                logging.info(f"✅ Completed {aug_type} mock experiment")
            except Exception as e:
                logging.error(f"❌ Failed {aug_type} mock experiment: {e}")
                continue
        
        # Save results
        if all_results:
            save_mock_results(config_output_dir, config, all_results)
            create_prompt_analysis_report(config_output_dir, all_results)
            
            # Print summary
            total_prompts = sum(len(results) for results in all_results.values())
            logging.info(f"🎉 Mock experiment completed!")
            logging.info(f"📊 Total prompts generated: {total_prompts}")
            logging.info(f"📊 Augmentation types tested: {len(all_results)}")
            logging.info(f"📁 Results saved to: {config_output_dir}")
        else:
            logging.error(f"❌ No results generated for {config.experiment_name}")
    
    logging.info("🏁 All mock experiments completed!")


if __name__ == "__main__":
    main() 