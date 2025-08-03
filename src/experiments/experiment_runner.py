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
- Comprehensive logging and result tracking with token usage, pricing, and timing
"""

import os
import json
import time
import ast
import signal
import argparse
import logging
from typing import Dict, List, Tuple, Optional, Any, Union, NamedTuple
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

# LLM Framework imports
try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

# Token counting imports
try:
    import tiktoken
except ImportError:
    tiktoken = None

# Evaluation imports
from human_eval.data import write_jsonl, read_problems
import voyageai

# Local imports
try:
    # Try relative imports first (when run as part of package)
    from ..core.reranker import rerank_one_solution, cosine_similarity, remove_comments_and_docstrings
    from .prompt_utils import *
except ImportError:
    # Fall back to absolute imports (when run directly)
    import sys
    from pathlib import Path
    
    # Add the src directory to the path
    src_path = Path(__file__).parent.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    from core.reranker import rerank_one_solution, cosine_similarity, remove_comments_and_docstrings
    from experiments.prompt_utils import *

try:
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file
except ImportError:
    pass  # dotenv not installed, rely on system environment variables


class GenerationMetrics(NamedTuple):
    """Metrics for a single generation request"""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0
    latency_seconds: float = 0.0
    timestamp: str = ""
    model_name: str = ""
    success: bool = True
    error_message: str = ""


# Pricing information (as of 2024) - costs per 1M tokens
MODEL_PRICING = {
    # Anthropic Claude models
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet-20240620": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 1.00, "output": 5.00},
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


def calculate_cost(input_tokens: int, output_tokens: int, model_name: str) -> tuple[float, float, float]:
    """Calculate costs for input, output, and total tokens"""
    pricing = MODEL_PRICING.get(model_name, {"input": 0.0, "output": 0.0})
    
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    total_cost = input_cost + output_cost
    
    return input_cost, output_cost, total_cost


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
    """Base class for LLM providers with comprehensive metrics tracking"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.setup_client()
        self.total_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_cost': 0.0,
            'total_latency': 0.0,
            'requests': []  # Store individual request metrics
        }
    
    def setup_client(self):
        """Setup the LLM client"""
        raise NotImplementedError
    
    def generate(self, prompt: str) -> tuple[str, GenerationMetrics]:
        """Generate code from prompt and return response with metrics"""
        raise NotImplementedError
    
    def get_summary_metrics(self) -> dict:
        """Get summary metrics for all requests"""
        if self.total_metrics['total_requests'] == 0:
            return self.total_metrics
        
        avg_latency = self.total_metrics['total_latency'] / self.total_metrics['total_requests']
        success_rate = self.total_metrics['successful_requests'] / self.total_metrics['total_requests']
        
        return {
            **self.total_metrics,
            'average_latency': avg_latency,
            'success_rate': success_rate,
            'average_input_tokens': self.total_metrics['total_input_tokens'] / self.total_metrics['total_requests'],
            'average_output_tokens': self.total_metrics['total_output_tokens'] / self.total_metrics['total_requests'],
            'average_cost_per_request': self.total_metrics['total_cost'] / self.total_metrics['total_requests']
        }


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider with comprehensive metrics tracking"""
    
    def setup_client(self):
        if not anthropic:
            raise ImportError("anthropic package not installed. Install with: pip install anthropic")
        self.client = anthropic.Anthropic(api_key=self.config.api_key)
    
    def generate(self, prompt: str) -> tuple[str, GenerationMetrics]:
        start_time = time.time()
        timestamp = datetime.now().isoformat()
        
        try:
            # Estimate input tokens
            input_tokens = estimate_tokens_anthropic(prompt)
            
            response = self.client.messages.create(
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            
            end_time = time.time()
            latency = end_time - start_time
            
            # Extract response text
            response_text = response.content[0].text
            
            # Get actual token usage from response
            actual_input_tokens = response.usage.input_tokens if hasattr(response, 'usage') else input_tokens
            actual_output_tokens = response.usage.output_tokens if hasattr(response, 'usage') else estimate_tokens_anthropic(response_text)
            total_tokens = actual_input_tokens + actual_output_tokens
            
            # Calculate costs
            input_cost, output_cost, total_cost = calculate_cost(
                actual_input_tokens, actual_output_tokens, self.config.model_name
            )
            
            metrics = GenerationMetrics(
                input_tokens=actual_input_tokens,
                output_tokens=actual_output_tokens,
                total_tokens=total_tokens,
                input_cost=input_cost,
                output_cost=output_cost,
                total_cost=total_cost,
                latency_seconds=latency,
                timestamp=timestamp,
                model_name=self.config.model_name,
                success=True,
                error_message=""
            )
            
            # Update total metrics
            self.total_metrics['total_requests'] += 1
            self.total_metrics['successful_requests'] += 1
            self.total_metrics['total_input_tokens'] += actual_input_tokens
            self.total_metrics['total_output_tokens'] += actual_output_tokens
            self.total_metrics['total_cost'] += total_cost
            self.total_metrics['total_latency'] += latency
            self.total_metrics['requests'].append(metrics._asdict())
            
            return response_text, metrics
            
        except Exception as e:
            end_time = time.time()
            latency = end_time - start_time
            
            logging.error(f"Anthropic API error: {e}")
            
            metrics = GenerationMetrics(
                input_tokens=estimate_tokens_anthropic(prompt),
                output_tokens=0,
                total_tokens=estimate_tokens_anthropic(prompt),
                input_cost=0.0,
                output_cost=0.0,
                total_cost=0.0,
                latency_seconds=latency,
                timestamp=timestamp,
                model_name=self.config.model_name,
                success=False,
                error_message=str(e)
            )
            
            # Update total metrics
            self.total_metrics['total_requests'] += 1
            self.total_metrics['failed_requests'] += 1
            self.total_metrics['total_latency'] += latency
            self.total_metrics['requests'].append(metrics._asdict())
            
            return "", metrics


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider with comprehensive metrics tracking"""
    
    def setup_client(self):
        if not openai:
            raise ImportError("openai package not installed. Install with: pip install openai")
        self.client = openai.OpenAI(api_key=self.config.api_key)
    
    def generate(self, prompt: str) -> tuple[str, GenerationMetrics]:
        start_time = time.time()
        timestamp = datetime.now().isoformat()
        
        try:
            # Estimate input tokens
            input_tokens = count_tokens_openai(prompt, self.config.model_name)
            
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            end_time = time.time()
            latency = end_time - start_time
            
            # Extract response text
            response_text = response.choices[0].message.content or ""
            
            # Get actual token usage from response
            actual_input_tokens = response.usage.prompt_tokens if response.usage else input_tokens
            actual_output_tokens = response.usage.completion_tokens if response.usage else count_tokens_openai(response_text, self.config.model_name)
            total_tokens = actual_input_tokens + actual_output_tokens
            
            # Calculate costs
            input_cost, output_cost, total_cost = calculate_cost(
                actual_input_tokens, actual_output_tokens, self.config.model_name
            )
            
            metrics = GenerationMetrics(
                input_tokens=actual_input_tokens,
                output_tokens=actual_output_tokens,
                total_tokens=total_tokens,
                input_cost=input_cost,
                output_cost=output_cost,
                total_cost=total_cost,
                latency_seconds=latency,
                timestamp=timestamp,
                model_name=self.config.model_name,
                success=True,
                error_message=""
            )
            
            # Update total metrics
            self.total_metrics['total_requests'] += 1
            self.total_metrics['successful_requests'] += 1
            self.total_metrics['total_input_tokens'] += actual_input_tokens
            self.total_metrics['total_output_tokens'] += actual_output_tokens
            self.total_metrics['total_cost'] += total_cost
            self.total_metrics['total_latency'] += latency
            self.total_metrics['requests'].append(metrics._asdict())
            
            return response_text, metrics
            
        except Exception as e:
            end_time = time.time()
            latency = end_time - start_time
            
            logging.error(f"OpenAI API error: {e}")
            
            metrics = GenerationMetrics(
                input_tokens=count_tokens_openai(prompt, self.config.model_name),
                output_tokens=0,
                total_tokens=count_tokens_openai(prompt, self.config.model_name),
                input_cost=0.0,
                output_cost=0.0,
                total_cost=0.0,
                latency_seconds=latency,
                timestamp=timestamp,
                model_name=self.config.model_name,
                success=False,
                error_message=str(e)
            )
            
            # Update total metrics
            self.total_metrics['total_requests'] += 1
            self.total_metrics['failed_requests'] += 1
            self.total_metrics['total_latency'] += latency
            self.total_metrics['requests'].append(metrics._asdict())
            
            return "", metrics


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
    """Run experiment for a single augmentation type with comprehensive metrics tracking"""
    experiment_start_time = time.time()
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
    total_problems = len(problems)
    
    # Metrics tracking
    experiment_metrics = {
        'augmentation_type': augmentation_type,
        'total_problems': total_problems,
        'start_time': datetime.now().isoformat(),
        'end_time': None,
        'duration_seconds': 0,
        'successful_generations': 0,
        'failed_generations': 0,
        'syntax_valid_count': 0,
        'passed_count': 0,
        'total_cost': 0.0,
        'total_tokens': 0,
        'individual_metrics': []
    }
    
    logging.info(f"🚀 Starting {augmentation_type} experiment with {total_problems} problems")
    logging.info(f"💰 Model pricing: Input=${MODEL_PRICING.get(config.model_name, {}).get('input', 'N/A')}/1M tokens, Output=${MODEL_PRICING.get(config.model_name, {}).get('output', 'N/A')}/1M tokens")
    
    for i, (task_id, problem_data) in enumerate(tqdm(problems.items(), desc=f"Processing {augmentation_type}"), 1):
        problem_prompt = problem_data['prompt']
        
        # Get augmented context if available
        context = augmented_data.get(task_id, None) if augmentation_type != 'no_rag' else None
        
        # Create prompt
        prompt = create_prompt(problem_prompt, config.model_type, context)
        
        # Generate solution
        try:
            raw_output, metrics = provider.generate(prompt)
            generated_code = extract_python_code(raw_output)
            
            # Evaluate solution
            is_valid = is_syntactically_valid(generated_code)
            passed = False
            
            if is_valid:
                passed = evaluate_solution(generated_code, task_id, problems)
            
            # Update experiment metrics
            experiment_metrics['successful_generations'] += 1
            if is_valid:
                experiment_metrics['syntax_valid_count'] += 1
            if passed:
                experiment_metrics['passed_count'] += 1
            experiment_metrics['total_cost'] += metrics.total_cost
            experiment_metrics['total_tokens'] += metrics.total_tokens
            experiment_metrics['individual_metrics'].append(metrics._asdict())
            
            result = {
                'task_id': task_id,
                'prompt': problem_prompt,
                'augmentation_type': augmentation_type,
                'raw_output': raw_output,
                'generated_code': generated_code,
                'is_syntactically_valid': is_valid,
                'passed': passed,
                'augmented_context': context if context else "",
                'metrics': metrics._asdict()
            }
            
            results.append(result)
            
            # Log progress every 10 problems or on completion
            if i % 10 == 0 or i == total_problems:
                current_pass_rate = experiment_metrics['passed_count'] / i * 100
                current_cost = experiment_metrics['total_cost']
                avg_latency = sum(m['latency_seconds'] for m in experiment_metrics['individual_metrics']) / len(experiment_metrics['individual_metrics'])
                
                logging.info(f"📊 Progress {i}/{total_problems}: Pass rate: {current_pass_rate:.1f}%, Cost: ${current_cost:.4f}, Avg latency: {avg_latency:.2f}s")
            
            logging.debug(f"✅ {task_id}: {'PASSED' if passed else 'FAILED'} (${metrics.total_cost:.6f}, {metrics.latency_seconds:.2f}s)")
            
        except Exception as e:
            logging.error(f"❌ Error processing {task_id}: {e}")
            
            # Create empty metrics for failed requests
            failed_metrics = GenerationMetrics(
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                input_cost=0.0,
                output_cost=0.0,
                total_cost=0.0,
                latency_seconds=0.0,
                timestamp=datetime.now().isoformat(),
                model_name=config.model_name,
                success=False,
                error_message=str(e)
            )
            
            experiment_metrics['failed_generations'] += 1
            experiment_metrics['individual_metrics'].append(failed_metrics._asdict())
            
            result = {
                'task_id': task_id,
                'prompt': problem_prompt,
                'augmentation_type': augmentation_type,
                'raw_output': "",
                'generated_code': "",
                'is_syntactically_valid': False,
                'passed': False,
                'error': str(e),
                'augmented_context': context if context else "",
                'metrics': failed_metrics._asdict()
            }
            results.append(result)
    
    # Finalize experiment metrics
    experiment_end_time = time.time()
    experiment_metrics['end_time'] = datetime.now().isoformat()
    experiment_metrics['duration_seconds'] = experiment_end_time - experiment_start_time
    
    # Get provider summary metrics
    provider_metrics = provider.get_summary_metrics()
    
    # Log comprehensive experiment summary
    log_experiment_summary(augmentation_type, experiment_metrics, provider_metrics, config)
    
    # Add experiment-level metrics to each result for later analysis
    for result in results:
        result['experiment_metrics'] = experiment_metrics
        result['provider_metrics'] = provider_metrics
    
    return results


def log_experiment_summary(augmentation_type: str, experiment_metrics: dict, provider_metrics: dict, config: ExperimentConfig):
    """Log a comprehensive summary of the experiment with all metrics"""
    
    logging.info("\n" + "="*80)
    logging.info(f"🎯 EXPERIMENT SUMMARY: {augmentation_type.upper()}")
    logging.info("="*80)
    
    # Basic metrics
    total_problems = experiment_metrics['total_problems']
    passed = experiment_metrics['passed_count']
    syntax_valid = experiment_metrics['syntax_valid_count']
    successful = experiment_metrics['successful_generations']
    failed = experiment_metrics['failed_generations']
    
    logging.info(f"📊 Results Overview:")
    logging.info(f"   • Total problems: {total_problems}")
    logging.info(f"   • Successful generations: {successful}/{total_problems} ({successful/total_problems*100:.1f}%)")
    logging.info(f"   • Failed generations: {failed}")
    logging.info(f"   • Syntax valid: {syntax_valid}/{total_problems} ({syntax_valid/total_problems*100:.1f}%)")
    logging.info(f"   • Passed tests: {passed}/{total_problems} ({passed/total_problems*100:.1f}%)")
    
    # Timing metrics
    duration = experiment_metrics['duration_seconds']
    avg_time_per_problem = duration / total_problems if total_problems > 0 else 0
    
    logging.info(f"\n⏱️  Timing Metrics:")
    logging.info(f"   • Total duration: {duration:.2f} seconds ({duration/60:.1f} minutes)")
    logging.info(f"   • Average per problem: {avg_time_per_problem:.2f} seconds")
    logging.info(f"   • Average API latency: {provider_metrics.get('average_latency', 0):.2f} seconds")
    
    # Token and cost metrics
    total_cost = experiment_metrics['total_cost']
    total_tokens = experiment_metrics['total_tokens']
    avg_cost_per_problem = total_cost / total_problems if total_problems > 0 else 0
    
    logging.info(f"\n💰 Cost & Token Metrics:")
    logging.info(f"   • Total cost: ${total_cost:.4f}")
    logging.info(f"   • Average cost per problem: ${avg_cost_per_problem:.4f}")
    logging.info(f"   • Total tokens: {total_tokens:,}")
    logging.info(f"   • Total input tokens: {provider_metrics.get('total_input_tokens', 0):,}")
    logging.info(f"   • Total output tokens: {provider_metrics.get('total_output_tokens', 0):,}")
    logging.info(f"   • Average tokens per request: {provider_metrics.get('average_input_tokens', 0) + provider_metrics.get('average_output_tokens', 0):.1f}")
    
    # API performance metrics
    success_rate = provider_metrics.get('success_rate', 0) * 100
    
    logging.info(f"\n🔌 API Performance:")
    logging.info(f"   • API success rate: {success_rate:.1f}%")
    logging.info(f"   • Total API requests: {provider_metrics.get('total_requests', 0)}")
    logging.info(f"   • Failed API requests: {provider_metrics.get('failed_requests', 0)}")
    
    # Model information
    model_pricing = MODEL_PRICING.get(config.model_name, {})
    input_price = model_pricing.get('input', 'N/A')
    output_price = model_pricing.get('output', 'N/A')
    
    logging.info(f"\n🤖 Model Information:")
    logging.info(f"   • Model: {config.model_name}")
    logging.info(f"   • Provider: {config.model_type}")
    logging.info(f"   • Input pricing: ${input_price}/1M tokens" if input_price != 'N/A' else "   • Input pricing: N/A")
    logging.info(f"   • Output pricing: ${output_price}/1M tokens" if output_price != 'N/A' else "   • Output pricing: N/A")
    
    logging.info("="*80 + "\n")


def save_detailed_metrics(output_dir: Path, config: ExperimentConfig, all_results: Dict[str, List[Dict]], 
                         all_provider_metrics: Dict[str, dict]):
    """Save detailed metrics including token usage, costs, and timing"""
    
    # Create metrics directory
    metrics_dir = output_dir / "detailed_metrics"
    metrics_dir.mkdir(exist_ok=True)
    
    # Save provider metrics for each augmentation type
    for aug_type, provider_metrics in all_provider_metrics.items():
        provider_file = metrics_dir / f"{aug_type}_provider_metrics.json"
        with open(provider_file, 'w') as f:
            json.dump(provider_metrics, f, indent=2)
    
    # Create comprehensive metrics summary
    comprehensive_metrics = {
        'experiment_info': {
            'model_name': config.model_name,
            'model_type': config.model_type,
            'benchmark': config.benchmark,
            'augmentation_types': config.augmentation_types,
            'timestamp': datetime.now().isoformat(),
            'experiment_name': config.experiment_name
        },
        'overall_summary': {},
        'per_augmentation_summary': {},
        'detailed_breakdowns': {}
    }
    
    # Calculate overall metrics across all augmentation types
    total_cost = 0
    total_tokens = 0
    total_requests = 0
    total_problems = 0
    total_passed = 0
    total_syntax_valid = 0
    
    for aug_type, results in all_results.items():
        # Extract metrics from results
        aug_cost = sum(r.get('metrics', {}).get('total_cost', 0) for r in results)
        aug_tokens = sum(r.get('metrics', {}).get('total_tokens', 0) for r in results)
        aug_passed = sum(1 for r in results if r.get('passed', False))
        aug_syntax_valid = sum(1 for r in results if r.get('is_syntactically_valid', False))
        aug_problems = len(results)
        
        total_cost += aug_cost
        total_tokens += aug_tokens
        total_problems += aug_problems
        total_passed += aug_passed
        total_syntax_valid += aug_syntax_valid
        
        # Per-augmentation summary
        comprehensive_metrics['per_augmentation_summary'][aug_type] = {
            'total_problems': aug_problems,
            'passed': aug_passed,
            'pass_rate': aug_passed / aug_problems if aug_problems > 0 else 0,
            'syntax_valid': aug_syntax_valid,
            'syntax_rate': aug_syntax_valid / aug_problems if aug_problems > 0 else 0,
            'total_cost': aug_cost,
            'total_tokens': aug_tokens,
            'avg_cost_per_problem': aug_cost / aug_problems if aug_problems > 0 else 0,
            'avg_tokens_per_problem': aug_tokens / aug_problems if aug_problems > 0 else 0
        }
        
        # Detailed breakdown with individual problem metrics
        comprehensive_metrics['detailed_breakdowns'][aug_type] = [
            {
                'task_id': r['task_id'],
                'passed': r.get('passed', False),
                'syntax_valid': r.get('is_syntactically_valid', False),
                'metrics': r.get('metrics', {})
            }
            for r in results
        ]
        
        if aug_type in all_provider_metrics:
            total_requests += all_provider_metrics[aug_type].get('total_requests', 0)
    
    # Overall summary
    comprehensive_metrics['overall_summary'] = {
        'total_cost': total_cost,
        'total_tokens': total_tokens,
        'total_requests': total_requests,
        'total_problems_across_all_types': total_problems,
        'total_passed_across_all_types': total_passed,
        'total_syntax_valid_across_all_types': total_syntax_valid,
        'overall_pass_rate': total_passed / total_problems if total_problems > 0 else 0,
        'overall_syntax_rate': total_syntax_valid / total_problems if total_problems > 0 else 0,
        'avg_cost_per_problem': total_cost / total_problems if total_problems > 0 else 0,
        'avg_tokens_per_problem': total_tokens / total_problems if total_problems > 0 else 0
    }
    
    # Save comprehensive metrics
    comprehensive_file = metrics_dir / "comprehensive_metrics.json"
    with open(comprehensive_file, 'w') as f:
        json.dump(comprehensive_metrics, f, indent=2)
    
    # Create a CSV summary for easy analysis
    csv_data = []
    for aug_type, summary in comprehensive_metrics['per_augmentation_summary'].items():
        csv_data.append({
            'augmentation_type': aug_type,
            'model_name': config.model_name,
            'total_problems': summary['total_problems'],
            'passed': summary['passed'],
            'pass_rate': summary['pass_rate'],
            'syntax_valid': summary['syntax_valid'],
            'syntax_rate': summary['syntax_rate'],
            'total_cost': summary['total_cost'],
            'total_tokens': summary['total_tokens'],
            'avg_cost_per_problem': summary['avg_cost_per_problem'],
            'avg_tokens_per_problem': summary['avg_tokens_per_problem']
        })
    
    if csv_data:
        csv_file = metrics_dir / "experiment_summary.csv"
        pd.DataFrame(csv_data).to_csv(csv_file, index=False)
    
    logging.info(f"📊 Detailed metrics saved to: {metrics_dir}")
    logging.info(f"   • Comprehensive metrics: comprehensive_metrics.json")
    logging.info(f"   • CSV summary: experiment_summary.csv")
    logging.info(f"   • Provider metrics: {len(all_provider_metrics)} individual files")


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
    """Create a comprehensive summary report of the experiment"""
    report_path = output_dir / 'summary_report.md'
    
    with open(report_path, 'w') as f:
        f.write(f"# Experiment Summary Report\n\n")
        f.write(f"**Experiment Name**: {config.experiment_name}\n")
        f.write(f"**Model**: {config.model_name}\n")
        f.write(f"**Model Type**: {config.model_type}\n")
        f.write(f"**Benchmark**: {config.benchmark}\n")
        f.write(f"**Temperature**: {config.temperature}\n")
        f.write(f"**Max Tokens**: {config.max_tokens}\n")
        f.write(f"**Re-ranking Enabled**: {config.enable_reranking}\n")
        f.write(f"**Report Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Calculate totals across all augmentation types
        total_cost = sum(metric.get('total_cost', 0) for metric in metrics.values() if 'error' not in metric)
        total_tokens = sum(metric.get('total_tokens', 0) for metric in metrics.values() if 'error' not in metric)
        total_problems = sum(metric.get('total', 0) for metric in metrics.values() if 'error' not in metric)
        total_passed = sum(metric.get('passed', 0) for metric in metrics.values() if 'error' not in metric)
        
        f.write("## Overall Summary\n\n")
        f.write(f"- **Total Cost**: ${total_cost:.4f}\n")
        f.write(f"- **Total Tokens**: {total_tokens:,}\n")
        f.write(f"- **Total Problems Processed**: {total_problems}\n")
        f.write(f"- **Total Problems Passed**: {total_passed}\n")
        f.write(f"- **Overall Pass Rate**: {(total_passed/total_problems*100) if total_problems > 0 else 0:.1f}%\n\n")
        
        # Model pricing information
        model_pricing = MODEL_PRICING.get(config.model_name, {})
        if model_pricing:
            f.write("## Model Pricing\n\n")
            f.write(f"- **Input**: ${model_pricing.get('input', 'N/A')}/1M tokens\n")
            f.write(f"- **Output**: ${model_pricing.get('output', 'N/A')}/1M tokens\n\n")
        
        f.write("## Results by Augmentation Type\n\n")
        f.write("| Augmentation Type | Pass@1 | Syntax Accuracy | Passed/Total | Cost ($) | Tokens | Avg Cost/Problem | Avg Latency (s) |\n")
        f.write("|-------------------|--------|-----------------|--------------|----------|--------|------------------|------------------|\n")
        
        for aug_type, metric in metrics.items():
            if aug_type != 'reranked' and 'error' not in metric:
                pass_at_1 = metric.get('pass_at_1', 0) * 100
                syntax_acc = metric.get('syntax_accuracy', 0) * 100
                passed = metric.get('passed', 0)
                total = metric.get('total', 0)
                cost = metric.get('total_cost', 0)
                tokens = metric.get('total_tokens', 0)
                avg_cost = metric.get('avg_cost_per_problem', 0)
                avg_latency = metric.get('avg_latency', 0)
                
                f.write(f"| {aug_type} | {pass_at_1:.1f}% | {syntax_acc:.1f}% | {passed}/{total} | {cost:.4f} | {tokens:,} | {avg_cost:.6f} | {avg_latency:.2f} |\n")
            elif 'error' in metric:
                f.write(f"| {aug_type} | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR |\n")
        
        if 'reranked' in metrics and 'error' not in metrics['reranked']:
            metric = metrics['reranked']
            pass_at_1 = metric.get('pass_at_1', 0) * 100
            syntax_acc = metric.get('syntax_accuracy', 0) * 100
            passed = metric.get('passed', 0)
            total = metric.get('total', 0)
            cost = metric.get('total_cost', 0)
            tokens = metric.get('total_tokens', 0)
            avg_cost = metric.get('avg_cost_per_problem', 0)
            avg_latency = metric.get('avg_latency', 0)
            
            f.write(f"| **Reranked** | **{pass_at_1:.1f}%** | **{syntax_acc:.1f}%** | **{passed}/{total}** | **{cost:.4f}** | **{tokens:,}** | **{avg_cost:.6f}** | **{avg_latency:.2f}** |\n")
        
        # Performance analysis
        valid_metrics = {k: v for k, v in metrics.items() if 'error' not in v and v.get('total', 0) > 0}
        if valid_metrics:
            f.write("\n## Performance Analysis\n\n")
            
            # Best performing
            best_aug_type = max(valid_metrics.keys(), key=lambda x: valid_metrics[x]['pass_at_1'])
            best_pass_rate = valid_metrics[best_aug_type]['pass_at_1'] * 100
            f.write(f"- **Best Performing**: {best_aug_type} ({best_pass_rate:.1f}% pass rate)\n")
            
            # Most cost-efficient
            cost_efficiency = {k: v['pass_at_1'] / max(v['total_cost'], 0.0001) for k, v in valid_metrics.items()}
            most_efficient = max(cost_efficiency.keys(), key=lambda x: cost_efficiency[x])
            efficiency_score = cost_efficiency[most_efficient]
            f.write(f"- **Most Cost-Efficient**: {most_efficient} ({efficiency_score:.1f} pass rate per $)\n")
            
            # Fastest
            if all('avg_latency' in v for v in valid_metrics.values()):
                fastest = min(valid_metrics.keys(), key=lambda x: valid_metrics[x]['avg_latency'])
                fastest_time = valid_metrics[fastest]['avg_latency']
                f.write(f"- **Fastest**: {fastest} ({fastest_time:.2f}s avg latency)\n")
        
        # Error summary
        error_metrics = {k: v for k, v in metrics.items() if 'error' in v}
        if error_metrics:
            f.write("\n## Errors\n\n")
            for aug_type, metric in error_metrics.items():
                f.write(f"- **{aug_type}**: {metric['error']}\n")
        
        f.write(f"\n## Detailed Metrics\n\n")
        f.write("Detailed metrics including individual request data are available in the `detailed_metrics/` directory:\n\n")
        f.write("- `comprehensive_metrics.json`: Complete metrics breakdown\n")
        f.write("- `experiment_summary.csv`: CSV format for analysis\n")
        f.write("- `*_provider_metrics.json`: API provider metrics for each augmentation type\n\n")
        
        f.write(f"## Experiment Configuration\n\n")
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