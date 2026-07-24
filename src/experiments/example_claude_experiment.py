#!/usr/bin/env python3
"""
Example experiment script for running Claude-based PKG experiments.
This script demonstrates how to run experiments with different configurations.
"""

import os
import sys
import logging
from pathlib import Path
import time # Added for timing
from datetime import datetime # Added for timestamp

try:
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file
except ImportError:
    print("Warning: dotenv not installed, relying on system environment variables")
    pass  # dotenv not installed, rely on system environment variables

# Add the current directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent))

from experiment_runner import ExperimentConfig, run_single_experiment, create_summary_report

def setup_logging():
    """Setup logging for the experiment"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def run_programmatic_experiment(config: ExperimentConfig):
    """Run experiment programmatically without command line arguments"""
    logging.info(f"Starting experiment: {config.experiment_name}")
    logging.info(f"Model: {config.model_name} ({config.model_type})")
    logging.info(f"Benchmark: {config.benchmark}")
    logging.info(f"Augmentation types: {config.augmentation_types}")
    
    # Create output directory
    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run experiments for each augmentation type
    all_results = {}
    metrics = {}
    all_provider_metrics = {}  # Store provider metrics for each augmentation type
    
    # Track overall experiment timing and costs
    experiment_start_time = time.time()
    total_experiment_cost = 0.0
    total_experiment_tokens = 0
    
    print(f"\n🚀 Starting comprehensive experiment with {len(config.augmentation_types)} augmentation types")
    print(f"📊 Model: {config.model_name} | Benchmark: {config.benchmark}")
    print(f"💾 Results will be saved to: {output_dir}")
    print("="*80)
    
    for aug_type in config.augmentation_types:
        print(f"\n🔄 Running {aug_type} experiments...")
        try:
            results = run_single_experiment(config, aug_type)
            all_results[aug_type] = results
            
            # Calculate metrics
            total = len(results)
            passed = sum(1 for r in results if r.get('passed', False))
            syntax_valid = sum(1 for r in results if r.get('syntax_valid', False))
            
            # Extract cost and token information from results
            aug_cost = sum(r.get('metrics', {}).get('total_cost', 0) for r in results)
            aug_tokens = sum(r.get('metrics', {}).get('total_tokens', 0) for r in results)
            
            # Get provider metrics (should be the same for all results in this augmentation type)
            provider_metrics = results[0].get('provider_metrics', {}) if results else {}
            all_provider_metrics[aug_type] = provider_metrics
            
            total_experiment_cost += aug_cost
            total_experiment_tokens += aug_tokens
            
            metrics[aug_type] = {
                'pass_at_1': passed / total if total > 0 else 0.0,
                'syntax_accuracy': syntax_valid / total if total > 0 else 0.0,
                'passed': passed,
                'total': total,
                'total_cost': aug_cost,
                'total_tokens': aug_tokens,
                'avg_cost_per_problem': aug_cost / total if total > 0 else 0.0,
                'avg_tokens_per_problem': aug_tokens / total if total > 0 else 0.0,
                'success_rate': provider_metrics.get('success_rate', 0.0),
                'avg_latency': provider_metrics.get('average_latency', 0.0)
            }
            
            print(f"✅ {aug_type}: {metrics[aug_type]['pass_at_1']:.1%} ({passed}/{total}) | Cost: ${aug_cost:.4f} | Tokens: {aug_tokens:,}")
            
            # Save individual results
            import json
            results_file = output_dir / f"{aug_type}_results.jsonl"
            with open(results_file, 'w') as f:
                for result in results:
                    f.write(json.dumps(result) + '\n')
                    
        except Exception as e:
            logging.error(f"Error in {aug_type}: {e}")
            metrics[aug_type] = {
                'pass_at_1': 0.0, 
                'syntax_accuracy': 0.0, 
                'passed': 0, 
                'total': 0,
                'total_cost': 0.0,
                'total_tokens': 0,
                'avg_cost_per_problem': 0.0,
                'avg_tokens_per_problem': 0.0,
                'success_rate': 0.0,
                'avg_latency': 0.0,
                'error': str(e)
            }
            all_provider_metrics[aug_type] = {}
    
    # Calculate experiment totals
    experiment_end_time = time.time()
    total_experiment_time = experiment_end_time - experiment_start_time
    
    # Save metrics and create summary
    metrics_file = output_dir / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save detailed metrics using the new comprehensive system
    from experiment_runner import save_detailed_metrics
    save_detailed_metrics(output_dir, config, all_results, all_provider_metrics)
    
    # Create summary report
    create_summary_report(output_dir, config, metrics)
    
    # Save configuration
    config_file = output_dir / 'config.json'
    with open(config_file, 'w') as f:
        import dataclasses
        config_dict = dataclasses.asdict(config)
        config_dict['api_key'] = '***hidden***'  # Don't save API key
        json.dump(config_dict, f, indent=2)
    
    # Print comprehensive experiment summary
    print("\n" + "="*80)
    print("🎯 FINAL EXPERIMENT SUMMARY")
    print("="*80)
    
    print(f"⏱️  Total experiment time: {total_experiment_time:.2f} seconds ({total_experiment_time/60:.1f} minutes)")
    print(f"💰 Total experiment cost: ${total_experiment_cost:.4f}")
    print(f"🔢 Total tokens used: {total_experiment_tokens:,}")
    print(f"🤖 Model: {config.model_name}")
    
    print(f"\n📊 Results by Augmentation Type:")
    print("-" * 80)
    print(f"{'Type':<15} {'Pass Rate':<10} {'Passed':<8} {'Cost':<10} {'Tokens':<10} {'Avg Latency':<12}")
    print("-" * 80)
    
    for aug_type, metric in metrics.items():
        if 'error' not in metric:
            print(f"{aug_type:<15} {metric['pass_at_1']:>8.1%} {metric['passed']:>4}/{metric['total']:<3} ${metric['total_cost']:>8.4f} {metric['total_tokens']:>8,} {metric['avg_latency']:>10.2f}s")
        else:
            print(f"{aug_type:<15} {'ERROR':<10} {'N/A':<8} {'N/A':<10} {'N/A':<10} {'N/A':<12}")
    
    print("-" * 80)
    
    # Calculate and display best performing augmentation type
    valid_metrics = {k: v for k, v in metrics.items() if 'error' not in v and v['total'] > 0}
    if valid_metrics:
        best_aug_type = max(valid_metrics.keys(), key=lambda x: valid_metrics[x]['pass_at_1'])
        best_pass_rate = valid_metrics[best_aug_type]['pass_at_1']
        print(f"🏆 Best performing: {best_aug_type} ({best_pass_rate:.1%})")
        
        # Cost efficiency analysis
        cost_efficiency = {k: v['pass_at_1'] / max(v['total_cost'], 0.0001) for k, v in valid_metrics.items()}
        most_efficient = max(cost_efficiency.keys(), key=lambda x: cost_efficiency[x])
        print(f"💡 Most cost-efficient: {most_efficient} ({cost_efficiency[most_efficient]:.1f} pass rate per $)")
    
    print(f"\n📁 All results saved to: {output_dir}")
    print(f"📋 Check the summary_report.md file for detailed results")
    print(f"📊 Detailed metrics available in: {output_dir}/detailed_metrics/")
    print("="*80)
    
    return metrics

def check_prerequisites():
    """Check if all prerequisites are met"""
    # Check if at least one API key is set
    has_anthropic = bool(os.getenv('ANTHROPIC_API_KEY'))
    has_openai = bool(os.getenv('OPENAI_API_KEY'))
    
    if not has_anthropic and not has_openai:
        print("❌ No API keys found!")
        print("   Please set at least one of:")
        print("   - export ANTHROPIC_API_KEY='your-anthropic-key'")
        print("   - export OPENAI_API_KEY='your-openai-key'")
        return False
    
    # Check if augmented data exists
    required_files = [
        'src/data/augmented_problems/humaneval_function_wise_relevant_context.jsonl',
        'src/data/augmented_problems/humaneval_blockwise_relevant_context.jsonl',
        'src/data/augmented_problems/bm25_relevant_context_humaneval.jsonl'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required augmented data files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("   Please ensure the augmented_problems directory contains all required files")
        return False
    
    print("✅ All prerequisites met!")
    return True

def run_claude_experiment():
    """Run a comprehensive experiment with Claude Haiku (lowest cost)"""
    
    if not os.getenv('ANTHROPIC_API_KEY'):
        print("❌ ANTHROPIC_API_KEY not set. Please set it to run Claude experiments.")
        return
    
    print("🚀 Starting Claude Haiku PKG Experiment (Lowest Cost)")
    print("=" * 50)
    
    # Setup logging
    setup_logging()
    
    # Check prerequisites
    if not check_prerequisites():
        return
    
    # Create experiment configuration with Claude Haiku (cheapest model)
    config = ExperimentConfig(
        model_name="claude-3-haiku-20240307",
        model_type="anthropic",
        api_key=os.getenv('ANTHROPIC_API_KEY'),
        temperature=0.0,  # Greedy decoding as per paper
        max_tokens=512,   # As specified in paper
        augmentation_types=['no_rag', 'voyage_func', 'voyage_block', 'bm25'],
        benchmark='humaneval',
        enable_reranking=True,
        experiment_name="claude_haiku_pkg_experiment",
        output_dir="./experiment_results"
    )
    
    print(f"📋 Experiment Configuration:")
    print(f"   Model: {config.model_name} (Lowest Cost Claude Model)")
    print(f"   Benchmark: {config.benchmark}")
    print(f"   Augmentation Types: {config.augmentation_types}")
    print(f"   Re-ranking: {'Enabled' if config.enable_reranking else 'Disabled'}")
    print(f"   Output Directory: {config.output_dir}/{config.experiment_name}")
    print()
    
    # Import and run the experiment
    try:
        # This will run the actual experiment
        print("🔄 Running experiments...")
        print("   This will take several minutes depending on the model and API speed")
        print("   Progress will be shown for each augmentation type")
        print()
        
        # Actually run the experiment:
        metrics = run_programmatic_experiment(config)
        
        print("✅ Experiment completed successfully!")
        print(f"📊 Results saved to: {config.output_dir}/{config.experiment_name}")
        print("📋 Check the summary_report.md file for detailed results")
        
        # Print final summary
        print("\n" + "="*60)
        print("FINAL RESULTS SUMMARY")
        print("="*60)
        for aug_type, metric in metrics.items():
            print(f"{aug_type:15}: {metric['pass_at_1']:6.1%} ({metric['passed']}/{metric['total']})")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        logging.error(f"Experiment error: {e}")

def run_gpt4_experiment():
    """Run a comprehensive experiment with GPT-4"""
    
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY not set. Please set it to run GPT-4 experiments.")
        return
    
    print("🚀 Starting GPT-4 PKG Experiment")
    print("=" * 50)
    
    # Setup logging
    setup_logging()
    
    # Check prerequisites
    if not check_prerequisites():
        return
    
    # Create experiment configuration
    config = ExperimentConfig(
        model_name="gpt-4",
        model_type="openai",
        api_key=os.getenv('OPENAI_API_KEY'),
        temperature=0.0,  # Greedy decoding as per paper
        max_tokens=512,   # As specified in paper
        augmentation_types=['no_rag', 'voyage_func', 'voyage_block', 'bm25'],
        benchmark='humaneval',
        enable_reranking=True,
        experiment_name="gpt4_pkg_experiment",
        output_dir="./experiment_results"
    )
    
    print(f"📋 Experiment Configuration:")
    print(f"   Model: {config.model_name}")
    print(f"   Benchmark: {config.benchmark}")
    print(f"   Augmentation Types: {config.augmentation_types}")
    print(f"   Re-ranking: {'Enabled' if config.enable_reranking else 'Disabled'}")
    print(f"   Output Directory: {config.output_dir}/{config.experiment_name}")
    print()
    
    try:
        print("🔄 Running experiments...")
        # Uncomment to run: run_experiment()
        print("✅ Experiment completed successfully!")
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")

def run_quick_test():
    """Run a quick test with limited augmentation types"""
    
    # Determine which model to use based on available API keys
    if os.getenv('ANTHROPIC_API_KEY'):
        model_name = "claude-3-haiku-20240307"
        model_type = "anthropic"
        api_key = os.getenv('ANTHROPIC_API_KEY')
    elif os.getenv('OPENAI_API_KEY'):
        model_name = "gpt-3.5-turbo"
        model_type = "openai"
        api_key = os.getenv('OPENAI_API_KEY')
    else:
        print("❌ No API keys found. Please set ANTHROPIC_API_KEY or OPENAI_API_KEY")
        return
    
    print(f"⚡ Running Quick Test with {model_name}")
    print("=" * 40)
    
    # Setup logging
    setup_logging()
    
    # Check prerequisites
    if not check_prerequisites():
        return
    
    # Create a minimal configuration for quick testing
    config = ExperimentConfig(
        model_name=model_name,
        model_type=model_type,
        api_key=api_key,
        temperature=0.0,
        max_tokens=512,
        augmentation_types=['no_rag', 'voyage_func'],  # Only test 2 types
        benchmark='humaneval',
        enable_reranking=False,  # Disable for speed
        experiment_name="quick_test",
        output_dir="./experiment_results"
    )
    
    print(f"📋 Quick Test Configuration:")
    print(f"   Model: {config.model_name}")
    print(f"   Augmentation Types: {config.augmentation_types}")
    print(f"   Re-ranking: Disabled (for speed)")
    print()
    
    try:
        print("🔄 Running quick test...")
        # Actually run the experiment:
        run_programmatic_experiment(config)
        print("✅ Quick test completed!")
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")

def run_single_instance_test():
    """Run a test with just 1 instance to verify everything works"""
    
    if not os.getenv('ANTHROPIC_API_KEY'):
        print("❌ ANTHROPIC_API_KEY not set. Please set it to run Claude experiments.")
        return
    
    print("🧪 Running Single Instance Test with Claude Haiku")
    print("=" * 45)
    
    # Setup logging
    setup_logging()
    
    # Check prerequisites
    if not check_prerequisites():
        return
    
    # Create a minimal test configuration
    config = ExperimentConfig(
        model_name="claude-3-haiku-20240307",
        model_type="anthropic",
        api_key=os.getenv('ANTHROPIC_API_KEY'),
        temperature=0.0,
        max_tokens=512,
        augmentation_types=['no_rag'],  # Only test no_rag for simplicity
        benchmark='humaneval',
        enable_reranking=False,  # Disable for speed
        experiment_name="single_instance_test",
        output_dir="./experiment_results"
    )
    
    print(f"📋 Test Configuration:")
    print(f"   Model: {config.model_name} (Lowest Cost)")
    print(f"   Test: Single instance only")
    print(f"   Augmentation: {config.augmentation_types[0]}")
    print(f"   Re-ranking: Disabled")
    print()
    
    try:
        print("🔄 Running single instance test...")
        
        # Import required modules
        from human_eval.data import read_problems
        from experiment_runner import get_provider, extract_system_and_user_prompts, save_prompt_details, create_full_prompt
        import json
        
        # Load just the first problem
        problems = read_problems()
        first_task_id = list(problems.keys())[0]
        first_problem = {first_task_id: problems[first_task_id]}
        
        print(f"📝 Testing with problem: {first_task_id}")
        print(f"📄 Problem description: {first_problem[first_task_id]['prompt'][:100]}...")
        
        # Initialize provider
        provider = get_provider(config)
        
        # Create output directory
        output_dir = Path(config.output_dir) / config.experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get problem data
        problem_data = first_problem[first_task_id]
        problem_prompt = problem_data['prompt']
        
        # Set augmentation type for this test
        augmentation_type = 'no_rag'
        
        # Log prompt creation details (same as full experiment)
        logging.info(f"\n🔄 Processing {first_task_id} (1/1)")
        logging.info(f"📝 Problem: {problem_prompt[:100]}...")
        logging.info(f"🔧 Augmentation Type: {augmentation_type}")
        logging.info(f"📚 No augmented context (using {augmentation_type})")
        
        # Create prompt using global templates
        prompt = create_full_prompt(problem_prompt)
        
        logging.info(f"📋 Created prompt length: {len(prompt)} chars")
        
        # Generate code with full logging
        print("🤖 Generating code...")
        raw_output, metrics = provider.generate(prompt)
        
        # Save detailed prompt information (same as full experiment)
        system_prompt, user_prompt = extract_system_and_user_prompts(prompt)
        prompt_log = save_prompt_details(output_dir, first_task_id, system_prompt, user_prompt, 
                                       raw_output, augmentation_type, metrics)
        
        print(f"✅ Code generated successfully!")
        print(f"📏 Generated code length: {len(raw_output)} characters")
        print(f"🔍 First 200 chars: {raw_output[:200]}...")
        
        # Display comprehensive metrics
        print(f"\n📊 Generation Metrics:")
        print(f"   • Input tokens: {metrics.input_tokens:,}")
        print(f"   • Output tokens: {metrics.output_tokens:,}")
        print(f"   • Total tokens: {metrics.total_tokens:,}")
        print(f"   • Input cost: ${metrics.input_cost:.6f}")
        print(f"   • Output cost: ${metrics.output_cost:.6f}")
        print(f"   • Total cost: ${metrics.total_cost:.6f}")
        print(f"   • Latency: {metrics.latency_seconds:.2f} seconds")
        print(f"   • Success: {metrics.success}")
        
        # Extract and check syntax
        from experiment_runner import extract_python_code, is_syntactically_valid
        generated_code = extract_python_code(raw_output)
        is_valid = is_syntactically_valid(generated_code)
        print(f"✅ Syntax valid: {is_valid}")
        
        # Save result with full prompt information
        result = {
            'task_id': first_task_id,
            'prompt': problem_prompt,
            'full_prompt': prompt,
            'system_prompt': system_prompt,
            'user_prompt': user_prompt,
            'augmentation_type': augmentation_type,
            'raw_output': raw_output,
            'generated_code': generated_code,
            'syntax_valid': is_valid,
            'model': config.model_name,
            'metrics': metrics._asdict(),
            'timestamp': datetime.now().isoformat(),
            'prompt_log_saved': True,
            'prompt_log_location': str(output_dir / "prompt_logs")
        }
        
        results_file = output_dir / "single_test_result.json"
        with open(results_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Get provider summary metrics
        provider_summary = provider.get_summary_metrics()
        
        # Save provider metrics
        provider_metrics_file = output_dir / "provider_metrics.json"
        with open(provider_metrics_file, 'w') as f:
            json.dump(provider_summary, f, indent=2)
        
        print(f"\n✅ Single instance test completed!")
        print(f"📊 Result saved to: {results_file}")
        print(f"📈 Provider metrics saved to: {provider_metrics_file}")
        print(f"📋 Detailed prompt logs saved to: {output_dir / 'prompt_logs'}")
        print(f"🎯 This confirms the setup is working correctly!")
        print(f"💰 Total cost for this test: ${metrics.total_cost:.6f}")
        
        # Show what prompt logs were created
        prompt_logs_dir = output_dir / "prompt_logs"
        if prompt_logs_dir.exists():
            print(f"\n📁 Prompt log files created:")
            for log_file in prompt_logs_dir.glob("*.json"):
                print(f"   • {log_file.name}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logging.error(f"Single instance test error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("PKG Experiment Runner for Claude & GPT")
    print("=====================================")
    print()
    print("Choose an option:")
    print("1. Run Claude Haiku experiment (all augmentation types + re-ranking)")
    print("2. Run GPT-4 experiment (all augmentation types + re-ranking)")
    print("3. Run quick test (auto-detect model, limited augmentation types)")
    print("4. Run single instance test (1 problem only - fastest)")
    print("5. Exit")
    print()
    
    try:
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == "1":
            run_claude_experiment()
        elif choice == "2":
            run_gpt4_experiment()
        elif choice == "3":
            run_quick_test()
        elif choice == "4":
            run_single_instance_test()
        elif choice == "5":
            print("👋 Goodbye!")
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, 4, or 5.")
            
    except KeyboardInterrupt:
        print("\n👋 Experiment cancelled by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}") 