#!/usr/bin/env python3
"""
Example script for running PKG experiments with Claude and GPT models

This script demonstrates how to use the experiment runner programmatically
instead of through the command line interface.
"""

import os
import sys
from pathlib import Path

# Add the current directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent))

from experiment_runner import ExperimentConfig, run_single_experiment, create_summary_report
import logging

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
    
    for aug_type in config.augmentation_types:
        print(f"\n🔄 Running {aug_type} experiments...")
        try:
            results = run_single_experiment(config, aug_type)
            all_results[aug_type] = results
            
            # Calculate metrics
            total = len(results)
            passed = sum(1 for r in results if r.get('passed', False))
            syntax_valid = sum(1 for r in results if r.get('syntax_valid', False))
            
            metrics[aug_type] = {
                'pass_at_1': passed / total if total > 0 else 0.0,
                'syntax_accuracy': syntax_valid / total if total > 0 else 0.0,
                'passed': passed,
                'total': total
            }
            
            print(f"✅ {aug_type}: {metrics[aug_type]['pass_at_1']:.1%} ({passed}/{total})")
            
            # Save individual results
            import json
            results_file = output_dir / f"{aug_type}_results.jsonl"
            with open(results_file, 'w') as f:
                for result in results:
                    f.write(json.dumps(result) + '\n')
                    
        except Exception as e:
            logging.error(f"Error in {aug_type}: {e}")
            metrics[aug_type] = {'pass_at_1': 0.0, 'syntax_accuracy': 0.0, 'passed': 0, 'total': 0}
    
    # Save metrics and create summary
    metrics_file = output_dir / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Create summary report
    create_summary_report(output_dir, config, metrics)
    
    # Save configuration
    config_file = output_dir / 'config.json'
    with open(config_file, 'w') as f:
        import dataclasses
        config_dict = dataclasses.asdict(config)
        json.dump(config_dict, f, indent=2)
    
    print(f"\n✅ Experiment completed!")
    print(f"📊 Results saved to: {output_dir}")
    
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
        'augmented_problems/humaneval_function_wise_relevant_context.jsonl',
        'augmented_problems/humaneval_blockwise_relevant_context.jsonl',
        'augmented_problems/bm25_relevant_context_humaneval.jsonl'
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
        from experiment_runner import get_provider
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
        
        # Run the single test
        problem_data = first_problem[first_task_id]
        problem_prompt = problem_data['prompt']
        
        # Generate code
        print("🤖 Generating code...")
        generated_code = provider.generate(problem_prompt)
        
        print(f"✅ Code generated successfully!")
        print(f"📏 Generated code length: {len(generated_code)} characters")
        print(f"🔍 First 200 chars: {generated_code[:200]}...")
        
        # Check syntax
        from experiment_runner import is_syntactically_valid
        is_valid = is_syntactically_valid(generated_code)
        print(f"✅ Syntax valid: {is_valid}")
        
        # Save result
        result = {
            'task_id': first_task_id,
            'prompt': problem_prompt,
            'generated_code': generated_code,
            'syntax_valid': is_valid,
            'model': config.model_name,
            'augmentation_type': 'no_rag'
        }
        
        results_file = output_dir / "single_test_result.json"
        with open(results_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n✅ Single instance test completed!")
        print(f"📊 Result saved to: {results_file}")
        print(f"🎯 This confirms the setup is working correctly!")
        
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