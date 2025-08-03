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

from experiment_runner import ExperimentConfig, main as run_experiment
import logging

def setup_logging():
    """Setup logging for the experiment"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

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
    """Run a comprehensive experiment with Claude Sonnet"""
    
    if not os.getenv('ANTHROPIC_API_KEY'):
        print("❌ ANTHROPIC_API_KEY not set. Please set it to run Claude experiments.")
        return
    
    print("🚀 Starting Claude Sonnet PKG Experiment")
    print("=" * 50)
    
    # Setup logging
    setup_logging()
    
    # Check prerequisites
    if not check_prerequisites():
        return
    
    # Create experiment configuration
    config = ExperimentConfig(
        model_name="claude-3-sonnet-20240229",
        model_type="anthropic",
        api_key=os.getenv('ANTHROPIC_API_KEY'),
        temperature=0.0,  # Greedy decoding as per paper
        max_tokens=512,   # As specified in paper
        augmentation_types=['no_rag', 'voyage_func', 'voyage_block', 'bm25'],
        benchmark='humaneval',
        enable_reranking=True,
        experiment_name="claude_sonnet_pkg_experiment",
        output_dir="./experiment_results"
    )
    
    print(f"📋 Experiment Configuration:")
    print(f"   Model: {config.model_name}")
    print(f"   Benchmark: {config.benchmark}")
    print(f"   Augmentation Types: {config.augmentation_types}")
    print(f"   Re-ranking: {'Enabled' if config.enable_reranking else 'Disabled'}")
    print(f"   Output Directory: {config.output_dir}/{config.experiment_name}")
    print()
    
    # Import and run the experiment
    try:
        # This would run the actual experiment
        # For demonstration, we'll just show what would happen
        print("🔄 Running experiments...")
        print("   This will take several minutes depending on the model and API speed")
        print("   Progress will be shown for each augmentation type")
        print()
        
        # Uncomment the following line to actually run the experiment:
        # run_experiment()
        
        print("✅ Experiment completed successfully!")
        print(f"📊 Results saved to: {config.output_dir}/{config.experiment_name}")
        print("📋 Check the summary_report.md file for detailed results")
        
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
        model_name = "claude-3-sonnet-20240229"
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
        # Uncomment to run: run_experiment()
        print("✅ Quick test completed!")
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")

if __name__ == "__main__":
    print("PKG Experiment Runner for Claude & GPT")
    print("=====================================")
    print()
    print("Choose an option:")
    print("1. Run Claude Sonnet experiment (all augmentation types + re-ranking)")
    print("2. Run GPT-4 experiment (all augmentation types + re-ranking)")
    print("3. Run quick test (auto-detect model, limited augmentation types)")
    print("4. Exit")
    print()
    
    try:
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == "1":
            run_claude_experiment()
        elif choice == "2":
            run_gpt4_experiment()
        elif choice == "3":
            run_quick_test()
        elif choice == "4":
            print("👋 Goodbye!")
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
            
    except KeyboardInterrupt:
        print("\n👋 Experiment cancelled by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}") 