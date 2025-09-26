#!/usr/bin/env python3
"""
Simple script to run fine-tuning with your data files.

This script provides a simple interface to run fine-tuning using the
data/train.jsonl and data/eval.jsonl files in your project.
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path to import our fine_tune_models module
sys.path.insert(0, str(Path(__file__).parent))

from fine_tune_models import FineTuningManager, load_config


def main():
    """Run fine-tuning with the default data files."""
    
    # Check if API key is set
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return 1
    
    # Check if data files exist
    train_file = "data/train.jsonl"
    eval_file = "data/eval.jsonl"
    
    if not os.path.exists(train_file):
        print(f"❌ Error: Training file not found: {train_file}")
        return 1
    
    if not os.path.exists(eval_file):
        print(f"⚠️  Warning: Evaluation file not found: {eval_file}")
        print("Proceeding with training data only...")
        eval_file = None
    
    # Load configuration
    config = load_config("config.json")
    
    # Get model from config or use default
    model = config.get('models', {}).get('gpt-4.1-nano', 'gpt-4.1-nano-2025-04-14')
    
    # Get hyperparameters from config
    hyperparameters = config.get('hyperparameters', {}).get('default', {
        "n_epochs": 10,
        "batch_size": 1,
        "learning_rate_multiplier": 1.0
    })
    
    # Get test prompts from config
    test_prompts = config.get('test_prompts', [
        "What is the situation you engaged in or avoided?",
        "How could you have focused on different elements of the situation?",
        "What could you have done differently?"
    ])
    
    print("🚀 Starting OpenAI Fine-tuning Process")
    print("=" * 50)
    print(f"Training file: {train_file}")
    print(f"Evaluation file: {eval_file or 'None'}")
    print(f"Base model: {model}")
    print(f"Hyperparameters: {hyperparameters}")
    print("=" * 50)
    
    try:
        # Initialize fine-tuning manager
        manager = FineTuningManager()
        
        # Run complete fine-tuning process
        results = manager.run_complete_fine_tuning(
            train_file=train_file,
            eval_file=eval_file,
            model=model,
            hyperparameters=hyperparameters,
            test_prompts=test_prompts,
            suffix="reflection-coach"
        )
        
        # Print results
        print("\n" + "=" * 50)
        print("FINE-TUNING RESULTS")
        print("=" * 50)
        
        if results['success']:
            print("✅ Fine-tuning completed successfully!")
            print(f"Fine-tuned model ID: {results['fine_tuned_model']}")
            print(f"Job ID: {results['job_id']}")
            
            if results['test_results']:
                print(f"\nModel testing results:")
                print(f"Successful tests: {results['test_results']['success_count']}")
                print(f"Failed tests: {results['test_results']['error_count']}")
                
                print("\nSample test results:")
                for i, result in enumerate(results['test_results']['test_results'][:3], 1):
                    if result['success']:
                        print(f"\nTest {i}:")
                        print(f"Prompt: {result['prompt'][:100]}...")
                        print(f"Response: {result['response'][:200]}...")
            
            print(f"\n🎉 Your fine-tuned model is ready to use!")
            print(f"Model ID: {results['fine_tuned_model']}")
            print("\nYou can now use this model in your applications or test it further.")
            
        else:
            print("❌ Fine-tuning failed!")
            print("Errors:")
            for error in results['errors']:
                print(f"  - {error}")
            
            return 1
        
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
