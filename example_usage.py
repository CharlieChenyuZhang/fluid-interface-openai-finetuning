#!/usr/bin/env python3
"""
Example usage of the OpenAI Fine-Tuning script

This script demonstrates how to use the openai_finetuning.py script
for a complete fine-tuning workflow.
"""

import os
import subprocess
import sys
import time
import json
from pathlib import Path


def run_command(cmd, description):
    """Run a command and return the result"""
    print(f"\n🔄 {description}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ Success!")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return None


def main():
    """Example fine-tuning workflow"""
    
    # Check if API key is set
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Please set your OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        return
    
    print("🚀 OpenAI Fine-Tuning Example Workflow")
    print("=" * 50)
    
    # Step 1: Validate the training data
    print("\n📋 Step 1: Validating training data")
    result = run_command([
        sys.executable, "openai_finetuning.py", "validate", 
        "data/toy_chat_fine_tuning.jsonl"
    ], "Validating JSONL file")
    
    if not result:
        print("❌ Validation failed. Please fix the data file and try again.")
        return
    
    # Step 2: Upload the training data
    print("\n📋 Step 2: Uploading training data")
    result = run_command([
        sys.executable, "openai_finetuning.py", "upload",
        "data/toy_chat_fine_tuning.jsonl"
    ], "Uploading training file")
    
    if not result:
        print("❌ Upload failed. Please check your API key and try again.")
        return
    
    # Extract file ID from output (this is a simple approach)
    # In a real scenario, you'd parse the JSON response
    print("\n⚠️  Note: In a real workflow, you would parse the file ID from the upload response")
    print("   and use it in the next step. For this example, you'll need to:")
    print("   1. Copy the file ID from the upload output above")
    print("   2. Run: python openai_finetuning.py create --file-id <your-file-id>")
    print("   3. Then monitor the job with: python openai_finetuning.py monitor --job-id <job-id>")
    
    # Example of what the next steps would look like:
    print("\n📋 Example next steps (replace with actual IDs):")
    print("   # Create fine-tuning job")
    print("   python openai_finetuning.py create --file-id file-xxx --model gpt-4.1-nano-2025-04-14")
    print("   ")
    print("   # Monitor the job")
    print("   python openai_finetuning.py monitor --job-id ftjob-xxx")
    print("   ")
    print("   # Test the fine-tuned model")
    print("   python openai_finetuning.py test --model-id ft:gpt-4.1-nano-2025-04-14:openai::xxx")
    
    # Show available commands
    print("\n📋 Available commands:")
    print("   python openai_finetuning.py --help")
    print("   python openai_finetuning.py upload <file>")
    print("   python openai_finetuning.py create --file-id <id> --model <model>")
    print("   python openai_finetuning.py monitor --job-id <id>")
    print("   python openai_finetuning.py status --job-id <id>")
    print("   python openai_finetuning.py list")
    print("   python openai_finetuning.py test --model-id <id>")
    print("   python openai_finetuning.py validate <file>")


if __name__ == "__main__":
    main()
