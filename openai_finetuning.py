#!/usr/bin/env python3
"""
OpenAI Fine-Tuning Script

A comprehensive Python script for fine-tuning OpenAI models using the API.
This script handles the complete fine-tuning workflow including:
- Data validation and upload
- Fine-tuning job creation and monitoring
- Model evaluation and testing
- Checkpoint management

Usage:
    python openai_finetuning.py --help
    python openai_finetuning.py upload data/toy_chat_fine_tuning.jsonl
    python openai_finetuning.py create --file-id file-xxx --model gpt-4.1-nano-2025-04-14
    python openai_finetuning.py monitor --job-id ftjob-xxx
    python openai_finetuning.py test --model-id ft:gpt-4.1-nano-2025-04-14:openai::xxx
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import requests
from pathlib import Path


class OpenAIFineTuning:
    """OpenAI Fine-Tuning API client"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the fine-tuning client"""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass it directly.")
        
        self.base_url = "https://api.openai.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def upload_file(self, file_path: str, purpose: str = "fine-tune") -> Dict[str, Any]:
        """Upload a file for fine-tuning"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        print(f"Uploading file: {file_path}")
        
        with open(file_path, 'rb') as f:
            files = {
                'file': (file_path.name, f, 'application/json'),
                'purpose': (None, purpose)
            }
            
            response = requests.post(
                f"{self.base_url}/files",
                headers={"Authorization": f"Bearer {self.api_key}"},
                files=files
            )
        
        if response.status_code != 200:
            raise Exception(f"Upload failed: {response.status_code} - {response.text}")
        
        result = response.json()
        print(f"✅ File uploaded successfully!")
        print(f"   File ID: {result['id']}")
        print(f"   Filename: {result['filename']}")
        print(f"   Size: {result['bytes']} bytes")
        print(f"   Status: {result['status']}")
        
        return result
    
    def create_fine_tuning_job(
        self,
        training_file_id: str,
        model: str = "gpt-4.1-nano-2025-04-14",
        validation_file_id: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        suffix: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a fine-tuning job"""
        
        # Default hyperparameters
        default_hyperparameters = {
            "n_epochs": 10,
            "batch_size": 1,
            "learning_rate_multiplier": 1.0
        }
        
        if hyperparameters:
            default_hyperparameters.update(hyperparameters)
        
        payload = {
            "training_file": training_file_id,
            "model": model,
            "hyperparameters": default_hyperparameters
        }
        
        if validation_file_id:
            payload["validation_file"] = validation_file_id
        
        if suffix:
            payload["suffix"] = suffix
        
        print(f"Creating fine-tuning job...")
        print(f"   Model: {model}")
        print(f"   Training file: {training_file_id}")
        print(f"   Hyperparameters: {default_hyperparameters}")
        
        response = requests.post(
            f"{self.base_url}/fine_tuning/jobs",
            headers=self.headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"Job creation failed: {response.status_code} - {response.text}")
        
        result = response.json()
        print(f"✅ Fine-tuning job created successfully!")
        print(f"   Job ID: {result['id']}")
        print(f"   Status: {result['status']}")
        print(f"   Model: {result['model']}")
        
        return result
    
    def get_fine_tuning_job(self, job_id: str) -> Dict[str, Any]:
        """Get fine-tuning job details"""
        response = requests.get(
            f"{self.base_url}/fine_tuning/jobs/{job_id}",
            headers=self.headers
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to get job: {response.status_code} - {response.text}")
        
        return response.json()
    
    def list_fine_tuning_jobs(self, limit: int = 20) -> Dict[str, Any]:
        """List fine-tuning jobs"""
        response = requests.get(
            f"{self.base_url}/fine_tuning/jobs?limit={limit}",
            headers=self.headers
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to list jobs: {response.status_code} - {response.text}")
        
        return response.json()
    
    def get_fine_tuning_events(self, job_id: str) -> Dict[str, Any]:
        """Get fine-tuning job events"""
        response = requests.get(
            f"{self.base_url}/fine_tuning/jobs/{job_id}/events",
            headers=self.headers
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to get events: {response.status_code} - {response.text}")
        
        return response.json()
    
    def get_checkpoints(self, job_id: str) -> Dict[str, Any]:
        """Get fine-tuning job checkpoints"""
        response = requests.get(
            f"{self.base_url}/fine_tuning/jobs/{job_id}/checkpoints",
            headers=self.headers
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to get checkpoints: {response.status_code} - {response.text}")
        
        return response.json()
    
    def test_model(self, model_id: str, test_prompts: List[str]) -> List[Dict[str, Any]]:
        """Test a fine-tuned model with sample prompts"""
        results = []
        
        for prompt in test_prompts:
            print(f"Testing prompt: {prompt[:50]}...")
            
            payload = {
                "model": model_id,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 150,
                "temperature": 0.7
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                results.append({
                    "prompt": prompt,
                    "response": result["choices"][0]["message"]["content"],
                    "usage": result.get("usage", {})
                })
                print(f"✅ Response: {result['choices'][0]['message']['content'][:100]}...")
            else:
                print(f"❌ Error: {response.status_code} - {response.text}")
                results.append({
                    "prompt": prompt,
                    "error": f"{response.status_code} - {response.text}"
                })
        
        return results
    
    def monitor_job(self, job_id: str, poll_interval: int = 30) -> Dict[str, Any]:
        """Monitor a fine-tuning job until completion"""
        print(f"Monitoring job: {job_id}")
        print(f"Polling every {poll_interval} seconds...")
        
        while True:
            job = self.get_fine_tuning_job(job_id)
            status = job["status"]
            
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Status: {status}")
            
            if status == "succeeded":
                print("✅ Fine-tuning completed successfully!")
                print(f"   Fine-tuned model: {job.get('fine_tuned_model', 'N/A')}")
                print(f"   Trained tokens: {job.get('trained_tokens', 'N/A')}")
                print(f"   Training time: {job.get('finished_at', 0) - job.get('created_at', 0)} seconds")
                break
            elif status == "failed":
                print("❌ Fine-tuning failed!")
                error = job.get("error", {})
                print(f"   Error: {error}")
                break
            elif status in ["validating_files", "queued", "running"]:
                print(f"   Progress: {status}")
                if "estimated_finish" in job and job["estimated_finish"]:
                    eta = datetime.fromtimestamp(job["estimated_finish"])
                    print(f"   Estimated completion: {eta.strftime('%H:%M:%S')}")
            else:
                print(f"   Unknown status: {status}")
            
            time.sleep(poll_interval)
        
        return job


def validate_jsonl_file(file_path: str) -> bool:
    """Validate JSONL file format"""
    print(f"Validating file: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if len(lines) < 10:
            print("⚠️  Warning: File has fewer than 10 lines (minimum recommended)")
        
        valid_count = 0
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                if "messages" not in data:
                    print(f"❌ Line {i}: Missing 'messages' field")
                    return False
                
                messages = data["messages"]
                if not isinstance(messages, list) or len(messages) == 0:
                    print(f"❌ Line {i}: 'messages' must be a non-empty array")
                    return False
                
                for j, msg in enumerate(messages):
                    if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                        print(f"❌ Line {i}, Message {j}: Missing 'role' or 'content'")
                        return False
                    
                    if msg["role"] not in ["system", "user", "assistant", "function"]:
                        print(f"❌ Line {i}, Message {j}: Invalid role '{msg['role']}'")
                        return False
                
                valid_count += 1
                
            except json.JSONDecodeError as e:
                print(f"❌ Line {i}: Invalid JSON - {e}")
                return False
        
        print(f"✅ File validation passed! ({valid_count} valid examples)")
        return True
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="OpenAI Fine-Tuning Script")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Upload command
    upload_parser = subparsers.add_parser("upload", help="Upload training data file")
    upload_parser.add_argument("file_path", help="Path to JSONL training file")
    upload_parser.add_argument("--purpose", default="fine-tune", help="File purpose (default: fine-tune)")
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Create fine-tuning job")
    create_parser.add_argument("--file-id", required=True, help="Training file ID from upload")
    create_parser.add_argument("--model", default="gpt-4.1-nano-2025-04-14", help="Base model to fine-tune")
    create_parser.add_argument("--validation-file-id", help="Validation file ID (optional)")
    create_parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    create_parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    create_parser.add_argument("--learning-rate", type=float, default=1.0, help="Learning rate multiplier")
    create_parser.add_argument("--suffix", help="Suffix for fine-tuned model name")
    
    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Monitor fine-tuning job")
    monitor_parser.add_argument("--job-id", required=True, help="Fine-tuning job ID")
    monitor_parser.add_argument("--poll-interval", type=int, default=30, help="Polling interval in seconds")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Get job status")
    status_parser.add_argument("--job-id", required=True, help="Fine-tuning job ID")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List fine-tuning jobs")
    list_parser.add_argument("--limit", type=int, default=20, help="Number of jobs to list")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test fine-tuned model")
    test_parser.add_argument("--model-id", required=True, help="Fine-tuned model ID")
    test_parser.add_argument("--prompts", nargs="+", help="Test prompts")
    test_parser.add_argument("--prompt-file", help="File containing test prompts (one per line)")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate JSONL file")
    validate_parser.add_argument("file_path", help="Path to JSONL file to validate")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        client = OpenAIFineTuning()
        
        if args.command == "upload":
            if not validate_jsonl_file(args.file_path):
                print("❌ File validation failed. Please fix the issues and try again.")
                return
            
            result = client.upload_file(args.file_path, args.purpose)
            print(f"\n📋 Next steps:")
            print(f"1. Create fine-tuning job: python {sys.argv[0]} create --file-id {result['id']}")
            print(f"2. Monitor job: python {sys.argv[0]} monitor --job-id <job-id>")
        
        elif args.command == "create":
            hyperparameters = {
                "n_epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate_multiplier": args.learning_rate
            }
            
            result = client.create_fine_tuning_job(
                training_file_id=args.file_id,
                model=args.model,
                validation_file_id=args.validation_file_id,
                hyperparameters=hyperparameters,
                suffix=args.suffix
            )
            
            print(f"\n📋 Next steps:")
            print(f"1. Monitor job: python {sys.argv[0]} monitor --job-id {result['id']}")
            print(f"2. Check status: python {sys.argv[0]} status --job-id {result['id']}")
        
        elif args.command == "monitor":
            client.monitor_job(args.job_id, args.poll_interval)
        
        elif args.command == "status":
            job = client.get_fine_tuning_job(args.job_id)
            print(f"\n📊 Job Status:")
            print(f"   ID: {job['id']}")
            print(f"   Status: {job['status']}")
            print(f"   Model: {job['model']}")
            print(f"   Created: {datetime.fromtimestamp(job['created_at']).strftime('%Y-%m-%d %H:%M:%S')}")
            
            if job.get('finished_at'):
                print(f"   Finished: {datetime.fromtimestamp(job['finished_at']).strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"   Duration: {job['finished_at'] - job['created_at']} seconds")
            
            if job.get('fine_tuned_model'):
                print(f"   Fine-tuned model: {job['fine_tuned_model']}")
            
            if job.get('trained_tokens'):
                print(f"   Trained tokens: {job['trained_tokens']}")
        
        elif args.command == "list":
            result = client.list_fine_tuning_jobs(args.limit)
            jobs = result.get('data', [])
            
            print(f"\n📋 Fine-tuning Jobs (showing {len(jobs)} of {result.get('total_count', 'unknown')}):")
            for job in jobs:
                status_emoji = "✅" if job['status'] == "succeeded" else "❌" if job['status'] == "failed" else "⏳"
                print(f"   {status_emoji} {job['id']} - {job['status']} - {job['model']}")
                if job.get('fine_tuned_model'):
                    print(f"      Model: {job['fine_tuned_model']}")
        
        elif args.command == "test":
            prompts = []
            
            if args.prompts:
                prompts.extend(args.prompts)
            
            if args.prompt_file:
                with open(args.prompt_file, 'r') as f:
                    prompts.extend([line.strip() for line in f if line.strip()])
            
            if not prompts:
                # Default test prompts
                prompts = [
                    "What is the capital of France?",
                    "Tell me a joke.",
                    "How do I make a sandwich?",
                    "What is 2+2?",
                    "What is the weather like?"
                ]
            
            print(f"Testing model: {args.model_id}")
            results = client.test_model(args.model_id, prompts)
            
            print(f"\n📊 Test Results:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Prompt: {result['prompt']}")
                if 'error' in result:
                    print(f"   ❌ Error: {result['error']}")
                else:
                    print(f"   ✅ Response: {result['response']}")
                    if 'usage' in result:
                        print(f"   📊 Tokens: {result['usage']}")
        
        elif args.command == "validate":
            if validate_jsonl_file(args.file_path):
                print("✅ File is valid and ready for fine-tuning!")
            else:
                print("❌ File validation failed. Please fix the issues and try again.")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
