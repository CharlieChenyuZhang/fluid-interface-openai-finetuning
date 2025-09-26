#!/usr/bin/env python3
"""
OpenAI Fine-tuning Script

This script provides a comprehensive solution for fine-tuning OpenAI models using
supervised fine-tuning with your training and evaluation data.

Features:
- Data validation and format checking
- File upload to OpenAI
- Fine-tuning job creation and monitoring
- Model evaluation and testing
- Comprehensive error handling and logging

Usage:
    python fine_tune_models.py --help
    python fine_tune_models.py --train-file data/train.jsonl --eval-file data/eval.jsonl --model gpt-4.1-nano
"""

import os
import json
import time
import argparse
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

import openai
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fine_tuning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FineTuningManager:
    """Manages the complete fine-tuning process for OpenAI models."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the fine-tuning manager.
        
        Args:
            api_key: OpenAI API key. If None, will use OPENAI_API_KEY environment variable.
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.uploaded_files = {}
        self.fine_tuning_jobs = {}
        
    def validate_jsonl_file(self, file_path: str) -> Dict[str, Any]:
        """Validate a JSONL file for fine-tuning.
        
        Args:
            file_path: Path to the JSONL file
            
        Returns:
            Dictionary with validation results and statistics
        """
        logger.info(f"Validating JSONL file: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        validation_results = {
            'valid': True,
            'total_lines': 0,
            'valid_lines': 0,
            'errors': [],
            'sample_data': None
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    validation_results['total_lines'] += 1
                    
                    if not line.strip():
                        validation_results['errors'].append(f"Line {line_num}: Empty line")
                        continue
                    
                    try:
                        data = json.loads(line.strip())
                        
                        # Validate required structure
                        if 'messages' not in data:
                            validation_results['errors'].append(f"Line {line_num}: Missing 'messages' field")
                            continue
                        
                        messages = data['messages']
                        if not isinstance(messages, list) or len(messages) < 2:
                            validation_results['errors'].append(f"Line {line_num}: 'messages' must be a list with at least 2 items")
                            continue
                        
                        # Validate message structure
                        for i, message in enumerate(messages):
                            if not isinstance(message, dict):
                                validation_results['errors'].append(f"Line {line_num}, message {i}: Message must be a dictionary")
                                continue
                            
                            if 'role' not in message or 'content' not in message:
                                validation_results['errors'].append(f"Line {line_num}, message {i}: Missing 'role' or 'content' field")
                                continue
                            
                            if message['role'] not in ['user', 'assistant', 'system']:
                                validation_results['errors'].append(f"Line {line_num}, message {i}: Invalid role '{message['role']}'")
                                continue
                        
                        validation_results['valid_lines'] += 1
                        
                        # Store sample data from first valid line
                        if validation_results['sample_data'] is None:
                            validation_results['sample_data'] = data
                            
                    except json.JSONDecodeError as e:
                        validation_results['errors'].append(f"Line {line_num}: Invalid JSON - {str(e)}")
                        continue
            
            # Check minimum requirements
            if validation_results['valid_lines'] < 10:
                validation_results['errors'].append(f"File must have at least 10 valid training examples, found {validation_results['valid_lines']}")
                validation_results['valid'] = False
            
            if validation_results['errors']:
                validation_results['valid'] = False
            
            logger.info(f"Validation complete: {validation_results['valid_lines']}/{validation_results['total_lines']} valid lines")
            
        except Exception as e:
            validation_results['valid'] = False
            validation_results['errors'].append(f"File reading error: {str(e)}")
            logger.error(f"Error validating file {file_path}: {str(e)}")
        
        return validation_results
    
    def upload_file(self, file_path: str, purpose: str = "fine-tune") -> str:
        """Upload a file to OpenAI.
        
        Args:
            file_path: Path to the file to upload
            purpose: Purpose of the file (default: "fine-tune")
            
        Returns:
            File ID from OpenAI
        """
        logger.info(f"Uploading file: {file_path}")
        
        try:
            with open(file_path, 'rb') as f:
                response = self.client.files.create(
                    file=f,
                    purpose=purpose
                )
            
            file_id = response.id
            self.uploaded_files[file_path] = file_id
            
            logger.info(f"File uploaded successfully. ID: {file_id}")
            logger.info(f"File details: {response.filename}, {response.bytes} bytes")
            
            return file_id
            
        except Exception as e:
            logger.error(f"Error uploading file {file_path}: {str(e)}")
            raise
    
    def create_fine_tuning_job(
        self,
        training_file_id: str,
        model: str = "gpt-4.1-nano-2025-04-14",
        validation_file_id: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        suffix: Optional[str] = None
    ) -> str:
        """Create a fine-tuning job.
        
        Args:
            training_file_id: ID of the training file
            model: Base model to fine-tune
            validation_file_id: ID of the validation file (optional)
            hyperparameters: Hyperparameters for fine-tuning
            suffix: Suffix for the fine-tuned model name
            
        Returns:
            Fine-tuning job ID
        """
        logger.info(f"Creating fine-tuning job for model: {model}")
        
        # Default hyperparameters
        if hyperparameters is None:
            hyperparameters = {
                "n_epochs": 10,
                "batch_size": 1,
                "learning_rate_multiplier": 1.0
            }
        
        job_params = {
            "training_file": training_file_id,
            "model": model,
            "hyperparameters": hyperparameters
        }
        
        if validation_file_id:
            job_params["validation_file"] = validation_file_id
        
        if suffix:
            job_params["suffix"] = suffix
        
        try:
            response = self.client.fine_tuning.jobs.create(**job_params)
            job_id = response.id
            self.fine_tuning_jobs[job_id] = response
            
            logger.info(f"Fine-tuning job created successfully. ID: {job_id}")
            logger.info(f"Job status: {response.status}")
            
            return job_id
            
        except Exception as e:
            logger.error(f"Error creating fine-tuning job: {str(e)}")
            raise
    
    def monitor_job(self, job_id: str, check_interval: int = 60) -> Dict[str, Any]:
        """Monitor a fine-tuning job until completion.
        
        Args:
            job_id: ID of the fine-tuning job
            check_interval: Seconds between status checks
            
        Returns:
            Final job status and details
        """
        logger.info(f"Monitoring fine-tuning job: {job_id}")
        
        while True:
            try:
                job = self.client.fine_tuning.jobs.retrieve(job_id)
                self.fine_tuning_jobs[job_id] = job
                
                logger.info(f"Job status: {job.status}")
                
                if job.status in ['succeeded', 'failed', 'cancelled']:
                    logger.info(f"Job completed with status: {job.status}")
                    
                    if job.status == 'succeeded':
                        logger.info(f"Fine-tuned model ID: {job.fine_tuned_model}")
                        logger.info(f"Trained tokens: {job.trained_tokens}")
                    
                    return {
                        'status': job.status,
                        'job': job,
                        'fine_tuned_model': getattr(job, 'fine_tuned_model', None),
                        'trained_tokens': getattr(job, 'trained_tokens', None),
                        'error': getattr(job, 'error', None)
                    }
                
                time.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Error monitoring job {job_id}: {str(e)}")
                raise
    
    def test_model(self, model_id: str, test_prompts: List[str]) -> Dict[str, Any]:
        """Test a fine-tuned model with sample prompts.
        
        Args:
            model_id: ID of the fine-tuned model
            test_prompts: List of test prompts
            
        Returns:
            Test results
        """
        logger.info(f"Testing model: {model_id}")
        
        results = {
            'model_id': model_id,
            'test_results': [],
            'success_count': 0,
            'error_count': 0
        }
        
        for i, prompt in enumerate(test_prompts):
            try:
                response = self.client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.7
                )
                
                result = {
                    'prompt': prompt,
                    'response': response.choices[0].message.content,
                    'success': True,
                    'usage': response.usage
                }
                
                results['test_results'].append(result)
                results['success_count'] += 1
                
                logger.info(f"Test {i+1}/{len(test_prompts)} completed successfully")
                
            except Exception as e:
                result = {
                    'prompt': prompt,
                    'error': str(e),
                    'success': False
                }
                
                results['test_results'].append(result)
                results['error_count'] += 1
                
                logger.error(f"Test {i+1}/{len(test_prompts)} failed: {str(e)}")
        
        logger.info(f"Model testing complete: {results['success_count']} successful, {results['error_count']} failed")
        
        return results
    
    def run_complete_fine_tuning(
        self,
        train_file: str,
        eval_file: Optional[str] = None,
        model: str = "gpt-4.1-nano-2025-04-14",
        hyperparameters: Optional[Dict[str, Any]] = None,
        test_prompts: Optional[List[str]] = None,
        suffix: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run the complete fine-tuning process.
        
        Args:
            train_file: Path to training data file
            eval_file: Path to evaluation data file (optional)
            model: Base model to fine-tune
            hyperparameters: Hyperparameters for fine-tuning
            test_prompts: Prompts to test the fine-tuned model
            suffix: Suffix for the fine-tuned model name
            
        Returns:
            Complete fine-tuning results
        """
        logger.info("Starting complete fine-tuning process")
        
        results = {
            'success': False,
            'training_file_validation': None,
            'eval_file_validation': None,
            'training_file_id': None,
            'eval_file_id': None,
            'job_id': None,
            'fine_tuned_model': None,
            'test_results': None,
            'errors': []
        }
        
        try:
            # Step 1: Validate training data
            logger.info("Step 1: Validating training data")
            train_validation = self.validate_jsonl_file(train_file)
            results['training_file_validation'] = train_validation
            
            if not train_validation['valid']:
                results['errors'].extend(train_validation['errors'])
                logger.error("Training data validation failed")
                return results
            
            # Step 2: Validate evaluation data (if provided)
            if eval_file:
                logger.info("Step 2: Validating evaluation data")
                eval_validation = self.validate_jsonl_file(eval_file)
                results['eval_file_validation'] = eval_validation
                
                if not eval_validation['valid']:
                    results['errors'].extend(eval_validation['errors'])
                    logger.error("Evaluation data validation failed")
                    return results
            
            # Step 3: Upload training file
            logger.info("Step 3: Uploading training file")
            training_file_id = self.upload_file(train_file)
            results['training_file_id'] = training_file_id
            
            # Step 4: Upload evaluation file (if provided)
            eval_file_id = None
            if eval_file:
                logger.info("Step 4: Uploading evaluation file")
                eval_file_id = self.upload_file(eval_file)
                results['eval_file_id'] = eval_file_id
            
            # Step 5: Create fine-tuning job
            logger.info("Step 5: Creating fine-tuning job")
            job_id = self.create_fine_tuning_job(
                training_file_id=training_file_id,
                model=model,
                validation_file_id=eval_file_id,
                hyperparameters=hyperparameters,
                suffix=suffix
            )
            results['job_id'] = job_id
            
            # Step 6: Monitor job
            logger.info("Step 6: Monitoring fine-tuning job")
            job_result = self.monitor_job(job_id)
            
            if job_result['status'] == 'succeeded':
                results['fine_tuned_model'] = job_result['fine_tuned_model']
                logger.info(f"Fine-tuning completed successfully! Model ID: {job_result['fine_tuned_model']}")
                
                # Step 7: Test model (if test prompts provided)
                if test_prompts:
                    logger.info("Step 7: Testing fine-tuned model")
                    test_results = self.test_model(job_result['fine_tuned_model'], test_prompts)
                    results['test_results'] = test_results
                
                results['success'] = True
            else:
                results['errors'].append(f"Fine-tuning job failed with status: {job_result['status']}")
                if job_result.get('error'):
                    results['errors'].append(f"Job error: {job_result['error']}")
                logger.error(f"Fine-tuning job failed: {job_result['status']}")
        
        except Exception as e:
            results['errors'].append(f"Unexpected error: {str(e)}")
            logger.error(f"Unexpected error in fine-tuning process: {str(e)}")
        
        return results


def load_config(config_file: str = "config.json") -> Dict[str, Any]:
    """Load configuration from JSON file.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Config file {config_file} not found, using defaults")
        return {}
    except Exception as e:
        logger.error(f"Error loading config file: {str(e)}")
        return {}


def main():
    """Main function to run the fine-tuning process."""
    parser = argparse.ArgumentParser(description="Fine-tune OpenAI models")
    parser.add_argument("--train-file", required=True, help="Path to training data file (JSONL)")
    parser.add_argument("--eval-file", help="Path to evaluation data file (JSONL)")
    parser.add_argument("--model", default="gpt-4.1-nano-2025-04-14", help="Base model to fine-tune")
    parser.add_argument("--config", default="config.json", help="Path to configuration file")
    parser.add_argument("--suffix", help="Suffix for the fine-tuned model name")
    parser.add_argument("--hyperparameters", help="JSON string of hyperparameters")
    parser.add_argument("--test-only", action="store_true", help="Only test an existing fine-tuned model")
    parser.add_argument("--model-id", help="Fine-tuned model ID for testing")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize fine-tuning manager
    try:
        manager = FineTuningManager()
    except ValueError as e:
        logger.error(str(e))
        return 1
    
    # Test only mode
    if args.test_only:
        if not args.model_id:
            logger.error("--model-id is required for test-only mode")
            return 1
        
        test_prompts = config.get('test_prompts', [
            "What is the capital of France?",
            "Tell me a joke.",
            "How do I make a sandwich?"
        ])
        
        logger.info("Running model tests only")
        test_results = manager.test_model(args.model_id, test_prompts)
        
        print("\n" + "="*50)
        print("MODEL TEST RESULTS")
        print("="*50)
        print(f"Model ID: {test_results['model_id']}")
        print(f"Successful tests: {test_results['success_count']}")
        print(f"Failed tests: {test_results['error_count']}")
        print("\nDetailed results:")
        
        for i, result in enumerate(test_results['test_results'], 1):
            print(f"\nTest {i}:")
            print(f"Prompt: {result['prompt']}")
            if result['success']:
                print(f"Response: {result['response']}")
            else:
                print(f"Error: {result['error']}")
        
        return 0
    
    # Prepare hyperparameters
    hyperparameters = None
    if args.hyperparameters:
        try:
            hyperparameters = json.loads(args.hyperparameters)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid hyperparameters JSON: {str(e)}")
            return 1
    else:
        # Use default hyperparameters from config
        hyperparameters = config.get('hyperparameters', {}).get('default')
    
    # Get test prompts
    test_prompts = config.get('test_prompts', [
        "What is the capital of France?",
        "Tell me a joke.",
        "How do I make a sandwich?"
    ])
    
    # Run complete fine-tuning process
    logger.info("Starting fine-tuning process")
    results = manager.run_complete_fine_tuning(
        train_file=args.train_file,
        eval_file=args.eval_file,
        model=args.model,
        hyperparameters=hyperparameters,
        test_prompts=test_prompts,
        suffix=args.suffix
    )
    
    # Print results
    print("\n" + "="*50)
    print("FINE-TUNING RESULTS")
    print("="*50)
    
    if results['success']:
        print("✅ Fine-tuning completed successfully!")
        print(f"Fine-tuned model ID: {results['fine_tuned_model']}")
        print(f"Job ID: {results['job_id']}")
        
        if results['test_results']:
            print(f"\nModel testing results:")
            print(f"Successful tests: {results['test_results']['success_count']}")
            print(f"Failed tests: {results['test_results']['error_count']}")
    else:
        print("❌ Fine-tuning failed!")
        print("Errors:")
        for error in results['errors']:
            print(f"  - {error}")
    
    return 0 if results['success'] else 1


if __name__ == "__main__":
    exit(main())
