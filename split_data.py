#!/usr/bin/env python3
"""
Data splitting script for OpenAI fine-tuning data.

This script splits a JSONL file containing conversation data into train, 
evaluation, and test sets for fine-tuning purposes.
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple


def load_jsonl_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries containing the conversation data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    data = []
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    print(f"Loading data from {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON on line {line_num}: {e}")
                continue
    
    print(f"Loaded {len(data)} conversations")
    return data


def validate_data(data: List[Dict[str, Any]]) -> None:
    """
    Validate that the data has the expected structure for OpenAI fine-tuning.
    
    Args:
        data: List of conversation dictionaries
        
    Raises:
        ValueError: If data structure is invalid
    """
    if not data:
        raise ValueError("No data found in file")
    
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item {i} is not a dictionary")
        
        if 'messages' not in item:
            raise ValueError(f"Item {i} missing 'messages' key")
        
        messages = item['messages']
        if not isinstance(messages, list):
            raise ValueError(f"Item {i}: 'messages' is not a list")
        
        if not messages:
            raise ValueError(f"Item {i}: 'messages' list is empty")
        
        # Check that messages have required structure
        for j, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError(f"Item {i}, message {j}: not a dictionary")
            
            if 'role' not in message or 'content' not in message:
                raise ValueError(f"Item {i}, message {j}: missing 'role' or 'content'")
            
            if message['role'] not in ['user', 'assistant', 'system']:
                raise ValueError(f"Item {i}, message {j}: invalid role '{message['role']}'")


def split_data(data: List[Dict[str, Any]], 
               train_ratio: float = 0.8, 
               eval_ratio: float = 0.1, 
               test_ratio: float = 0.1,
               random_seed: int = 42) -> Tuple[List[Dict[str, Any]], 
                                               List[Dict[str, Any]], 
                                               List[Dict[str, Any]]]:
    """
    Split data into train, evaluation, and test sets.
    
    Args:
        data: List of conversation dictionaries
        train_ratio: Proportion of data for training (default: 0.8)
        eval_ratio: Proportion of data for evaluation (default: 0.1)
        test_ratio: Proportion of data for testing (default: 0.1)
        random_seed: Random seed for reproducible splits (default: 42)
        
    Returns:
        Tuple of (train_data, eval_data, test_data)
        
    Raises:
        ValueError: If ratios don't sum to 1.0 or are invalid
    """
    if abs(train_ratio + eval_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")
    
    if any(ratio < 0 or ratio > 1 for ratio in [train_ratio, eval_ratio, test_ratio]):
        raise ValueError("All ratios must be between 0 and 1")
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Shuffle the data
    data_copy = data.copy()
    random.shuffle(data_copy)
    
    total_size = len(data_copy)
    train_size = int(total_size * train_ratio)
    eval_size = int(total_size * eval_ratio)
    
    # Split the data
    train_data = data_copy[:train_size]
    eval_data = data_copy[train_size:train_size + eval_size]
    test_data = data_copy[train_size + eval_size:]
    
    return train_data, eval_data, test_data


def save_jsonl_data(data: List[Dict[str, Any]], file_path: str) -> None:
    """
    Save data to a JSONL file.
    
    Args:
        data: List of conversation dictionaries
        file_path: Path where to save the file
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(data)} conversations to {file_path}")


def main():
    """Main function to handle command line arguments and execute the split."""
    parser = argparse.ArgumentParser(
        description="Split JSONL data for OpenAI fine-tuning into train/eval/test sets"
    )
    parser.add_argument(
        "input_file",
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "--output-dir",
        default="./data",
        help="Directory to save output files (default: ./data)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Proportion of data for training (default: 0.8)"
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.1,
        help="Proportion of data for evaluation (default: 0.1)"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Proportion of data for testing (default: 0.1)"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42)"
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Prefix for output filenames (default: empty)"
    )
    
    args = parser.parse_args()
    
    try:
        # Load and validate data
        data = load_jsonl_data(args.input_file)
        validate_data(data)
        
        # Split the data
        print(f"Splitting data with ratios: train={args.train_ratio}, eval={args.eval_ratio}, test={args.test_ratio}")
        train_data, eval_data, test_data = split_data(
            data, 
            args.train_ratio, 
            args.eval_ratio, 
            args.test_ratio,
            args.random_seed
        )
        
        # Prepare output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output filenames
        prefix = f"{args.prefix}_" if args.prefix else ""
        train_file = output_dir / f"{prefix}train.jsonl"
        eval_file = output_dir / f"{prefix}eval.jsonl"
        test_file = output_dir / f"{prefix}test.jsonl"
        
        # Save the split data
        save_jsonl_data(train_data, train_file)
        save_jsonl_data(eval_data, eval_file)
        save_jsonl_data(test_data, test_file)
        
        # Print summary
        print("\n" + "="*50)
        print("SPLIT SUMMARY")
        print("="*50)
        print(f"Total conversations: {len(data)}")
        print(f"Train set: {len(train_data)} conversations ({len(train_data)/len(data)*100:.1f}%)")
        print(f"Eval set:  {len(eval_data)} conversations ({len(eval_data)/len(data)*100:.1f}%)")
        print(f"Test set:  {len(test_data)} conversations ({len(test_data)/len(data)*100:.1f}%)")
        print(f"\nOutput files:")
        print(f"  Train: {train_file}")
        print(f"  Eval:  {eval_file}")
        print(f"  Test:  {test_file}")
        print("="*50)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
