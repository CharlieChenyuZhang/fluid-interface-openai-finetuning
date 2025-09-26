# Chat Fine-Tuning Analysis Tool

A Python script for analyzing chat datasets used for OpenAI fine-tuning. This tool validates data format, provides statistics, and estimates token counts for fine-tuning costs.

## Features

- **Format Validation**: Checks for common format errors in chat datasets
- **Statistical Analysis**: Provides distribution statistics for messages and tokens
- **Cost Estimation**: Estimates fine-tuning costs based on token usage
- **Data Warnings**: Identifies potential issues like missing system/user messages

## Installation

### Using Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Direct Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Analysis

```bash
# Make sure virtual environment is activated
source venv/bin/activate  # On Windows: venv\Scripts\activate

python chat_fine_tuning_analysis.py data/toy_chat_fine_tuning.jsonl
```

### Show First Example

```bash
python chat_fine_tuning_analysis.py data/toy_chat_fine_tuning.jsonl --show-example
```

### Help

```bash
python chat_fine_tuning_analysis.py --help
```

## Dataset Format

The script expects a JSONL file where each line contains a JSON object with a `messages` array. Each message should have:

- `role`: "system", "user", "assistant", or "function"
- `content`: The message content (string)

Example:

```json
{
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "What is the capital of France?" },
    { "role": "assistant", "content": "The capital of France is Paris." }
  ]
}
```

## Output

The script provides:

1. **Format Validation**: Lists any format errors found
2. **Data Analysis**: Statistics on message counts, token distributions
3. **Cost Estimation**: Estimated tokens and epochs for fine-tuning

## Requirements

- Python 3.7+
- tiktoken
- numpy

## License

This project is based on the OpenAI Cookbook examples for fine-tuning analysis.
