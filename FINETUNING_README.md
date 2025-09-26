# OpenAI Fine-Tuning Script

A comprehensive Python script for fine-tuning OpenAI models using the API. This script handles the complete fine-tuning workflow including data validation, upload, job creation, monitoring, and model testing.

## Features

- **Data Validation**: Validates JSONL format and structure
- **File Upload**: Uploads training data to OpenAI
- **Job Management**: Creates and monitors fine-tuning jobs
- **Model Testing**: Tests fine-tuned models with sample prompts
- **Checkpoint Support**: Lists and uses model checkpoints
- **Comprehensive Monitoring**: Real-time job status and progress tracking

## Installation

### Prerequisites

- Python 3.7+
- OpenAI API key

### Setup

1. **Clone or download the script files**

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set your OpenAI API key**:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

## Quick Start

### 1. Validate Your Data

First, validate your JSONL training file:

```bash
python openai_finetuning.py validate data/toy_chat_fine_tuning.jsonl
```

### 2. Upload Training Data

Upload your validated training file:

```bash
python openai_finetuning.py upload data/toy_chat_fine_tuning.jsonl
```

This will return a file ID that you'll need for the next step.

### 3. Create Fine-Tuning Job

Create a fine-tuning job using the uploaded file:

```bash
python openai_finetuning.py create --file-id file-xxx --model gpt-4.1-nano-2025-04-14
```

### 4. Monitor the Job

Monitor your fine-tuning job until completion:

```bash
python openai_finetuning.py monitor --job-id ftjob-xxx
```

### 5. Test Your Model

Test your fine-tuned model:

```bash
python openai_finetuning.py test --model-id ft:gpt-4.1-nano-2025-04-14:openai::xxx
```

## Detailed Usage

### Available Commands

#### `validate` - Validate JSONL file

```bash
python openai_finetuning.py validate <file_path>
```

Validates the format and structure of your JSONL training file.

#### `upload` - Upload training data

```bash
python openai_finetuning.py upload <file_path> [--purpose fine-tune]
```

Uploads your training data file to OpenAI.

#### `create` - Create fine-tuning job

```bash
python openai_finetuning.py create --file-id <file_id> [options]
```

Options:

- `--model`: Base model to fine-tune (default: gpt-4.1-nano-2025-04-14)
- `--validation-file-id`: Optional validation file ID
- `--epochs`: Number of epochs (default: 10)
- `--batch-size`: Batch size (default: 1)
- `--learning-rate`: Learning rate multiplier (default: 1.0)
- `--suffix`: Suffix for fine-tuned model name

#### `monitor` - Monitor fine-tuning job

```bash
python openai_finetuning.py monitor --job-id <job_id> [--poll-interval 30]
```

Monitors a fine-tuning job until completion.

#### `status` - Get job status

```bash
python openai_finetuning.py status --job-id <job_id>
```

Gets the current status of a fine-tuning job.

#### `list` - List fine-tuning jobs

```bash
python openai_finetuning.py list [--limit 20]
```

Lists your fine-tuning jobs.

#### `test` - Test fine-tuned model

```bash
python openai_finetuning.py test --model-id <model_id> [options]
```

Options:

- `--prompts`: Test prompts (space-separated)
- `--prompt-file`: File containing test prompts (one per line)

## Data Format

Your training data should be in JSONL format with the following structure:

```json
{
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "What is the capital of France?" },
    { "role": "assistant", "content": "The capital of France is Paris." }
  ]
}
```

### Supported Roles

- `system`: System instructions
- `user`: User input
- `assistant`: Model response
- `function`: Function call results

### Requirements

- Minimum 10 examples
- Each line must be valid JSON
- Each example must have a `messages` array
- Each message must have `role` and `content` fields

## Configuration

The script includes a `config.json` file with default settings:

```json
{
  "models": {
    "gpt-4.1-nano": "gpt-4.1-nano-2025-04-14",
    "gpt-4.1-mini": "gpt-4.1-mini-2025-04-14",
    "gpt-4.1": "gpt-4.1-2025-04-14"
  },
  "hyperparameters": {
    "default": {
      "n_epochs": 10,
      "batch_size": 1,
      "learning_rate_multiplier": 1.0
    }
  }
}
```

## Example Workflow

See `example_usage.py` for a complete example workflow:

```bash
python example_usage.py
```

## Best Practices

### Data Quality

- Use high-quality, representative examples
- Ensure consistent formatting and style
- Include diverse examples that cover your use case
- Start with 50-100 well-crafted examples

### Hyperparameters

- Start with default settings
- Use fewer epochs for smaller datasets
- Adjust learning rate based on performance
- Monitor for overfitting

### Evaluation

- Set up evaluation metrics before fine-tuning
- Use holdout data for testing
- Compare against base model performance
- Test with real-world examples

## Troubleshooting

### Common Issues

1. **API Key Error**

   - Ensure `OPENAI_API_KEY` environment variable is set
   - Verify the API key has fine-tuning permissions

2. **File Upload Errors**

   - Check file format (must be JSONL)
   - Ensure file has at least 10 examples
   - Validate JSON structure

3. **Job Creation Errors**

   - Verify file ID is correct
   - Check model name is supported
   - Ensure sufficient API credits

4. **Monitoring Issues**
   - Jobs can take several minutes to hours
   - Use appropriate poll intervals
   - Check job status manually if needed

### Getting Help

- Check the [OpenAI Fine-tuning Documentation](https://platform.openai.com/docs/guides/fine-tuning)
- Review error messages for specific issues
- Use the `validate` command to check data format
- Test with the provided example data first

## Cost Considerations

Fine-tuning costs include:

- Training tokens (based on your data)
- Model usage (fine-tuned models cost more per token)
- Storage (fine-tuned models are stored)

Estimate costs using the [OpenAI Pricing Calculator](https://openai.com/pricing).

## Safety and Compliance

- Review OpenAI's [Usage Policies](https://openai.com/policies/usage-policies)
- Ensure your training data complies with policies
- Test your fine-tuned model thoroughly before deployment
- Monitor model outputs for safety issues

## License

This script is provided as-is for educational and development purposes. Please review OpenAI's terms of service for commercial usage.
