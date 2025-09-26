# OpenAI Fine-tuning Guide

This guide explains how to use the fine-tuning scripts to train custom OpenAI models with your data.

## Overview

The fine-tuning system consists of two main scripts:

1. **`fine_tune_models.py`** - Comprehensive fine-tuning script with full control
2. **`run_fine_tuning.py`** - Simple script to run fine-tuning with your existing data files

## Prerequisites

1. **OpenAI API Key**: Set your API key as an environment variable:

   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

2. **Python Dependencies**: Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. **Data Files**: Ensure you have your training and evaluation data in JSONL format:
   - `data/train.jsonl` - Training data (required)
   - `data/eval.jsonl` - Evaluation data (optional but recommended)

## Quick Start

### Option 1: Simple Fine-tuning (Recommended for beginners)

Run fine-tuning with your existing data files:

```bash
python run_fine_tuning.py
```

This script will:

- Use `data/train.jsonl` for training
- Use `data/eval.jsonl` for evaluation (if available)
- Apply default hyperparameters from `config.json`
- Test the fine-tuned model with sample prompts
- Provide detailed results and the fine-tuned model ID

### Option 2: Advanced Fine-tuning

For more control over the fine-tuning process:

```bash
python fine_tune_models.py \
    --train-file data/train.jsonl \
    --eval-file data/eval.jsonl \
    --model gpt-4.1-nano-2025-04-14 \
    --suffix my-custom-model
```

## Configuration

### config.json

The `config.json` file contains default settings:

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
    },
    "conservative": {
      "n_epochs": 5,
      "batch_size": 1,
      "learning_rate_multiplier": 0.5
    },
    "aggressive": {
      "n_epochs": 20,
      "batch_size": 2,
      "learning_rate_multiplier": 2.0
    }
  },
  "test_prompts": [
    "What is the capital of France?",
    "Tell me a joke.",
    "How do I make a sandwich?"
  ]
}
```

## Data Format

Your JSONL files should contain one JSON object per line with the following structure:

```json
{
  "messages": [
    { "role": "user", "content": "Your question here" },
    { "role": "assistant", "content": "Expected response here" }
  ]
}
```

### Example Training Data

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What is the situation you engaged in or avoided?\n\nI avoided reaching out to my coach for a follow-up lesson."
    },
    {
      "role": "assistant",
      "content": "I could have used thinking about the situation to actually reach out to my coach about a follow up lesson"
    }
  ]
}
```

## Command Line Options

### fine_tune_models.py

```bash
python fine_tune_models.py --help
```

**Required Arguments:**

- `--train-file`: Path to training data file (JSONL)

**Optional Arguments:**

- `--eval-file`: Path to evaluation data file (JSONL)
- `--model`: Base model to fine-tune (default: gpt-4.1-nano-2025-04-14)
- `--config`: Path to configuration file (default: config.json)
- `--suffix`: Suffix for the fine-tuned model name
- `--hyperparameters`: JSON string of hyperparameters
- `--test-only`: Only test an existing fine-tuned model
- `--model-id`: Fine-tuned model ID for testing

### Examples

**Basic fine-tuning:**

```bash
python fine_tune_models.py --train-file data/train.jsonl
```

**With evaluation data:**

```bash
python fine_tune_models.py \
    --train-file data/train.jsonl \
    --eval-file data/eval.jsonl
```

**Custom hyperparameters:**

```bash
python fine_tune_models.py \
    --train-file data/train.jsonl \
    --hyperparameters '{"n_epochs": 5, "batch_size": 1, "learning_rate_multiplier": 0.5}'
```

**Test existing model:**

```bash
python fine_tune_models.py \
    --test-only \
    --model-id ft:gpt-4.1-nano-2025-04-14:your-org::abc123
```

## Monitoring and Results

### Job Monitoring

The script automatically monitors your fine-tuning job and provides updates:

```
2024-01-15 10:30:00 - INFO - Creating fine-tuning job for model: gpt-4.1-nano-2025-04-14
2024-01-15 10:30:05 - INFO - Fine-tuning job created successfully. ID: ftjob-abc123
2024-01-15 10:30:05 - INFO - Job status: running
2024-01-15 10:35:00 - INFO - Job status: running
2024-01-15 10:40:00 - INFO - Job status: succeeded
2024-01-15 10:40:00 - INFO - Fine-tuning completed successfully! Model ID: ft:gpt-4.1-nano-2025-04-14:your-org::abc123
```

### Results

Upon successful completion, you'll receive:

1. **Fine-tuned Model ID**: Use this to make API calls with your custom model
2. **Job ID**: For tracking and reference
3. **Test Results**: Performance on sample prompts
4. **Training Statistics**: Number of tokens trained, etc.

### Using Your Fine-tuned Model

Once fine-tuning is complete, use your model ID in API calls:

```python
import openai

client = openai.OpenAI()

response = client.chat.completions.create(
    model="ft:gpt-4.1-nano-2025-04-14:your-org::abc123",
    messages=[
        {"role": "user", "content": "Your prompt here"}
    ]
)

print(response.choices[0].message.content)
```

## Troubleshooting

### Common Issues

1. **API Key Not Set**

   ```
   Error: OPENAI_API_KEY environment variable is not set.
   ```

   Solution: Set your API key: `export OPENAI_API_KEY='your-key'`

2. **Invalid Data Format**

   ```
   Line 5: Missing 'messages' field
   ```

   Solution: Ensure your JSONL follows the correct format with 'messages' array

3. **Insufficient Training Data**

   ```
   File must have at least 10 valid training examples, found 5
   ```

   Solution: Add more training examples to your JSONL file

4. **File Upload Failed**
   ```
   Error uploading file: Invalid file format
   ```
   Solution: Ensure your file is valid JSONL with proper encoding

### Logs

Check the `fine_tuning.log` file for detailed logs of the fine-tuning process.

### Validation

The script automatically validates your data before uploading:

- Checks JSON format
- Validates message structure
- Ensures minimum number of examples
- Verifies role types (user, assistant, system)

## Best Practices

1. **Start Small**: Begin with 50-100 high-quality examples
2. **Quality over Quantity**: Focus on well-crafted, representative examples
3. **Use Evaluation Data**: Always provide evaluation data for better results
4. **Monitor Performance**: Test your model thoroughly before production use
5. **Iterate**: Fine-tune based on results and add more data as needed

## Cost Considerations

- Fine-tuning costs depend on the number of tokens in your training data
- Monitor your usage in the OpenAI dashboard
- Start with smaller datasets to test effectiveness
- Consider using smaller models (nano/mini) for cost efficiency

## Support

For issues or questions:

1. Check the logs in `fine_tuning.log`
2. Verify your data format matches the requirements
3. Ensure your API key has sufficient credits
4. Review the OpenAI fine-tuning documentation

## Next Steps

After successful fine-tuning:

1. Test your model with various prompts
2. Compare performance against the base model
3. Iterate and improve your training data
4. Deploy your fine-tuned model in your application
