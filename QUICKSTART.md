# Quick Start Guide

Get up and running with OpenAI fine-tuning in 5 minutes!

## 1. Setup (2 minutes)

```bash
# Install dependencies
pip install -r requirements.txt

# Set your API key
export OPENAI_API_KEY="your-api-key-here"

# Test installation
python test_installation.py
```

## 2. Validate Your Data (30 seconds)

```bash
python openai_finetuning.py validate data/toy_chat_fine_tuning.jsonl
```

## 3. Upload & Fine-tune (2 minutes)

```bash
# Upload your data
python openai_finetuning.py upload data/toy_chat_fine_tuning.jsonl

# Create fine-tuning job (use the file ID from upload)
python openai_finetuning.py create --file-id file-xxx --model gpt-4.1-nano-2025-04-14

# Monitor the job (use the job ID from create)
python openai_finetuning.py monitor --job-id ftjob-xxx
```

## 4. Test Your Model (30 seconds)

```bash
# Test with default prompts
python openai_finetuning.py test --model-id ft:gpt-4.1-nano-2025-04-14:openai::xxx

# Test with custom prompts
python openai_finetuning.py test --model-id ft:gpt-4.1-nano-2025-04-14:openai::xxx --prompts "What is AI?" "Tell me about machine learning"
```

## That's it! 🎉

Your fine-tuned model is ready to use. Check the full documentation in `FINETUNING_README.md` for advanced features and troubleshooting.

## Need Help?

- Run `python openai_finetuning.py --help` for command help
- Check `example_usage.py` for a complete workflow
- See `FINETUNING_README.md` for detailed documentation
