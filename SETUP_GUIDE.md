# Setup and Testing Guide

## Understanding MODEL_PATH

The `MODEL_PATH` can be either:

### Option 1: Hugging Face Model ID (Recommended for Testing)

```
MODEL_PATH=TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

The model will be downloaded automatically from Hugging Face.

**Popular options:**

-   `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (1.1B - great for testing)
-   `meta-llama/Llama-2-7b-hf` (7B - requires access request)
-   `meta-llama/Llama-3.2-1B` (1B - newer model)

### Option 2: Local Path

```
MODEL_PATH=C:\Users\adamp\Desktop\Models\llama-2-7b
```

Use this if you've already downloaded the model locally.

## Quick Start - Running the Test

1. **Install dependencies:**

```bash
pip install --user torch transformers datasets python-dotenv accelerate
```

**Important:** Wait for the installation to complete before proceeding to step 2!

2. **Login to hugging face:**

```bash
python -m huggingface_hub.commands.huggingface_cli login
```

3. **Create your .env file:**

```bash
cp .env.example .env
```

4. **Edit .env file** (optional - test script uses TinyLlama by default):

```
HUGGINGFACE_API_TOKEN=your_token_here
MODEL_PATH=TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

5. **Run the test:**

```bash
python test_train.py
```

This will:

-   Use a small subset of the IMDB dataset (100 samples)
-   Train for just 1 epoch
-   Complete in a few minutes
-   Show you if everything is working correctly

## Notes

-   The test uses TinyLlama (1.1B parameters) which is small and fast
-   For Llama-2 models, you need to request access on Hugging Face and use your API token
-   The test script automatically detects if you have a GPU and uses it
