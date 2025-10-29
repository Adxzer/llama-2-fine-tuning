# Quick Start Guide: Fine-tuning with Local Llama Models

This guide helps you set up fine-tuning using models downloaded directly from Meta (without HuggingFace).

## Prerequisites

-   GPU with at least 16GB VRAM (24GB recommended for 8B models)
-   Python 3.8+
-   CUDA-capable GPU

## Step-by-Step Setup

### 1. Install Dependencies

```powershell
# Install llama-cookbook and dependencies
pip install llama-cookbook ipywidgets

# Install llama-stack CLI for downloading models
pip install llama-stack -U
```

### 2. Download a Llama Model

You have two options:

**Option A: Using the PowerShell Script**

```powershell
cd finetuning
.\download_llama_model.ps1
```

**Option B: Manual Download**

```powershell
# List available models
llama model list

# Download your chosen model
llama model download --source meta --model-id Llama-3.1-8B-Instruct

# When prompted, paste your custom URL from Meta:
# https://download.llamameta.net/*?Policy=...
```

### 3. Choose the Right Model

Based on your GPU memory:

| Model                 | GPU Memory | Best For                   |
| --------------------- | ---------- | -------------------------- |
| Llama-3.2-1B-Instruct | 8GB+       | Testing, small tasks       |
| Llama-3.2-3B-Instruct | 16GB+      | Good balance               |
| Llama-3.1-8B-Instruct | 24GB+      | Best quality (recommended) |
| Llama-2-7b-hf         | 24GB+      | If you need Llama 2        |

### 4. Find Your Model Path

After downloading, your model will be at:

```
C:\Users\<YourUsername>\.llama\checkpoints\<ModelName>\
```

For example:

```
C:\Users\adamp\.llama\checkpoints\Llama-3.1-8B-Instruct\
```

### 5. Update the Notebook

Open `quickstart_peft_finetuning.ipynb` and in **Step 1** (cell 6), update the model path:

```python
# Replace this line:
train_config.model_name = "meta-llama/Meta-Llama-3.1-8B"

# With your local path:
train_config.model_name = r"C:\Users\adamp\.llama\checkpoints\Llama-3.1-8B-Instruct"
```

### 6. Run the Notebook

Execute the cells in order:

1. **Step 0**: Install dependencies (if needed)
2. **Step 1**: Load the model (this will take a few minutes)
3. **Step 2**: Test the base model
4. **Step 3**: Load the dataset
5. **Step 4**: Prepare for PEFT (LoRA)
6. **Step 5**: Fine-tune (this takes the longest)
7. **Step 6**: Save the model
8. **Step 7**: Test the fine-tuned model

## Troubleshooting

### "Out of Memory" Errors

If you get OOM errors, try:

1. **Reduce context length** in Step 1:

    ```python
    train_config.context_length = 512  # or 1024
    ```

2. **Use 4-bit quantization** instead of 8-bit:

    ```python
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    ```

3. **Use a smaller model** (3B or 1B instead of 8B)

### Model Not Found

If the model path isn't working:

1. Check the exact path where your model was downloaded
2. Use raw string notation: `r"C:\Users\..."`
3. Or use forward slashes: `"C:/Users/..."`

### Download URL Expired

Your custom URL is valid for 48 hours and 5 downloads. If expired:

1. Visit Meta's Llama download page
2. Request the models again (instant approval if already approved)
3. Use the new URL provided

## What Gets Fine-tuned?

This notebook uses **LoRA (Low-Rank Adaptation)**:

-   Only trains ~1% of the model parameters
-   Creates a small adapter file (~10-100MB)
-   Original model stays unchanged
-   Much faster and cheaper than full fine-tuning
-   Works great for most use cases

## Next Steps After Fine-tuning

Your fine-tuned model will be saved to `./meta-llama-samsum/`

To use it:

```python
from peft import PeftModel
from transformers import LlamaForCausalLM

# Load base model
base_model = LlamaForCausalLM.from_pretrained("path/to/base/model")

# Load fine-tuned adapter
model = PeftModel.from_pretrained(base_model, "./meta-llama-samsum")
```

## Resources

-   [Full fine-tuning documentation](./README.md)
-   [LLM fine-tuning overview](./LLM_finetuning_overview.md)
-   [Single GPU guide](./singlegpu_finetuning.md)
-   [Multi GPU guide](./multigpu_finetuning.md)
-   [Custom datasets](./datasets/README.md)
