import os
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch

# Load environment variables
load_dotenv()

print("🚀 Starting test training run...")

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")  # Using smaller model for testing
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs/")
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "checkpoints/")

print(f"📦 Loading model: {MODEL_PATH}")

# Load a small test dataset (using a tiny subset)
print("📚 Loading test dataset...")
dataset = load_dataset("imdb", split="train[:100]")  # Only 100 samples for quick test

# Load tokenizer and model
print("🔧 Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, token=HUGGINGFACE_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    token=HUGGINGFACE_TOKEN,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128, padding="max_length")

print("🔄 Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text", "label"])

# Training arguments (small scale for testing)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,  # Just 1 epoch for testing
    per_device_train_batch_size=2,
    save_steps=50,
    save_total_limit=2,
    logging_steps=10,
    learning_rate=3e-5,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Start training
print("🎯 Starting training...")
trainer.train()

print("✅ Test training completed successfully!")
print(f"📁 Outputs saved to: {OUTPUT_DIR}")
