# Configuration settings for fine-tuning the Llama-2-7B model

class Config:
    def __init__(self):
        # Model and training parameters
        self.model_name = "meta-llama/Llama-2-7B"
        self.output_dir = "checkpoints/"
        self.num_epochs = 3
        self.batch_size = 2
        self.learning_rate = 5e-5
        self.gradient_accumulation_steps = 4
        self.max_length = 1024
        self.use_fp16 = True
        self.logging_steps = 100
        self.save_steps = 500
        self.evaluation_steps = 500

    def display(self):
        print("Training Configuration:")
        print(f"Model Name: {self.model_name}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Number of Epochs: {self.num_epochs}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Gradient Accumulation Steps: {self.gradient_accumulation_steps}")
        print(f"Max Length: {self.max_length}")
        print(f"Use FP16: {self.use_fp16}")
        print(f"Logging Steps: {self.logging_steps}")
        print(f"Save Steps: {self.save_steps}")
        print(f"Evaluation Steps: {self.evaluation_steps}")