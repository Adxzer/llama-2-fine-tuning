def load_model(model_name, quantization_config=None):
    from transformers import LlamaForCausalLM
    import torch

    model = LlamaForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quantization_config,
        use_cache=False,
        torch_dtype=torch.float16,
    )
    return model

def save_model(model, output_dir):
    model.save_pretrained(output_dir)

def evaluate_model(model, tokenizer, eval_input):
    import torch

    model.eval()
    with torch.inference_mode():
        model_input = tokenizer(eval_input, return_tensors="pt").to("cuda")
        output = model.generate(**model_input, max_new_tokens=100)
        return tokenizer.decode(output[0], skip_special_tokens=True)