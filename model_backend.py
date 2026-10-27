"""Actual local pre-trained model execution using Hugging Face Transformers."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

_MODEL_CACHE = {}

def load_generator(model_id):
    if model_id not in _MODEL_CACHE:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        _MODEL_CACHE[model_id] = (tokenizer, model)
    tokenizer, model = _MODEL_CACHE[model_id]
    return {"tokenizer": tokenizer, "model": model, "model_id": model_id}

def generate_completion(generator, prompt, max_new_tokens=45, temperature=0.7, seed=42):
    tokenizer, model = generator["tokenizer"], generator["model"]
    torch.manual_seed(seed)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=True,
            temperature=max(0.1, float(temperature)),
            top_p=0.92,
            pad_token_id=tokenizer.pad_token_id,
            no_repeat_ngram_size=3
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)
