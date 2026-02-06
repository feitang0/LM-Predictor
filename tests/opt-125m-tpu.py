import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import PyTorch XLA for TPU
import torch_xla

# Get TPU device
device = torch_xla.device()
print(f"Using device: {device}")


kwargs = {
        "low_cpu_mem_usage": True,  # Keep RAM footprint low during load
        "device_map": None,  # Disable accelerate device mapping
    }

model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-125m",
    **kwargs
).to(device)

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m", cache_dir="/workspace/cache")

batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
seq_lens = [32, 64, 128, 256, 512, 1024]
cache_lens = [32, 64, 128, 256, 512, 1024]

print("Testing Prefill Phase...")
for batch_size in batch_sizes:
    print(f"Batch Size: {batch_size}")
    for seq_len in seq_lens:
        print(f"\tSeq Length: {seq_len}")
        for i in range(5):
            input_text = tokenizer.decode([tokenizer.eos_token_id] * seq_len)
            batch_input_texts = [input_text] * batch_size
            inputs = tokenizer(batch_input_texts, return_tensors="pt").to(device)

            torch_xla.sync()
            start = time.perf_counter()
            with torch.no_grad():
                outputs = model(**inputs, use_cache=True)
            torch_xla.sync()
            end = time.perf_counter()

            print(f"\t\tRound {i}: {end - start:.6f} seconds")

print("Testing Decode Phase...")
for batch_size in batch_sizes:
    print(f"Batch Size: {batch_size}")
    for cache_len in cache_lens:
        print(f"\tCache Length: {cache_len}")
        for i in range(5):
            input_text = tokenizer.decode([tokenizer.eos_token_id] * cache_len)
            batch_input_texts = [input_text] * batch_size
            inputs = tokenizer(batch_input_texts, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs, use_cache=True)
                past_key_values = outputs.past_key_values
            next_token_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

            torch_xla.sync()
            start = time.perf_counter()
            with torch.no_grad():
                outputs = model(input_ids=next_token_id, past_key_values=past_key_values, use_cache=True)
            torch_xla.sync()
            end = time.perf_counter()

            print(f"\t\tRound {i}: {end - start:.6f} seconds")
