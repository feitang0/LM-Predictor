import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", device_map="auto", cache_dir="/workspace/cache", use_safetensors=True)
# model = torch.compile(model)
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m", cache_dir="/workspace/cache")

# text = "Hello world!"
# tokens = tokenizer.tokenize(text)
# token_ids = tokenizer.encode(text)
# 
# print("Tokens:", tokens)
# print("Token IDs:", token_ids)
# print("EOS ID:", tokenizer.eos_token_id)
# print("EOS Token:", tokenizer.decode([tokenizer.eos_token_id]))
# print("EOS IDs", tokenizer.encode("<|endoftext|>"))

# seq_lens = [128, 256, 512, 1024, 2048, 4096, 8192]
batch_sizes = [1, 2, 4, 8]
seq_lens = [128, 256, 512, 1024]

print("Testing Prefill Phase...")
for batch_size in batch_sizes:
    print(f"Batch Size: {batch_size}")
    for seq_len in seq_lens:
        print(f"\tSeq Length: {seq_len}")
        for i in range(5):
            input = tokenizer.decode([tokenizer.eos_token_id] * seq_len)
            batch_inputs = [input] * batch_size
            # print("Input:", input)
            # inputs = tokenizer(["A list of colors: red, blue"], return_tensors="pt").to(model.device)
            inputs = tokenizer(batch_inputs, return_tensors="pt").to(model.device)

            start = time.perf_counter()
            generated_ids = model.generate(**inputs, max_new_tokens=1, use_cache=True)
            end = time.perf_counter()

            print(f"\t\tRound {i}: {end - start:.6f} seconds")

# print("Output:", tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0])
