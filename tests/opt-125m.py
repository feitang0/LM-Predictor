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
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
seq_lens = [32, 64, 128, 256, 512, 1024]
cache_lens = [32, 64, 128, 256, 512, 1024]

print("Testing Prefill Phase...")
for batch_size in batch_sizes:
# for batch_size in range(1, 65, 1):
    print(f"Batch Size: {batch_size}")
    for seq_len in seq_lens:
    # for seq_len in range(1, 1025, 1):
        print(f"\tSeq Length: {seq_len}")
        for i in range(5):
            input_text = tokenizer.decode([tokenizer.eos_token_id] * seq_len)
            batch_input_texts = [input_text] * batch_size
            # print("Input:", input)
            # inputs = tokenizer(["A list of colors: red, blue"], return_tensors="pt").to(model.device)
            inputs = tokenizer(batch_input_texts, return_tensors="pt").to(model.device)

            torch.cuda.synchronize()
            start = time.perf_counter()
            # generated_ids = model.generate(**batch_inputs, max_new_tokens=1, use_cache=True)
            with torch.no_grad():
                outputs = model(**inputs, use_cache=True)
            torch.cuda.synchronize()
            end = time.perf_counter()

            print(f"\t\tRound {i}: {end - start:.6f} seconds")

# print("Output:", tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0])

print("Testing Decode Phase...")
for batch_size in batch_sizes:
# for batch_size in range(1, 65, 1):
    print(f"Batch Size: {batch_size}")
    for cache_len in cache_lens:
    # for cache_len in range(1, 1025, 1):
        print(f"\tCache Length: {cache_len}")
        for i in range(5):
            input_text = tokenizer.decode([tokenizer.eos_token_id] * cache_len)
            batch_input_texts = [input_text] * batch_size
            # print("Input:", input)
            # inputs = tokenizer(["A list of colors: red, blue"], return_tensors="pt").to(model.device)
            inputs = tokenizer(batch_input_texts, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model(**inputs, use_cache=True)
                past_key_values = outputs.past_key_values
            # torch.cuda.synchronize()
            next_token_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

            torch.cuda.synchronize()
            start = time.perf_counter()
            # generated_ids = model.generate(**batch_inputs, max_new_tokens=1, use_cache=True)
            with torch.no_grad():
                outputs = model(input_ids=next_token_id, past_key_values=past_key_values, use_cache=True)
            torch.cuda.synchronize()
            end = time.perf_counter()

            print(f"\t\tRound {i}: {end - start:.6f} seconds")
