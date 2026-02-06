import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", device_map="auto")
model = torch.compile(model)
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", cache_dir="./cache")

text = "Hello world!"
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.encode(text)

print("Tokens:", tokens)
print("Token IDs:", token_ids)
print("EOS ID:", tokenizer.eos_token_id)
print("EOS Token:", tokenizer.decode([tokenizer.eos_token_id]))
print("EOS IDs", tokenizer.encode("<|endoftext|>"))

input = tokenizer.decode([tokenizer.eos_token_id] * 8)
print("Input:", input)
# inputs = tokenizer(["A list of colors: red, blue"], return_tensors="pt").to(model.device)
inputs = tokenizer(input, return_tensors="pt").to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=1)
print(tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0])
