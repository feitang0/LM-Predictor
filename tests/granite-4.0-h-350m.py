import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "ibm-granite/granite-4.0-h-350M"
tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(model_path)
model.eval()

chat = [
        {"role": "user", "content": "Please list one IBM Research laboratory located in the United States. You should only output its name and location."},
        ]
chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
input_tokens = tokenizer(chat, return_tensors="pt")
output = model.generate(**input_tokens, max_new_tokens=100)
output = tokenizer.batch_decode(output)
print(output[0])
