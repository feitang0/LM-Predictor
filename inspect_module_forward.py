# from transformers.models.llama.modeling_llama import LlamaSdpaAttention
# import transformers.models.llama.modeling_llama.LlamaMLP
import transformers
import torch
import inspect
print(inspect.getsource(transformers.models.llama.modeling_llama.LlamaSdpaAttention.__init__))
print(inspect.getsource(transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward))
print(inspect.getsource(transformers.models.llama.modeling_llama.LlamaMLP.__init__))
print(inspect.getsource(transformers.models.llama.modeling_llama.LlamaMLP.forward))
print(inspect.getsource(torch.nn.Linear.__init__))
print(inspect.getsource(torch.nn.Linear.forward))
print(inspect.getsource(transformers.models.llama.modeling_llama.LlamaDecoderLayer.forward))
