import os
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from dotenv import load_dotenv
import torch
import argparse
from deepspeed.profiling.flops_profiler import get_model_profile
from deepspeed.accelerator import get_accelerator

load_dotenv()


def inspect_model_architecture(model_id):
    """Inspect model architecture without loading weights."""
    print(f"=== Model Architecture: {model_id} ===")
    
    # Only download config
    config = AutoConfig.from_pretrained(
        model_id,
        token=os.getenv('HUGGINGFACE_HUB_TOKEN')
    )

    # Create model from config on meta device (no weight download)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)

    # Print model configuration
    print("\nModel Configuration:")
    print(config)
    
    # Print full architecture
    print("\nFull Model Architecture:")
    print(model)

    
    return model, config


def create_fake_kv_cache(model, batch_size, context_len, device='cpu'):
    """Create fake KV cache for simulating decode phase."""
    config = model.config
    num_layers = getattr(config, 'num_hidden_layers', getattr(config, 'n_layer', 32))
    num_heads = getattr(config, 'num_attention_heads', getattr(config, 'n_head', 32))
    hidden_size = getattr(config, 'hidden_size', getattr(config, 'd_model', 4096))
    head_dim = hidden_size // num_heads
    
    # Get model's dtype to match KV cache dtype
    model_dtype = next(model.parameters()).dtype
    
    try:
        # Try to use DynamicCache (newer transformers)
        from transformers import DynamicCache
        cache = DynamicCache()
        
        for layer_idx in range(num_layers):
            # Create fake key and value tensors with matching dtype
            # Shape: [batch_size, num_heads, context_len, head_dim]
            fake_key = torch.randn(batch_size, num_heads, context_len, head_dim, 
                                 device=device, dtype=model_dtype)
            fake_value = torch.randn(batch_size, num_heads, context_len, head_dim, 
                                   device=device, dtype=model_dtype)
            cache.update(fake_key, fake_value, layer_idx)
        
        return cache
        
    except ImportError:
        # Fallback to tuple format for older transformers
        past_key_values = []
        for _ in range(num_layers):
            # Each layer has (key, value) tensors with matching dtype
            # Shape: [batch_size, num_heads, context_len, head_dim]
            fake_key = torch.randn(batch_size, num_heads, context_len, head_dim, 
                                 device=device, dtype=model_dtype)
            fake_value = torch.randn(batch_size, num_heads, context_len, head_dim, 
                                   device=device, dtype=model_dtype)
            past_key_values.append((fake_key, fake_value))
        
        return tuple(past_key_values)


def profile_prefill_stage(model_id, batch_size=1, seq_len=512):
    """Profile prefill stage FLOPs - processing initial prompt."""
    print(f"\n=== Prefill Stage Profiling: {model_id} ===")
    print(f"Batch size: {batch_size}, Sequence length: {seq_len}")
    print("🚀 Profiling prefill stage (initial prompt processing)")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=os.getenv('HUGGINGFACE_HUB_TOKEN')
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    with torch.device("cpu"):
        # Create model with random weights (no download)
        config = AutoConfig.from_pretrained(
            model_id,
            token=os.getenv('HUGGINGFACE_HUB_TOKEN')
        )
        model = AutoModelForCausalLM.from_config(config)
        model.eval()
        
        # Create prefill input constructor
        def prefill_input_constructor():
            # Create dummy prompt
            dummy_text = tokenizer.pad_token * (seq_len - 2)
            inputs = tokenizer(
                [dummy_text] * batch_size,
                padding=True,
                truncation=True,
                max_length=seq_len,
                return_tensors="pt"
            )
            # Prefill stage: no past_key_values, use_cache=True to generate cache
            inputs_dict = dict(inputs)
            inputs_dict['use_cache'] = True
            return inputs_dict
        
        # Profile prefill
        flops, macs, params = get_model_profile(
            model=model,
            kwargs=prefill_input_constructor(),
            print_profile=True,
            detailed=True,
            module_depth=-1,
            top_modules=3,
            warm_up=0,  # No warmup needed for FLOP counting
            as_string=True,
            output_file=None,
            ignore_modules=None
        )
        
        print(f"\n=== Prefill Summary ===")
        print(f"Total Parameters: {params}")
        print(f"Total MACs: {macs}")
        print(f"Total FLOPs: {flops}")
        
        return flops, macs, params


def profile_decode_stage(model_id, batch_size=1, context_len=512):
    """Profile decode stage FLOPs - generating one token with KV cache."""
    print(f"\n=== Decode Stage Profiling: {model_id} ===")
    print(f"Batch size: {batch_size}, Context length: {context_len}")
    print("⚡ Profiling decode stage (single token generation with KV cache)")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=os.getenv('HUGGINGFACE_HUB_TOKEN')
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    with torch.device("cpu"):
        # Create model with random weights (no download)
        config = AutoConfig.from_pretrained(
            model_id,
            token=os.getenv('HUGGINGFACE_HUB_TOKEN')
        )
        model = AutoModelForCausalLM.from_config(config)
        model.eval()
        
        # Create fake KV cache simulating previous tokens
        fake_past_kv = create_fake_kv_cache(model, batch_size, context_len, device='cpu')
        
        # Create decode input constructor
        def decode_input_constructor():
            # Single new token input
            inputs = tokenizer(
                [tokenizer.pad_token] * batch_size,
                return_tensors="pt"
            )
            # Decode stage: provide past_key_values and use_cache=True
            inputs_dict = dict(inputs)
            inputs_dict['past_key_values'] = fake_past_kv
            inputs_dict['use_cache'] = True
            return inputs_dict
        
        # Profile decode
        flops, macs, params = get_model_profile(
            model=model,
            kwargs=decode_input_constructor(),
            print_profile=True,
            detailed=True,
            module_depth=-1,
            top_modules=3,
            warm_up=0,  # No warmup needed for FLOP counting
            as_string=True,
            output_file=None,
            ignore_modules=None
        )
        
        print(f"\n=== Decode Summary ===")
        print(f"Total Parameters: {params}")
        print(f"Total MACs: {macs}")
        print(f"Total FLOPs: {flops}")
        print(f"Context length used: {context_len}")
        
        return flops, macs, params



def main():
    parser = argparse.ArgumentParser(description='Inspect model architecture and profile FLOPs')
    parser.add_argument('--model_id', type=str, default="meta-llama/Llama-2-7b-hf", 
                       help='Model ID to inspect')
    parser.add_argument('--profile_prefill', action='store_true',
                       help='Profile prefill stage FLOPs (initial prompt processing)')
    parser.add_argument('--profile_decode', action='store_true',
                       help='Profile decode stage FLOPs (single token generation with KV cache)')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for FLOPs profiling')
    parser.add_argument('--seq_len', type=int, default=512,
                       help='Sequence length for prefill profiling')
    parser.add_argument('--context_len', type=int, default=512,
                       help='Context length for decode profiling (simulated KV cache size)')
    
    args = parser.parse_args()
    
    # Inspect architecture
    inspect_model_architecture(args.model_id)
    
    # Profile prefill stage if requested
    if args.profile_prefill:
        profile_prefill_stage(args.model_id, args.batch_size, args.seq_len)
    
    # Profile decode stage if requested
    if args.profile_decode:
        profile_decode_stage(args.model_id, args.batch_size, args.context_len)


if __name__ == "__main__":
    main()