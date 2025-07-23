import os
from transformers import AutoModelForCausalLM, AutoConfig
from dotenv import load_dotenv
import torch
import argparse
from datetime import datetime

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


def run_inference_with_hooks(model_id, text="Hello world", enable_profiling=False, profile_dir="./tensorboard_logs"):
    """Run inference with hooks to track each layer's progress."""
    print(f"\n=== Running Inference with Layer Tracking: {model_id} ===")
    
    # Get config and tokenizer
    config = AutoConfig.from_pretrained(
        model_id,
        token=os.getenv('HUGGINGFACE_HUB_TOKEN')
    )
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=os.getenv('HUGGINGFACE_HUB_TOKEN')
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create model on actual device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = AutoModelForCausalLM.from_config(config)
    model = model.to(device)
    model.eval()
    
    # Hook to track layer activations
    layer_outputs = {}
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                # For layers that return tuples, take the first element
                activation = output[0]
            else:
                activation = output
            
            if hasattr(activation, 'shape'):
                print(f"  {name}: {activation.shape} | {activation.dtype}")
                layer_outputs[name] = {
                    'shape': activation.shape,
                    'dtype': activation.dtype,
                    'mean': activation.mean().item() if activation.numel() > 0 else 0,
                    'std': activation.std().item() if activation.numel() > 0 else 0
                }
        return hook
    
    # Register hooks for all layers
    hooks = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only leaf modules
            hook = module.register_forward_hook(hook_fn(name))
            hooks.append(hook)
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt").to(device)
    print(f"Input text: '{text}'")
    print(f"Input shape: {inputs.input_ids.shape}")
    
    # Run inference
    print("\nLayer-by-layer progress:")
    
    if enable_profiling:
        # Setup TensorBoard profiling
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(profile_dir, f"profile_{timestamp}")
        os.makedirs(log_dir, exist_ok=True)
        print(f"TensorBoard profiling enabled. Logs will be saved to: {log_dir}")
        print("To view results, run: tensorboard --logdir {}".format(profile_dir))
        
        # Profile with detailed trace including matrix operations
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            with_modules=True
        ) as prof:
            with torch.no_grad():
                outputs = model(**inputs)
        
        # Export traces
        prof.export_chrome_trace(os.path.join(log_dir, "trace.json"))
        prof.export_stacks(os.path.join(log_dir, "profiler_stacks.txt"), "self_cpu_time_total")
        
        # Also save TensorBoard format
        trace_handler = torch.profiler.tensorboard_trace_handler(log_dir)
        trace_handler(prof)
    else:
        with torch.no_grad():
            outputs = model(**inputs)
    
    # Print summary
    print("\nLayer Statistics Summary:")
    for name, stats in layer_outputs.items():
        print(f"{name}: shape={stats['shape']}, mean={stats['mean']:.4f}, std={stats['std']:.4f}")
    
    # Clean up hooks
    for hook in hooks:
        hook.remove()
    
    return outputs, layer_outputs



def main():
    parser = argparse.ArgumentParser(description='Inspect model architecture and run inference with layer tracking')
    parser.add_argument('--model_id', type=str, default="meta-llama/Llama-2-7b-hf", 
                       help='Model ID to inspect')
    parser.add_argument('--run_inference', action='store_true',
                       help='Run inference with layer-by-layer tracking')
    parser.add_argument('--text', type=str, default="Hello world",
                       help='Text for inference')
    parser.add_argument('--enable_profiling', action='store_true',
                       help='Enable TensorBoard profiling for detailed computation tracking')
    parser.add_argument('--profile_dir', type=str, default="./tensorboard_logs",
                       help='Directory to save TensorBoard profiling logs')
    
    args = parser.parse_args()
    
    # Inspect architecture
    inspect_model_architecture(args.model_id)
    
    # Run inference with hooks if requested
    if args.run_inference:
        run_inference_with_hooks(args.model_id, args.text, args.enable_profiling, args.profile_dir)


if __name__ == "__main__":
    main()