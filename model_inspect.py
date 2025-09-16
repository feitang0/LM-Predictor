import os
from transformers import AutoModelForCausalLM, AutoConfig
from dotenv import load_dotenv
import torch
import argparse
import json
from flop_analyzer import FlopAnalyzer

load_dotenv()


def load_model_on_meta_device(model_id):
    """Load model on meta device without downloading weights."""
    config = AutoConfig.from_pretrained(
        model_id,
        token=os.getenv('HUGGINGFACE_HUB_TOKEN')
    )
    
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    
    return model


def module_to_dict(module):
    """Recursively convert a PyTorch nn.Module into a nested dict."""
    children = dict(module.named_children())
    if not children:  # leaf node
        return {
            "class": module.__class__.__name__,
            "module": str(module)
        }
    return {
        "class": module.__class__.__name__,
        "children": {name: module_to_dict(child) for name, child in children.items()}
    }


def inspect_model_architecture(model_id):
    """Inspect model architecture without loading weights."""
    print(f"=== Model Architecture: {model_id} ===")
    
    # Load model on meta device
    model = load_model_on_meta_device(model_id)
    
    # Print full architecture
    print("\nFull Model Architecture:")
    print(model)


def analyze_model_flops(model_id, batch_size=1, seq_len=2048):
    """Analyze model FLOPs and memory usage."""
    print(f"\n=== FLOP Analysis: {model_id} ===")
    
    # Load model and analyze FLOPs
    model = load_model_on_meta_device(model_id)
    
    # Analyze FLOPs and memory
    analyzer = FlopAnalyzer()
    results = analyzer.analyze_model(model, batch_size, seq_len)
    analyzer.print_analysis_summary(results, batch_size, seq_len)
    
    # Save FLOP analysis results
    flop_filename = f"flop_analysis_{model_id.replace('/', '_')}.json"
    with open(flop_filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFLOP analysis saved to: {flop_filename}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Inspect model architecture and analyze FLOPs')
    parser.add_argument('--model_id', type=str, default="meta-llama/Llama-2-7b-hf", 
                       help='Model ID to inspect')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for FLOP analysis')
    parser.add_argument('--seq_len', type=int, default=2048, help='Sequence length for FLOP analysis')
    parser.add_argument('--analyze', action='store_true', help='Run FLOP and memory analysis')
    
    args = parser.parse_args()
    
    # Always inspect architecture
    inspect_model_architecture(args.model_id)
    
    # Additionally run FLOP analysis if requested
    if args.analyze:
        analyze_model_flops(args.model_id, args.batch_size, args.seq_len)


if __name__ == "__main__":
    main()