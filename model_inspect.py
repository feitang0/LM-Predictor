import os
from transformers import AutoModelForCausalLM, AutoConfig
from dotenv import load_dotenv
import torch
import argparse
import json

load_dotenv()


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
    
    # Only download config
    config = AutoConfig.from_pretrained(
        model_id,
        token=os.getenv('HUGGINGFACE_HUB_TOKEN')
    )

    # Create model from config on meta device (no weight download)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)

    
    # Print full architecture
    print("\nFull Model Architecture:")
    print(model)

    # Convert model architecture to JSON
    arch_dict = module_to_dict(model)
    json_repr = json.dumps(arch_dict, indent=2)
    
    # Print JSON representation
    print("\n=== Model Architecture (JSON) ===")
    print(json_repr)
    
    # Save to file
    filename = f"model_arch_{model_id.replace('/', '_')}.json"
    with open(filename, "w") as f:
        f.write(json_repr)
    print(f"\nArchitecture saved to: {filename}")
    
    return model, config




def main():
    parser = argparse.ArgumentParser(description='Inspect model architecture')
    parser.add_argument('--model_id', type=str, default="meta-llama/Llama-2-7b-hf", 
                       help='Model ID to inspect')
    
    args = parser.parse_args()
    
    # Inspect architecture
    inspect_model_architecture(args.model_id)


if __name__ == "__main__":
    main()