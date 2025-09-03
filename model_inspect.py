import os
from transformers import AutoModelForCausalLM, AutoConfig
from dotenv import load_dotenv
import torch
import argparse
import json

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

    # Print all parameter names and dtypes
    print("\n=== All Parameter Data Types ===")
    for name, param in model.named_parameters():
        print(f"{name}: {param.dtype}")
    
    
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