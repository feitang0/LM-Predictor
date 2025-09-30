#!/usr/bin/env python3
"""
Modular model analyzer using meta device inspection and module database.
Replaces legacy hardcoded approach with composable analysis system.
"""

import os
import json
import argparse
import re
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, AutoConfig

load_dotenv()


class ModelAnalyzer:
    """Modular model analyzer using meta device and module database."""

    def __init__(self, model_id: str = None, model_json_path: str = None):
        """Initialize analyzer with model ID and/or model JSON path."""
        self.model_id = model_id
        self.model_json_path = model_json_path
        self.model = None
        self.config = None

        # If model_json_path is provided, extract model_id from JSON for display
        if model_json_path:
            try:
                with open(model_json_path, 'r') as f:
                    model_structure = json.load(f)
                if not model_id:  # Only override if model_id not explicitly provided
                    self.model_id = model_structure.get("model_id", "unknown")
            except Exception as e:
                print(f"Warning: Could not extract model_id from JSON: {e}")
                if not model_id:
                    self.model_id = "unknown"

    def load_model_architecture(self) -> None:
        """Load model architecture on meta device without weights."""
        print(f"Loading model architecture: {self.model_id}")

        # Load config
        hf_token = os.getenv('HUGGINGFACE_HUB_TOKEN')
        self.config = AutoConfig.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            token=hf_token
        )

        # Load model on meta device
        with torch.device("meta"):
            self.model = AutoModelForCausalLM.from_config(self.config)

        print(f"✓ Loaded {self.model.__class__.__name__} architecture")

        # Print model configuration and structure
        print(f"\nModel Configuration:")
        print(self.config)
        print(f"\nModel Structure:")
        self.print_enhanced_model()

    def collect_module_classes(self, module, classes=None):
        """Collect all unique module classes in the model."""
        if classes is None:
            classes = {}

        simple_name = module.__class__.__name__
        full_name = f"{module.__class__.__module__}.{module.__class__.__name__}"
        classes[simple_name] = full_name

        for child in module.children():
            self.collect_module_classes(child, classes)

        return classes

    def print_enhanced_model(self) -> None:
        """Print model with enhanced class names while keeping compact format."""
        if self.model is None:
            self.load_model_architecture()

        # Get all module class mappings
        class_mappings = self.collect_module_classes(self.model)

        # Get original string representation
        model_str = str(self.model)

        # Sort by length (longest first) to avoid partial replacements
        sorted_mappings = sorted(class_mappings.items(), key=lambda x: len(x[0]), reverse=True)

        # Replace each simple class name with full class name
        # Use word boundary replacement to avoid partial matches
        enhanced_str = model_str
        for simple_name, full_name in sorted_mappings:
            # Use regex with word boundary to ensure exact class name matches
            pattern = r'\b' + re.escape(simple_name) + r'\('
            replacement = full_name + "("
            enhanced_str = re.sub(pattern, replacement, enhanced_str)

        print(enhanced_str)

    def get_enhanced_model_str(self) -> str:
        """Get enhanced model structure as string with full class names."""
        if self.model is None:
            self.load_model_architecture()

        # Get all module class mappings
        class_mappings = self.collect_module_classes(self.model)

        # Get original string representation
        model_str = str(self.model)

        # Sort by length (longest first) to avoid partial replacements
        sorted_mappings = sorted(class_mappings.items(), key=lambda x: len(x[0]), reverse=True)

        # Replace each simple class name with full class name
        # Use word boundary replacement to avoid partial matches
        enhanced_str = model_str
        for simple_name, full_name in sorted_mappings:
            # Use regex with word boundary to ensure exact class name matches
            pattern = r'\b' + re.escape(simple_name) + r'\('
            replacement = full_name + "("
            enhanced_str = re.sub(pattern, replacement, enhanced_str)

        return enhanced_str

    def _extract_basic_layers_from_json(self, layers, parent_path="", repeat_multiplier=1):
        """Extract all basic layers from JSON structure with repeat handling."""
        basic_layers = []

        for layer in layers:
            current_repeat = repeat_multiplier * layer.get("repeat", 1)
            current_path = f"{parent_path}.{layer['name']}" if parent_path else layer['name']

            if "sub_layers" not in layer:  # This is a basic layer
                basic_layers.append({
                    "name": layer["name"],
                    "class": layer["class"],
                    "parameters": layer.get("parameters", {}),
                    "path": current_path,
                    "repeat": current_repeat
                })
            else:  # Composite layer - recurse into sub_layers
                basic_layers.extend(
                    self._extract_basic_layers_from_json(
                        layer["sub_layers"], current_path, current_repeat
                    )
                )

        return basic_layers

    def _analyze_basic_layer(self, layer_info, batch_size, seq_len, w_dtype_bytes, a_dtype_bytes):
        """Analyze a single basic layer using registry."""
        from generated_modules.registry import compute_flops, compute_memory

        # Substitute dynamic parameters in layer parameters
        resolved_params = {}
        for param_name, param_value in layer_info["parameters"].items():
            if isinstance(param_value, str) and "{" in param_value:
                # Replace placeholders with actual values
                param_str = param_value
                param_str = param_str.replace("{batch_size}", str(batch_size))
                param_str = param_str.replace("{seq_len}", str(seq_len))
                param_str = param_str.replace("{w_dtype_bytes}", str(w_dtype_bytes))
                param_str = param_str.replace("{a_dtype_bytes}", str(a_dtype_bytes))
                param_str = param_str.replace("{dtype_bytes}", str(a_dtype_bytes))  # Legacy fallback
                param_str = param_str.replace("{index_dtype_bytes}", "4")
                resolved_params[param_name] = eval(param_str)  # Safe for arithmetic expressions
            else:
                resolved_params[param_name] = param_value

        # Compute using registry with full class name (DESIGN.md recommended pattern)
        try:
            flops = compute_flops(layer_info["class"], **resolved_params)
            memory = compute_memory(layer_info["class"], **resolved_params)

            # Apply repeat multiplier
            total_flops = flops * layer_info["repeat"]
            total_memory_reads = memory["reads"] * layer_info["repeat"]
            total_memory_writes = memory["writes"] * layer_info["repeat"]
            total_memory_intermediates = memory["intermediates"] * layer_info["repeat"]

            return {
                "layer_path": layer_info["path"],
                "layer_name": layer_info["name"],
                "class": layer_info["class"],
                "repeat_count": layer_info["repeat"],
                "flops": total_flops,
                "memory_reads": total_memory_reads,
                "memory_writes": total_memory_writes,
                "memory_intermediates": total_memory_intermediates,
                "parameters": resolved_params
            }

        except Exception as e:
            print(f"  ⚠ Failed to analyze {layer_info['class']}: {e}")
            return {
                "layer_path": layer_info["path"],
                "layer_name": layer_info["name"],
                "class": layer_info["class"],
                "repeat_count": layer_info["repeat"],
                "flops": 0,
                "memory_reads": 0,
                "memory_writes": 0,
                "memory_intermediates": 0,
                "error": str(e)
            }

    def analyze(self, seqlen: int, batchsize: int, w_bit: int = 16, a_bit: int = 16,
                kv_bit: Optional[int] = None, **_) -> Dict[str, Any]:
        """
        Analyze model using JSON structure for layer-by-layer computation.

        Args:
            seqlen: sequence length
            batchsize: batch size
            w_bit: weight bit width (for dtype_bytes calculation)
            a_bit: activation bit width (for dtype_bytes calculation)
            kv_bit: unused (legacy compatibility)
            **kwargs: other legacy parameters (ignored)

        Returns:
            Analysis results in legacy format
        """
        print(f"\n=== Analyzing {self.model_id} ===")
        print(f"Parameters: B={batchsize}, S={seqlen}, w_bit={w_bit}, a_bit={a_bit}")

        # Load JSON model structure
        if not self.model_json_path:
            raise ValueError("No model JSON path provided for analysis")

        try:
            with open(self.model_json_path, 'r') as f:
                model_structure = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Model JSON not found: {self.model_json_path}")

        # Extract all basic layers from JSON
        basic_layers = self._extract_basic_layers_from_json(model_structure["layers"])

        print(f"\nAnalyzing {len(basic_layers)} basic layers:")

        # Analyze each basic layer
        w_dtype_bytes = w_bit // 8
        a_dtype_bytes = a_bit // 8
        results = []

        for layer_info in basic_layers:
            analysis = self._analyze_basic_layer(layer_info, batchsize, seqlen, w_dtype_bytes, a_dtype_bytes)
            results.append(analysis)

            # Print progress (only for layers with FLOPs)
            if analysis["flops"] > 0:
                repeat_str = f" (×{analysis['repeat_count']})" if analysis['repeat_count'] > 1 else ""
                print(f"  ✓ {analysis['layer_path']}{repeat_str}: {analysis['flops']:,} FLOPs")

        # Calculate totals
        total_flops = sum(r["flops"] for r in results)
        total_memory_reads = sum(r["memory_reads"] for r in results)
        total_memory_writes = sum(r["memory_writes"] for r in results)
        total_memory_intermediates = sum(r["memory_intermediates"] for r in results)

        # Return legacy format for compatibility
        legacy_results = {
            "decode": {},
            "prefill": {},
            "total_results": {
                "decode": {
                    "OPs": total_flops,
                    "memory_access": total_memory_reads + total_memory_writes + total_memory_intermediates,
                    "load_weight": total_memory_reads,
                    "load_act": 0,
                    "store_act": total_memory_writes,
                    "load_kv_cache": 0,
                    "store_kv_cache": 0,
                    "inference_time": total_flops / 1e12
                },
                "prefill": {
                    "OPs": total_flops,
                    "memory_access": total_memory_reads + total_memory_writes + total_memory_intermediates,
                    "load_weight": total_memory_reads,
                    "load_act": 0,
                    "store_act": total_memory_writes,
                    "load_kv_cache": 0,
                    "store_kv_cache": 0,
                    "inference_time": total_flops / 1e12
                }
            },
            "json_based_results": {
                "basic_layer_analyses": results,
                "totals": {
                    "flops": total_flops,
                    "memory_reads": total_memory_reads,
                    "memory_writes": total_memory_writes,
                    "memory_intermediates": total_memory_intermediates
                },
                "model_json_used": self.model_json_path
            }
        }

        print(f"\n=== Analysis Summary ===")
        print(f"Total FLOPs: {total_flops:,}")
        print(f"Memory Reads: {total_memory_reads:,} bytes")
        print(f"Memory Writes: {total_memory_writes:,} bytes")
        print(f"Memory Intermediates: {total_memory_intermediates:,} bytes")

        return legacy_results


def extract_basic_layer_classes(architecture: Dict[str, Any]) -> set[str]:
    """
    Recursively extract all unique basic layer class names from architecture.
    Basic layers are those WITHOUT 'sub_layers' field.

    Args:
        architecture: Architecture JSON with layers structure

    Returns:
        Set of unique class names for basic layers
    """
    classes: set[str] = set()

    def traverse(layer: Dict[str, Any]) -> None:
        if "sub_layers" not in layer:
            # Basic layer - add its class
            class_name = layer.get("class", "")
            if class_name:
                classes.add(class_name)
        else:
            # Composite layer - recurse into sub_layers
            for sub_layer in layer.get("sub_layers", []):
                traverse(sub_layer)

    for layer in architecture.get("layers", []):
        traverse(layer)

    return classes


def main():
    """Command-line interface for model analysis."""
    parser = argparse.ArgumentParser(description='Analyze LLM model FLOPs and memory using modular approach')
    parser.add_argument('--model_id', type=str, default="meta-llama/Llama-2-7b-hf",
                       help='Model ID for inspection (default: meta-llama/Llama-2-7b-hf)')
    parser.add_argument('--model_json', type=str, default=None,
                       help='Path to model JSON file (required for --analyze)')
    parser.add_argument('--analyze', action='store_true',
                       help='Run FLOPs/memory analysis (requires --model_json)')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size (default: 1)')
    parser.add_argument('--seq_len', type=int, default=2048,
                       help='Sequence length (default: 2048)')
    parser.add_argument('--w_bit', type=int, default=16,
                       help='Weight bit width (default: 16)')
    parser.add_argument('--a_bit', type=int, default=16,
                       help='Activation bit width (default: 16)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output file path (default: auto-generated)')
    parser.add_argument('--generate-arch', action='store_true',
                       help='Generate standardized model architecture JSON during inspection')
    parser.add_argument('--ensure-modules', action='store_true',
                       help='Ensure all modules needed by architecture are analyzed and generated')
    parser.add_argument('--populate-arch', action='store_true',
                       help='Populate parameters for architecture JSON (requires architecture to exist in models/)')

    args = parser.parse_args()

    try:
        if args.analyze:
            # Analysis mode requires model_json
            if not args.model_json:
                print("Error: --analyze requires --model_json")
                return 1

            analyzer = ModelAnalyzer(model_id=args.model_id, model_json_path=args.model_json)
            results = analyzer.analyze(
                seqlen=args.seq_len,
                batchsize=args.batch_size,
                w_bit=args.w_bit,
                a_bit=args.a_bit
            )

            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"Analysis saved to: {args.output}")
        else:
            # Default: Inspection mode using model_id
            analyzer = ModelAnalyzer(model_id=args.model_id)
            analyzer.load_model_architecture()

            if args.generate_arch:
                # Generate architecture using enhanced model string
                from model_architecture_agent import ModelArchitectureAgent

                # Get enhanced model structure (with full class paths)
                enhanced_structure = analyzer.get_enhanced_model_str()

                # Use agent to generate architecture
                agent = ModelArchitectureAgent()
                architecture = agent.generate_architecture_with_agent(
                    model_id=args.model_id,
                    model_structure=enhanced_structure
                )

                # Save to models/ directory
                os.makedirs("models", exist_ok=True)
                output_path = f"models/{args.model_id.replace('/', '-')}.json"
                with open(output_path, 'w') as f:
                    json.dump(architecture, f, indent=2)
                print(f"✓ Architecture saved to {output_path}")

            if args.ensure_modules:
                # Ensure all modules needed by architecture are analyzed and generated
                from module_analyzer import ModuleAnalyzer as ModuleAnalyzerCore

                # Load architecture JSON from models/ directory
                arch_path = f"models/{args.model_id.replace('/', '-')}.json"
                if not os.path.exists(arch_path):
                    raise FileNotFoundError(
                        f"Architecture JSON not found: {arch_path}\n"
                        f"Run with --generate-arch first to create the architecture file."
                    )

                with open(arch_path, 'r') as f:
                    architecture_json = json.load(f)

                # Extract all basic layer classes
                basic_classes = extract_basic_layer_classes(architecture_json)
                print(f"\n=== Module Ensure ===")
                print(f"Found {len(basic_classes)} unique basic layer types in architecture")

                # Use ModuleAnalyzer to check and generate missing modules
                module_analyzer = ModuleAnalyzerCore()

                generated = []
                cached = []
                failed = []

                for class_name in sorted(basic_classes):
                    print(f"\nChecking module: {class_name}")

                    if module_analyzer.is_cached(class_name):
                        print(f"  ✓ Already in database")
                        cached.append(class_name)
                    else:
                        print(f"  → Analyzing with agent...")
                        try:
                            # This will analyze AND generate the module
                            module_analyzer.analyze_module(class_name, force=False, generate=True)
                            print(f"  ✓ Analysis and generation complete")
                            generated.append(class_name)
                        except Exception as e:
                            print(f"  ✗ Failed: {e}")
                            failed.append(class_name)

                # Summary
                print(f"\n=== Module Ensure Summary ===")
                print(f"Cached: {len(cached)}")
                print(f"Generated: {len(generated)}")
                if generated:
                    for cls in generated:
                        print(f"  - {cls}")
                print(f"Failed: {len(failed)}")
                if failed:
                    for cls in failed:
                        print(f"  - {cls}")

            if args.populate_arch:
                # Populate parameters for architecture JSON
                from populate_parameters_agent import PopulateParametersAgent

                # Load architecture JSON from models/ directory
                arch_path = f"models/{args.model_id.replace('/', '-')}.json"
                if not os.path.exists(arch_path):
                    raise FileNotFoundError(
                        f"Architecture JSON not found: {arch_path}\n"
                        f"Run with --generate-arch first to create the architecture file."
                    )

                with open(arch_path, 'r') as f:
                    architecture_json = json.load(f)

                # Get model config and enhanced structure (already in memory)
                model_config = analyzer.config.to_dict()
                enhanced_structure = analyzer.get_enhanced_model_str()

                # Populate architecture using agent
                # Runtime parameters (batch_size, seq_len, dtype_bytes) will be populated as templates
                agent = PopulateParametersAgent()
                populated_arch = agent.populate_architecture_with_agent(
                    architecture_json=architecture_json,
                    model_config=model_config,
                    enhanced_model_structure=enhanced_structure
                )

                # Save populated architecture
                output_path = f"models/{args.model_id.replace('/', '-')}_populated.json"
                with open(output_path, 'w') as f:
                    json.dump(populated_arch, f, indent=2)
                print(f"✓ Populated architecture saved to {output_path}")
                print(f"  Runtime parameters use template format: {{batch_size}}, {{seq_len}}, {{w_dtype_bytes}}, {{a_dtype_bytes}}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
