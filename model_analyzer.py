#!/usr/bin/env python3
"""
Modular model analyzer using meta device inspection and module database.
Replaces legacy hardcoded approach with composable analysis system.
"""

import os
import json
import argparse
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, AutoConfig

from module_analyzer import ModuleAnalyzer as ModuleDB
from module_analyzer_agent import ModuleAnalyzerAgent

load_dotenv()


@dataclass
class AnalysisParams:
    """Parameters for model analysis."""
    batch_size: int
    seq_len: int
    dtype_bytes: int = 2  # FP16 default


class ModelAnalyzer:
    """Modular model analyzer using meta device and module database."""

    def __init__(self, model_id: str):
        """Initialize analyzer with model ID."""
        self.model_id = model_id
        self.module_analyzer = ModuleDB()
        self.agent = ModuleAnalyzerAgent()
        self.model = None
        self.config = None

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

    def inspect_model_structure(self) -> Dict[str, Any]:
        """Inspect model structure using simple module representation."""
        if self.model is None:
            self.load_model_architecture()

        def module_to_dict(module):
            children = dict(module.named_children())
            if not children:  # leaf node
                return self.module_repr(module)
            return {name: module_to_dict(child) for name, child in children.items()}

        return module_to_dict(self.model)

    def module_repr(self, module) -> Dict[str, Any]:
        """Simple module representation for leaf nodes."""
        return {
            "class_name": f"{module.__class__.__module__}.{module.__class__.__name__}",
            "repr": str(module)
        }

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
        import re
        enhanced_str = model_str
        for simple_name, full_name in sorted_mappings:
            # Use regex with word boundary to ensure exact class name matches
            pattern = r'\b' + re.escape(simple_name) + r'\('
            replacement = full_name + "("
            enhanced_str = re.sub(pattern, replacement, enhanced_str)

        print(enhanced_str)


    def analyze_module(self, module_info: Dict[str, Any], params: AnalysisParams) -> Dict[str, Any]:
        """Analyze a single module using database/agent."""
        full_class_name = module_info["full_class_name"]
        module_params = module_info["parameters"]

        # Prepare analysis parameters
        analysis_params = {
            "B": params.batch_size,
            "S": params.seq_len,
            "dtype_bytes": params.dtype_bytes,
            **module_params
        }

        try:
            # Try module database first (cache-first approach)
            results = self.module_analyzer.analyze_module(full_class_name, analysis_params)

            if results is None:
                print(f"  Module {full_class_name} not in database, using agent...")
                # Use agent for unknown modules
                agent_results = self.agent.analyze_module_with_agent(full_class_name)

                # Store in database for future use
                self.module_analyzer.store_module_analysis(full_class_name, agent_results)

                # Compute with parameters
                results = self.module_analyzer.analyze_module(full_class_name, analysis_params)

            return {
                "module_path": module_info["path"],
                "class_name": module_info["class_name"],
                "full_class_name": full_class_name,
                "flops": results["flops"] if results else 0,
                "memory_reads": results["memory"]["reads"] if results else 0,
                "memory_writes": results["memory"]["writes"] if results else 0,
                "memory_intermediates": results["memory"]["intermediates"] if results else 0,
                "parameters_used": analysis_params
            }

        except Exception as e:
            print(f"  ⚠ Failed to analyze {full_class_name}: {e}")
            return {
                "module_path": module_info["path"],
                "class_name": module_info["class_name"],
                "full_class_name": full_class_name,
                "flops": 0,
                "memory_reads": 0,
                "memory_writes": 0,
                "memory_intermediates": 0,
                "error": str(e)
            }

    def analyze_model_recursive(self, module_info: Dict[str, Any], params: AnalysisParams,
                               results: List[Dict[str, Any]]) -> None:
        """Recursively analyze model structure."""
        # Analyze current module if it's a leaf or known composite module
        if not module_info["children"] or self._is_analyzable_module(module_info):
            analysis = self.analyze_module(module_info, params)
            results.append(analysis)
            print(f"  ✓ {analysis['module_path']}: {analysis['flops']:,} FLOPs")

        # Recurse into children for non-analyzable modules
        if module_info["children"] and not self._is_analyzable_module(module_info):
            for _, child_info in module_info["children"].items():
                self.analyze_model_recursive(child_info, params, results)

    def _is_analyzable_module(self, module_info: Dict[str, Any]) -> bool:
        """Check if module should be analyzed as a unit vs decomposed."""
        class_name = module_info["class_name"]

        # Always analyze these as atomic units
        atomic_modules = {
            "Linear", "Conv1d", "Conv2d", "LayerNorm", "RMSNorm",
            "MultiheadAttention", "Embedding"
        }

        # LLM-specific modules that should be analyzed as units
        llm_modules = {
            "LlamaAttention", "LlamaMLP", "LlamaDecoderLayer",
            "GPTAttention", "GPTMLP", "GPTBlock",
            "BertAttention", "BertMLP", "BertLayer"
        }

        return class_name in atomic_modules or class_name in llm_modules

    def analyze(self, seqlen: int, batchsize: int, w_bit: int = 16, a_bit: int = 16,
                kv_bit: Optional[int] = None, **_) -> Dict[str, Any]:
        """
        Main analysis method for compatibility with legacy interface.

        Args:
            seqlen: sequence length
            batchsize: batch size
            w_bit: weight bit width (for dtype_bytes calculation)
            a_bit: activation bit width (for dtype_bytes calculation)
            kv_bit: key-value cache bit width (unused in new approach)
            **kwargs: other legacy parameters (ignored)

        Returns:
            Analysis results in legacy format
        """
        print(f"\n=== Analyzing {self.model_id} ===")
        print(f"Parameters: B={batchsize}, S={seqlen}, w_bit={w_bit}, a_bit={a_bit}")

        # Calculate dtype bytes from bit width (kv_bit unused in new approach)
        if kv_bit is None:
            kv_bit = a_bit
        dtype_bytes = a_bit // 8

        # Load architecture if needed
        if self.model is None:
            self.load_model_architecture()

        # Get model structure
        structure = self.inspect_model_structure()

        # Analyze recursively
        params = AnalysisParams(batchsize, seqlen, dtype_bytes)
        results = []

        print(f"\nAnalyzing modules:")
        self.analyze_model_recursive(structure, params, results)

        # Convert to legacy format for compatibility
        total_flops = sum(r["flops"] for r in results)
        total_memory_reads = sum(r["memory_reads"] for r in results)
        total_memory_writes = sum(r["memory_writes"] for r in results)
        total_memory_intermediates = sum(r["memory_intermediates"] for r in results)

        # Legacy format - single stage results
        legacy_results = {
            "decode": {},
            "prefill": {},
            "total_results": {
                "decode": {
                    "OPs": total_flops,
                    "memory_access": total_memory_reads + total_memory_writes + total_memory_intermediates,
                    "load_weight": total_memory_reads,
                    "load_act": 0,  # Simplified
                    "store_act": total_memory_writes,
                    "load_kv_cache": 0,  # Simplified
                    "store_kv_cache": 0,  # Simplified
                    "inference_time": total_flops / 1e12  # Simplified estimate
                },
                "prefill": {
                    "OPs": total_flops,
                    "memory_access": total_memory_reads + total_memory_writes + total_memory_intermediates,
                    "load_weight": total_memory_reads,
                    "load_act": 0,  # Simplified
                    "store_act": total_memory_writes,
                    "load_kv_cache": 0,  # Simplified
                    "store_kv_cache": 0,  # Simplified
                    "inference_time": total_flops / 1e12  # Simplified estimate
                }
            },
            "modular_results": {
                "module_analyses": results,
                "totals": {
                    "flops": total_flops,
                    "memory_reads": total_memory_reads,
                    "memory_writes": total_memory_writes,
                    "memory_intermediates": total_memory_intermediates
                },
                "model_structure": structure
            }
        }

        print(f"\n=== Analysis Summary ===")
        print(f"Total FLOPs: {total_flops:,}")
        print(f"Memory Reads: {total_memory_reads:,} bytes")
        print(f"Memory Writes: {total_memory_writes:,} bytes")
        print(f"Memory Intermediates: {total_memory_intermediates:,} bytes")

        return legacy_results


def main():
    """Command-line interface for model analysis."""
    parser = argparse.ArgumentParser(description='Analyze LLM model FLOPs and memory using modular approach')
    parser.add_argument('--model_id', type=str, default="meta-llama/Llama-2-7b-hf",
                       help='Model ID to analyze (default: meta-llama/Llama-2-7b-hf)')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size (default: 1)')
    parser.add_argument('--seq_len', type=int, default=2048,
                       help='Sequence length (default: 2048)')
    parser.add_argument('--analyze', action='store_true',
                       help='Run analysis (default: True, kept for compatibility)')
    parser.add_argument('--w_bit', type=int, default=16,
                       help='Weight bit width (default: 16)')
    parser.add_argument('--a_bit', type=int, default=16,
                       help='Activation bit width (default: 16)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output file path (default: auto-generated)')
    parser.add_argument('--inspect_only', action='store_true',
                       help='Only inspect model structure without analysis')
    parser.add_argument('--unique_modules', action='store_true',
                       help='Output unique module classes found in the model')

    args = parser.parse_args()

    try:
        analyzer = ModelAnalyzer(args.model_id)

        if args.inspect_only:
            analyzer.load_model_architecture()
            analyzer.print_enhanced_model()
            # structure = analyzer.inspect_model_structure()
            # print(json.dumps(structure, indent=2))
        elif args.unique_modules:
            analyzer.load_model_architecture()
            unique_modules = analyzer.collect_module_classes(analyzer.model)
            for full_name in sorted(set(unique_modules.values())):
                print(full_name)
        else:
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

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
