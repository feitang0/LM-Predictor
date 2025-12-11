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


def extract_model_params(config) -> Dict[str, Any]:
    """Extract standardized parameters from any HuggingFace config.

    Handles different naming conventions across model families:
    - GPT-2: n_embd, n_head, n_layer, n_inner
    - Llama/Mistral: hidden_size, num_attention_heads, num_hidden_layers
    - BERT: hidden_size, num_attention_heads, num_hidden_layers

    Args:
        config: HuggingFace model config object

    Returns:
        Dictionary with standardized parameter names
    """
    params = {}

    # hidden_size
    params['hidden_size'] = (
        getattr(config, 'hidden_size', None) or
        getattr(config, 'n_embd', None) or
        getattr(config, 'd_model', None)
    )

    # num_heads
    params['num_heads'] = (
        getattr(config, 'num_attention_heads', None) or
        getattr(config, 'n_head', None) or
        getattr(config, 'num_heads', None)
    )

    # num_layers
    params['num_layers'] = (
        getattr(config, 'num_hidden_layers', None) or
        getattr(config, 'n_layer', None) or
        getattr(config, 'num_layers', None)
    )

    # intermediate_size
    params['intermediate_size'] = (
        getattr(config, 'intermediate_size', None) or
        getattr(config, 'n_inner', None) or
        (params['hidden_size'] * 4 if params['hidden_size'] else None)
    )

    # vocab_size
    params['vocab_size'] = getattr(config, 'vocab_size', None)

    # head_dim (derived)
    if params['hidden_size'] and params['num_heads']:
        params['head_dim'] = params['hidden_size'] // params['num_heads']
    else:
        params['head_dim'] = None

    # has_bias (default True for most models)
    params['has_bias'] = not getattr(config, 'bias', True) == False

    return params


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
        print(self.model)
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

    def _populate_value(self, value: Any, params: Dict[str, Any], config: Any = None, is_formula: bool = False) -> Any:
        """Recursively populate template variables in a value and evaluate expressions.

        Args:
            value: The value to process (string, dict, list, or other)
            params: Dictionary of parameter name -> value mappings
            config: HuggingFace model config object for resolving {config.xxx} patterns
            is_formula: If True, this is a formula field that should be evaluated

        Returns:
            The value with all template variables substituted and expressions evaluated

        Raises:
            ValueError: If any template variable in a formula cannot be resolved
        """
        if isinstance(value, str):
            if not is_formula:
                # Non-formula string (like 'analysis', 'operation') - return as-is
                return value

            # Formula string - substitute and evaluate
            result = value

            # Handle {config.xxx} patterns first
            config_vars = re.findall(r'\{config\.(\w+)\}', result)
            for attr in config_vars:
                if config and hasattr(config, attr):
                    config_value = getattr(config, attr)
                    # Handle special case: n_inner defaults to 4 * hidden_size if None
                    if attr == 'n_inner' and config_value is None:
                        config_value = 4 * getattr(config, 'hidden_size', 0)
                    result = result.replace(f"{{config.{attr}}}", str(config_value))
                else:
                    raise ValueError(f"Config attribute not found: config.{attr}")

            # Handle regular params
            for name, val in params.items():
                result = result.replace(f"{{{name}}}", str(val))

            # Check for remaining unresolved variables
            remaining = re.findall(r'\{[\w.]+\}', result)
            if remaining:
                raise ValueError(f"Unresolved template variables: {remaining}")

            # Try to evaluate arithmetic expression
            try:
                evaluated = eval(result)
                # Return as int if it's a whole number, otherwise as-is
                if isinstance(evaluated, float) and evaluated.is_integer():
                    return int(evaluated)
                return evaluated
            except:
                return result  # Return as-is if not evaluatable (e.g., pure strings)

        elif isinstance(value, dict):
            result = {}
            for k, v in value.items():
                # Determine if this key contains a formula
                key_is_formula = k in ('flops', 'read', 'write')
                result[k] = self._populate_value(v, params, config=config, is_formula=key_is_formula)
            return result
        elif isinstance(value, list):
            return [self._populate_value(item, params, config=config, is_formula=is_formula) for item in value]
        else:
            return value

    def populate_template(
        self,
        model_json: Dict[str, Any],
        batch_size: int,
        seq_len: int,
        cache_len: int = 0,
        w_bytes: int = 2,
        a_bytes: int = 2,
        config: Any = None,
        config_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Populate template variables in model analysis JSON and evaluate to numbers.

        Args:
            model_json: Model analysis JSON with {variable} placeholders
            batch_size: Batch size
            seq_len: Sequence length
            cache_len: KV cache length (default 0 for prefill)
            w_bytes: Weight precision in bytes (default 2 for fp16)
            a_bytes: Activation precision in bytes (default 2 for fp16)
            config: HuggingFace model config object for resolving {config.xxx} patterns
            config_params: Additional params dict (legacy, for non-config variables)

        Returns:
            JSON with all formulas evaluated to numeric values

        Raises:
            ValueError: If any template variable cannot be resolved
        """
        # Build complete parameter dictionary
        params = {
            # Runtime parameters
            'batch_size': batch_size,
            'seq_len': seq_len,
            'cache_len': cache_len,
            'w_bytes': w_bytes,
            'a_bytes': a_bytes,
        }

        # Add additional parameters if provided
        if config_params:
            params.update(config_params)

        # Recursively populate and evaluate
        return self._populate_value(model_json, params, config=config)

    def flatten_kernels(self, kernel: Dict[str, Any], repeat_multiplier: int = 1, path: str = "", variables: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Recursively flatten nested kernel structure into a single-level list.

        Args:
            kernel: Kernel dict with kernel_type, operation, flops, memory_access, sub_kernels, repeat fields
            repeat_multiplier: Cumulative repeat count from parent kernels
            path: Hierarchical path string for debugging/tracing
            variables: Variable dict for evaluating repeat formulas (batch_size, seq_len, config.*, etc.)

        Returns:
            List of flattened kernel dicts with:
            - operation: str (operation description)
            - path: str (hierarchical trace)
            - flops_formula: str (original formula)
            - mem_read_formula: str (original formula)
            - mem_write_formula: str (original formula)
            - repeat: int (total repeat multiplier)
        """
        # Evaluate repeat field if it exists (may be template like "{config.num_hidden_layers}")
        repeat_count = 1
        if "repeat" in kernel:
            repeat_value = kernel["repeat"]
            if isinstance(repeat_value, str) and "{" in repeat_value:
                # Evaluate template variable
                if variables:
                    try:
                        repeat_count = int(self._populate_value(repeat_value, variables, config=variables.get('_config'), is_formula=True))
                    except Exception as e:
                        print(f"Warning: Could not evaluate repeat formula '{repeat_value}': {e}")
                        repeat_count = 1
                else:
                    # No variables provided, keep as-is (will be evaluated later)
                    repeat_count = repeat_value
            else:
                repeat_count = int(repeat_value)

        total_repeat = repeat_multiplier * repeat_count if isinstance(repeat_count, int) else repeat_multiplier

        # Build path string
        current_path = f"{path} > {kernel.get('operation', 'unknown')}" if path else kernel.get('operation', 'unknown')

        if kernel.get("kernel_type") == "basic":
            # Basic kernel - extract formulas and return as single item
            return [{
                "operation": kernel.get("operation", ""),
                "analysis": kernel.get("analysis", ""),
                "path": current_path,
                "flops_formula": kernel.get("flops", "0"),
                "mem_read_formula": kernel.get("memory_access", {}).get("read", "0"),
                "mem_write_formula": kernel.get("memory_access", {}).get("write", "0"),
                "repeat": total_repeat
            }]
        else:
            # Composite kernel - recursively flatten sub_kernels
            flat_list = []
            for sub_kernel in kernel.get("sub_kernels", []):
                flat_list.extend(self.flatten_kernels(sub_kernel, total_repeat, current_path, variables))
            return flat_list

    def analyze_kernel_json(
        self,
        kernel_json_path: str,
        model_id: str,
        batch_size: int,
        seq_len: int,
        cache_len: int = 0,
        w_bytes: int = 2,
        a_bytes: int = 2
    ) -> Dict[str, Any]:
        """Analyze kernel JSON file to calculate FLOPs and memory access.

        Args:
            kernel_json_path: Path to kernel JSON file (e.g., GPT2LMHeadModel.json)
            model_id: HuggingFace model ID for loading config
            batch_size: Batch size
            seq_len: Sequence length
            cache_len: KV cache length (0 for prefill, >0 for decode)
            w_bytes: Weight precision in bytes (2 for fp16, 4 for fp32)
            a_bytes: Activation precision in bytes

        Returns:
            Dictionary with:
            - kernels: List of evaluated kernels with numeric FLOPs/memory values
            - totals: Aggregate FLOPs and memory access
            - parameters: Runtime parameters used
        """
        print(f"\n=== Analyzing Kernel JSON: {kernel_json_path} ===")
        print(f"Model: {model_id}")
        print(f"Parameters: batch_size={batch_size}, seq_len={seq_len}, cache_len={cache_len}, w_bytes={w_bytes}, a_bytes={a_bytes}")

        # Load kernel JSON
        with open(kernel_json_path, 'r') as f:
            kernel_data = json.load(f)

        # Load model config for {config.xxx} resolution
        print(f"Loading model config from {model_id}...")
        hf_token = os.getenv('HUGGINGFACE_HUB_TOKEN')
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True, token=hf_token)
        print(f"✓ Loaded {config.__class__.__name__}")

        # Build variable dictionary for template substitution
        variables = {
            'batch_size': batch_size,
            'seq_len': seq_len,
            'cache_len': cache_len,
            'w_bytes': w_bytes,
            'a_bytes': a_bytes,
            '_config': config  # Store config for _populate_value
        }

        # Flatten kernels
        print(f"\nFlattening kernel hierarchy...")
        flat_kernels = []
        for kernel in kernel_data.get("kernels", []):
            flat_kernels.extend(self.flatten_kernels(kernel, repeat_multiplier=1, path="", variables=variables))

        print(f"✓ Flattened {len(flat_kernels)} basic kernels")

        # Evaluate formulas for each kernel
        print(f"\nEvaluating formulas...")
        results = []
        total_flops = 0
        total_mem_read = 0
        total_mem_write = 0

        for i, kernel in enumerate(flat_kernels):
            try:
                # Evaluate formulas
                flops = self._populate_value(kernel["flops_formula"], variables, config=config, is_formula=True)
                mem_read = self._populate_value(kernel["mem_read_formula"], variables, config=config, is_formula=True)
                mem_write = self._populate_value(kernel["mem_write_formula"], variables, config=config, is_formula=True)

                # Apply repeat multiplier
                repeat = kernel["repeat"]
                if isinstance(repeat, str):
                    # Evaluate if still a formula
                    repeat = int(self._populate_value(repeat, variables, config=config, is_formula=True))

                final_flops = flops * repeat
                final_mem_read = mem_read * repeat
                final_mem_write = mem_write * repeat

                # Accumulate totals
                total_flops += final_flops
                total_mem_read += final_mem_read
                total_mem_write += final_mem_write

                # Store result
                results.append({
                    "kernel_id": i,
                    "operation": kernel["operation"],
                    "path": kernel["path"],
                    "repeat": repeat,
                    "flops": final_flops,
                    "memory_read": final_mem_read,
                    "memory_write": final_mem_write,
                    "memory_total": final_mem_read + final_mem_write,
                    # Keep formulas for debugging
                    "flops_formula": kernel["flops_formula"],
                    "mem_read_formula": kernel["mem_read_formula"],
                    "mem_write_formula": kernel["mem_write_formula"]
                })

                # Print progress for significant operations
                if final_flops > 0 or final_mem_read + final_mem_write > 0:
                    def format_num(n):
                        if n >= 1e12: return f"{n/1e12:.2f}T"
                        elif n >= 1e9: return f"{n/1e9:.2f}G"
                        elif n >= 1e6: return f"{n/1e6:.2f}M"
                        elif n >= 1e3: return f"{n/1e3:.2f}K"
                        else: return f"{n}"

                    repeat_str = f" (×{repeat})" if repeat > 1 else ""
                    print(f"  [{i+1}/{len(flat_kernels)}] {kernel['operation'][:60]}...{repeat_str}")
                    print(f"      FLOPs={format_num(final_flops)}, Mem={format_num(final_mem_read + final_mem_write)}B")

            except Exception as e:
                print(f"  ⚠ Error evaluating kernel {i}: {kernel['operation'][:50]}... - {e}")
                results.append({
                    "kernel_id": i,
                    "operation": kernel["operation"],
                    "path": kernel["path"],
                    "error": str(e),
                    "flops_formula": kernel["flops_formula"],
                    "mem_read_formula": kernel["mem_read_formula"],
                    "mem_write_formula": kernel["mem_write_formula"]
                })

        # Summary
        print(f"\n=== Analysis Summary ===")
        print(f"Total Kernels Evaluated: {len(results)}")
        print(f"Total FLOPs: {total_flops:,}")
        print(f"Total Memory Read: {total_mem_read:,} bytes ({total_mem_read/1e9:.2f} GB)")
        print(f"Total Memory Write: {total_mem_write:,} bytes ({total_mem_write/1e9:.2f} GB)")
        print(f"Total Memory Access: {total_mem_read + total_mem_write:,} bytes ({(total_mem_read + total_mem_write)/1e9:.2f} GB)")

        return {
            "class_name": kernel_data.get("class_name", "unknown"),
            "parameters": {
                "batch_size": batch_size,
                "seq_len": seq_len,
                "cache_len": cache_len,
                "w_bytes": w_bytes,
                "a_bytes": a_bytes
            },
            "kernels": results,
            "totals": {
                "flops": total_flops,
                "memory_read": total_mem_read,
                "memory_write": total_mem_write,
                "memory_total": total_mem_read + total_mem_write
            }
        }

    def _extract_all_layers_with_parameters_from_json(self, layers, parent_path="", repeat_multiplier=1):
        """
        Extract all layers (basic and composite) that have parameters populated.

        Both basic layers (no sub_layers) and composite layers (with sub_layers)
        may have parameters if they perform computation. This includes:
        - Basic layers: Linear, Embedding, SiLU, RMSNorm, etc.
        - Composite layers with internal ops: LlamaDecoderLayer (residuals),
          LlamaMLP (element-wise multiply), LlamaSdpaAttention (attention matmuls)
        """
        layers_with_params = []

        for layer in layers:
            current_repeat = repeat_multiplier * layer.get("repeat", 1)
            current_path = f"{parent_path}.{layer['name']}" if parent_path else layer['name']

            # Include this layer if it has parameters (basic OR composite)
            if "parameters" in layer and layer["parameters"]:
                layers_with_params.append({
                    "name": layer["name"],
                    "class": layer["class"],
                    "parameters": layer["parameters"],
                    "path": current_path,
                    "repeat": current_repeat
                })

            # Always recurse into sub_layers if present
            if "sub_layers" in layer:
                layers_with_params.extend(
                    self._extract_all_layers_with_parameters_from_json(
                        layer["sub_layers"], current_path, current_repeat
                    )
                )

        return layers_with_params

    def _analyze_layer(self, layer_info, batch_size, seq_len, w_dtype_bytes, a_dtype_bytes):
        """Analyze a single layer (basic or composite) using registry."""
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

        # Extract all layers with parameters (basic + composite) from JSON
        layers_to_analyze = self._extract_all_layers_with_parameters_from_json(model_structure["layers"])

        print(f"\nAnalyzing {len(layers_to_analyze)} layers (basic + composite):")

        # Analyze each layer
        w_dtype_bytes = w_bit // 8
        a_dtype_bytes = a_bit // 8
        results = []

        for layer_info in layers_to_analyze:
            analysis = self._analyze_layer(layer_info, batchsize, seqlen, w_dtype_bytes, a_dtype_bytes)
            results.append(analysis)

            # Print progress with per-instance values and memory info
            if analysis["flops"] > 0:
                repeat_count = analysis['repeat_count']
                # Calculate per-instance values
                per_instance_flops = analysis['flops'] // repeat_count if repeat_count > 1 else analysis['flops']
                per_instance_reads = analysis['memory_reads'] // repeat_count if repeat_count > 1 else analysis['memory_reads']
                per_instance_writes = analysis['memory_writes'] // repeat_count if repeat_count > 1 else analysis['memory_writes']
                per_instance_intermediates = analysis['memory_intermediates'] // repeat_count if repeat_count > 1 else analysis['memory_intermediates']

                repeat_str = f" (×{repeat_count})" if repeat_count > 1 else ""

                # Format with human-readable units
                def format_bytes(b):
                    if b >= 1e9:
                        return f"{b/1e9:.2f}GB"
                    elif b >= 1e6:
                        return f"{b/1e6:.2f}MB"
                    elif b >= 1e3:
                        return f"{b/1e3:.2f}KB"
                    else:
                        return f"{b}B"

                def format_flops(f):
                    if f >= 1e12:
                        return f"{f/1e12:.2f}T"
                    elif f >= 1e9:
                        return f"{f/1e9:.2f}G"
                    elif f >= 1e6:
                        return f"{f/1e6:.2f}M"
                    elif f >= 1e3:
                        return f"{f/1e3:.2f}K"
                    else:
                        return f"{f}"

                print(f"  ✓ {analysis['layer_path']}{repeat_str}: "
                      f"FLOPs={format_flops(per_instance_flops)}, "
                      f"Reads={format_bytes(per_instance_reads)}, "
                      f"Writes={format_bytes(per_instance_writes)}, "
                      f"Intermediates={format_bytes(per_instance_intermediates)}")

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
                "layer_analyses": results,
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


def extract_all_layer_classes(architecture: Dict[str, Any]) -> set[str]:
    """
    Recursively extract ALL unique layer class names from architecture.
    Includes both basic layers (no sub_layers) and composite layers (with sub_layers).

    Args:
        architecture: Architecture JSON with layers structure

    Returns:
        Set of unique class names for all layers (basic + composite)
    """
    classes: set[str] = set()

    def traverse(layer: Dict[str, Any]) -> None:
        # Add this layer's class (whether basic or composite)
        class_name = layer.get("class", "")
        if class_name:
            classes.add(class_name)

        # Recurse into sub_layers if present
        if "sub_layers" in layer:
            for sub_layer in layer.get("sub_layers", []):
                traverse(sub_layer)

    for layer in architecture.get("layers", []):
        traverse(layer)

    return classes


def find_layer_by_class(architecture: Dict[str, Any], class_name: str) -> Optional[Dict[str, Any]]:
    """
    Find first layer in architecture matching the given class name.

    Args:
        architecture: Architecture JSON with layers structure
        class_name: Full class name to search for

    Returns:
        Layer dictionary if found, None otherwise
    """
    def traverse(layer: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if layer.get("class") == class_name:
            return layer
        # Recurse into sub_layers if present
        for sub_layer in layer.get("sub_layers", []):
            result = traverse(sub_layer)
            if result:
                return result
        return None

    for layer in architecture.get("layers", []):
        result = traverse(layer)
        if result:
            return result
    return None


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
    parser.add_argument('--populate', action='store_true',
                       help='Populate template variables in model JSON and evaluate formulas')
    parser.add_argument('--cache_len', type=int, default=0,
                       help='KV cache length for decode phase (default: 0 for prefill)')
    parser.add_argument('--analyze-kernels', type=str, default=None,
                       help='Analyze kernel JSON file to calculate FLOPs and memory (provide path to kernel JSON)')

    args = parser.parse_args()

    try:
        if args.analyze_kernels:
            # Kernel analysis mode - requires model_id and kernel JSON path
            if not args.model_id:
                print("Error: --analyze-kernels requires --model_id")
                return 1

            analyzer = ModelAnalyzer()
            results = analyzer.analyze_kernel_json(
                kernel_json_path=args.analyze_kernels,
                model_id=args.model_id,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                cache_len=args.cache_len,
                w_bytes=args.w_bit // 8,
                a_bytes=args.a_bit // 8
            )

            # Save results if output specified
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"\n✓ Results saved to: {args.output}")
            else:
                # Auto-generate output filename
                import os.path
                base_name = os.path.basename(args.analyze_kernels).replace('.json', '')
                output_path = f"{base_name}_analysis_B{args.batch_size}_S{args.seq_len}_C{args.cache_len}.json"
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"\n✓ Results saved to: {output_path}")

        elif args.analyze:
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

        elif args.populate:
            # Populate template mode - requires model_json
            if not args.model_json:
                print("Error: --populate requires --model_json")
                return 1

            # Load model JSON
            with open(args.model_json, 'r') as f:
                model_json = json.load(f)

            # Load config if model_id provided
            config = None
            if args.model_id:
                print(f"Loading model config: {args.model_id}")
                analyzer = ModelAnalyzer(model_id=args.model_id)
                analyzer.load_model_architecture()
                config = analyzer.config
                print(f"Loaded config: {config.__class__.__name__}")

            # Create analyzer and populate template
            analyzer = ModelAnalyzer(model_json_path=args.model_json)
            populated = analyzer.populate_template(
                model_json=model_json,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                cache_len=args.cache_len,
                w_bytes=args.w_bit // 8,
                a_bytes=args.a_bit // 8,
                config=config
            )

            # Output
            output_path = args.output or args.model_json.replace('.json', '_populated.json')
            with open(output_path, 'w') as f:
                json.dump(populated, f, indent=2)
            print(f"Populated JSON saved to: {output_path}")

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

                # Extract all layer classes (both basic and composite)
                all_classes = extract_all_layer_classes(architecture_json)
                print(f"\n=== Module Ensure ===")
                print(f"Found {len(all_classes)} unique layer types (basic + composite)")

                # Use ModuleAnalyzer to check and generate missing modules
                module_analyzer = ModuleAnalyzerCore()

                generated = []
                cached = []
                failed = []

                for class_name in sorted(all_classes):
                    print(f"\nChecking module: {class_name}")

                    # Look up if this module has sub_layers in architecture
                    layer_info = find_layer_by_class(architecture_json, class_name)
                    sub_layers = layer_info.get('sub_layers', []) if layer_info else []

                    if module_analyzer.is_cached(class_name):
                        print(f"  ✓ Already in database")
                        cached.append(class_name)
                    else:
                        print(f"  → Analyzing with agent...")
                        try:
                            # This will analyze AND generate the module
                            # Pass sub_layers context for composite module detection
                            module_analyzer.analyze_module(class_name, force=False, generate=True, sub_layers=sub_layers)
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
