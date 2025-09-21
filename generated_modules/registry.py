"""Module registry for auto-generated FLOP/memory calculators."""

import importlib
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Type
from .base import BaseModule


class ModuleRegistry:
    """Registry for auto-discovering and managing generated module classes."""

    def __init__(self):
        self._modules: Dict[str, Type[BaseModule]] = {}
        self._module_db_path = Path(__file__).parent.parent / "module_db.json"
        self._load_modules()

    def _load_modules(self) -> None:
        """Auto-discover and load generated module classes."""
        if not self._module_db_path.exists():
            return

        with open(self._module_db_path, 'r') as f:
            db = json.load(f)

        for module_key, module_data in db.get("modules", {}).items():
            full_class_name = module_data.get("full_class_name", "")
            module_class = self._load_module_class(module_key, full_class_name)
            if module_class:
                self._modules[module_key] = module_class
                # Also register by full class name for convenience
                self._modules[full_class_name] = module_class

    def _load_module_class(self, module_key: str, full_class_name: str) -> Optional[Type[BaseModule]]:
        """Load a specific module class by key using manual mapping."""
        try:
            # Manual mapping for reliable module loading
            # Format: module_key -> (filename, class_name)
            module_mapping = {
                "transformers_LlamaMLP": ("transformers_llama_mlp", "TransformersLlamaMLP"),
                "torch_SiLU": ("torch_silu", "TorchSiLU"),
                "torch_ModuleList": ("torch_module_list", "TorchModuleList"),
                "torch_Linear": ("torch_linear", "TorchLinear"),
                "torch_Embedding": ("torch_embedding", "TorchEmbedding"),
                "transformers_LlamaDecoderLayer": ("transformers_llama_decoder_layer", "TransformersLlamaDecoderLayer"),
                "transformers_LlamaForCausalLM": ("transformers_llama_for_causal_lm", "TransformersLlamaForCausalLM"),
                "transformers_LlamaModel": ("transformers_llama_model", "TransformersLlamaModel"),
                "transformers_LlamaRMSNorm": ("transformers_llama_rms_norm", "TransformersLlamaRMSNorm"),
                "transformers_LlamaRotaryEmbedding": ("transformers_llama_rotary_embedding", "TransformersLlamaRotaryEmbedding"),
                "transformers_LlamaSdpaAttention": ("transformers_llama_sdpa_attention", "TransformersLlamaSdpaAttention"),
            }

            if module_key not in module_mapping:
                return None

            module_file, class_name = module_mapping[module_key]
            module_path = f"generated_modules.{module_file}"

            # Import the module and get the class
            module = importlib.import_module(module_path)
            return getattr(module, class_name, None)

        except (ImportError, AttributeError) as e:
            print(f"Warning: Could not load module {module_key}: {e}")
            return None

    def get_module(self, module_name: str) -> Optional[Type[BaseModule]]:
        """Get a module class by name or full class name."""
        return self._modules.get(module_name)

    def list_modules(self) -> Dict[str, str]:
        """List all registered modules with their full class names."""
        result = {}
        with open(self._module_db_path, 'r') as f:
            db = json.load(f)

        for module_key, module_data in db.get("modules", {}).items():
            if module_key in self._modules:
                result[module_key] = module_data.get("full_class_name", "")

        return result

    def compute_flops(self, module_name: str, **params: Dict[str, Any]) -> int:
        """Convenience method to compute FLOPs for a module."""
        module_class = self.get_module(module_name)
        if not module_class:
            raise ValueError(f"Module not found: {module_name}")

        instance = module_class()
        instance.validate_parameters(**params)
        return instance.compute_flops(**params)

    def compute_memory(self, module_name: str, **params: Dict[str, Any]) -> Dict[str, int]:
        """Convenience method to compute memory usage for a module."""
        module_class = self.get_module(module_name)
        if not module_class:
            raise ValueError(f"Module not found: {module_name}")

        instance = module_class()
        instance.validate_parameters(**params)
        return {
            "reads": instance.compute_memory_reads(**params),
            "writes": instance.compute_memory_writes(**params),
            "intermediates": instance.compute_intermediates(**params)
        }


# Global registry instance
_registry = None


def get_registry() -> ModuleRegistry:
    """Get the global module registry instance."""
    global _registry
    if _registry is None:
        _registry = ModuleRegistry()
    return _registry


def compute_flops(module_name: str, **params: Dict[str, Any]) -> int:
    """Convenience function to compute FLOPs."""
    return get_registry().compute_flops(module_name, **params)


def compute_memory(module_name: str, **params: Dict[str, Any]) -> Dict[str, int]:
    """Convenience function to compute memory usage."""
    return get_registry().compute_memory(module_name, **params)