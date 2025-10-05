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
        """Load a specific module class by key using naming from module_db.json."""
        try:
            # Read module data from database to get naming information
            with open(self._module_db_path, 'r') as f:
                db = json.load(f)

            module_data = db.get("modules", {}).get(module_key, {})

            # Try to read stored naming (new approach)
            file_name = module_data.get("generated_file_name")
            class_name = module_data.get("generated_class_name")

            # Fallback to derivation for legacy entries without naming fields
            if not file_name or not class_name:
                print(f"Warning: Module {module_key} missing naming fields, using fallback derivation")
                # Simple derivation: lowercase for file, capitalize parts for class
                file_name = module_key.lower() + ".py"
                parts = module_key.split("_")
                class_name = "".join(part.capitalize() for part in parts)

            # Remove .py extension if present
            if file_name.endswith(".py"):
                file_name = file_name[:-3]

            module_path = f"generated_modules.{file_name}"

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

    def get_required_parameters(self, module_name: str) -> Dict[str, str]:
        """Convenience method to get required parameters for a module."""
        module_class = self.get_module(module_name)
        if not module_class:
            raise ValueError(f"Module not found: {module_name}")

        instance = module_class()
        return instance.get_required_parameters()


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


def get_required_parameters(module_name: str) -> Dict[str, str]:
    """Convenience function to get required parameters."""
    return get_registry().get_required_parameters(module_name)