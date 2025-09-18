"""Base module for auto-generated FLOP/memory calculators."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Union
import json
import re


class BaseModule(ABC):
    """Abstract base class for auto-generated FLOP/memory calculator modules."""

    @abstractmethod
    def get_required_parameters(self) -> Dict[str, str]:
        """Return parameter names mapped to their types."""
        pass

    @abstractmethod
    def compute_flops(self, **params: Dict[str, Any]) -> int:
        """Calculate FLOPs using agent-analyzed formula."""
        pass

    @abstractmethod
    def compute_memory_reads(self, **params: Dict[str, Any]) -> int:
        """Calculate memory reads using agent-analyzed formula."""
        pass

    @abstractmethod
    def compute_memory_writes(self, **params: Dict[str, Any]) -> int:
        """Calculate memory writes using agent-analyzed formula."""
        pass

    @abstractmethod
    def compute_intermediates(self, **params: Dict[str, Any]) -> int:
        """Calculate intermediate memory using agent-analyzed formula."""
        pass

    def _evaluate_formula(self, formula: str, params: Dict[str, Any], registry=None) -> int:
        """
        Evaluate a formula template by substituting parameters and resolving module calls.

        Args:
            formula: Formula template with ${param} and {Module}() syntax
            params: Parameter values to substitute
            registry: Module registry for resolving dependencies

        Returns:
            Calculated result as integer
        """
        # Substitute parameters ${param_name}
        result_formula = formula
        for param_name, param_value in params.items():
            pattern = f"${{\\s*{re.escape(param_name)}\\s*}}"
            result_formula = re.sub(pattern, str(param_value), result_formula)

        # Handle module calls {ModuleName}(...) if registry provided
        if registry:
            # This would resolve module dependencies - for now just evaluate the expression
            pass

        # For now, evaluate as simple mathematical expression
        # In production, this should have proper expression parsing and validation
        try:
            # Use eval safely for mathematical expressions only
            # TODO: Replace with proper expression parser for security
            result = eval(result_formula)
            return int(result)
        except Exception as e:
            raise ValueError(f"Failed to evaluate formula '{result_formula}': {e}")

    def validate_parameters(self, **params: Dict[str, Any]) -> None:
        """Validate that all required parameters are provided with correct types."""
        required = self.get_required_parameters()

        for param_name, param_type in required.items():
            if param_name not in params:
                raise ValueError(f"Missing required parameter: {param_name}")

            value = params[param_name]
            if param_type == "int" and not isinstance(value, int):
                raise TypeError(f"Parameter {param_name} must be int, got {type(value)}")