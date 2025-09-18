#!/usr/bin/env python3
"""
Cache-first module analysis with Claude Code agent fallback.
"""

import json
import os
from typing import Dict, Any, Optional
from module_analyzer_agent import ModuleAnalyzerAgent


def extract_library_and_class(full_class_name: str) -> tuple[str, str]:
    """Extract library name and clean class name from full class path.

    Args:
        full_class_name: e.g. "transformers.models.llama.modeling_llama.LlamaMLP"

    Returns:
        tuple: (library_name, clean_class_name) e.g. ("transformers", "LlamaMLP")
    """
    parts = full_class_name.split('.')

    # Extract library (first part)
    library = parts[0]

    # Extract class name (last part)
    class_name = parts[-1]

    return library, class_name


def full_class_name_to_key(full_class_name: str) -> str:
    """Convert full class name to standardized database key.

    Args:
        full_class_name: e.g. "transformers.models.llama.modeling_llama.LlamaMLP"

    Returns:
        str: standardized key e.g. "transformers_LlamaMLP"
    """
    library, class_name = extract_library_and_class(full_class_name)
    return f"{library}_{class_name}"


class ModuleAnalyzer:
    """Simple cache-first module analyzer."""

    def __init__(self, db_path: str = "module_db.json"):
        """Initialize with database path."""
        self.db_path = db_path
        self.agent = ModuleAnalyzerAgent()

    def _load_db(self) -> Dict[str, Any]:
        """Load database or create empty one."""
        if not os.path.exists(self.db_path):
            return {"version": "1.0", "modules": {}}

        with open(self.db_path, 'r') as f:
            return json.load(f)

    def _save_db(self, db: Dict[str, Any]) -> None:
        """Save database to file."""
        with open(self.db_path, 'w') as f:
            json.dump(db, f, indent=2)

    def _module_to_key(self, analysis_result: Dict[str, Any]) -> str:
        """Convert analysis result to standardized database key.

        Args:
            analysis_result: Analysis result dict containing full_class_name

        Returns:
            str: standardized database key (e.g., "transformers_LlamaMLP")
        """
        full_class_name = analysis_result.get("full_class_name")
        if not full_class_name:
            raise ValueError("Analysis result must contain 'full_class_name'")

        return full_class_name_to_key(full_class_name)

    def is_cached(self, module_spec: str) -> bool:
        """Check if module is in cache by searching for matching class names."""
        db = self._load_db()
        modules = db.get("modules", {})

        # Search for module_spec in both keys and full_class_names
        for key, module_data in modules.items():
            full_class_name = module_data.get("full_class_name", "")

            # Check if module_spec matches the key or is the class name from full_class_name
            if (key == module_spec or
                module_spec in full_class_name or
                full_class_name.endswith(f".{module_spec}")):
                return True

        return False

    def get_cached(self, module_spec: str) -> Optional[Dict[str, Any]]:
        """Get cached module analysis by searching for matching class names."""
        db = self._load_db()
        modules = db.get("modules", {})

        # Search for module_spec in both keys and full_class_names
        for key, module_data in modules.items():
            full_class_name = module_data.get("full_class_name", "")

            # Check if module_spec matches the key or is the class name from full_class_name
            if (key == module_spec or
                module_spec in full_class_name or
                full_class_name.endswith(f".{module_spec}")):
                return module_data

        return None

    def analyze_module(self, module_spec: str) -> Dict[str, Any]:
        """Analyze module (cache-first)."""
        print(f"Analyzing {module_spec}...")

        # Check cache first
        cached = self.get_cached(module_spec)
        if cached:
            print(f"✓ Found in cache")
            return cached

        # Use agent
        print(f"⚡ Not cached, using agent...")
        result = self.agent.analyze_module_with_agent(module_spec)

        # Cache result using standardized key
        db = self._load_db()
        key = self._module_to_key(result)
        db["modules"][key] = result
        self._save_db(db)

        print(f"✓ Cached as {key}")
        return result


def main():
    """CLI with argparse."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Cache-first module analysis with Claude Code agent fallback"
    )
    parser.add_argument(
        "module_spec",
        help="Module specification to analyze (e.g., LlamaMLP, torch.nn.Linear)"
    )
    parser.add_argument(
        "--db-path",
        default="module_db.json",
        help="Path to module database file (default: module_db.json)"
    )

    args = parser.parse_args()

    analyzer = ModuleAnalyzer(db_path=args.db_path)

    try:
        result = analyzer.analyze_module(args.module_spec)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())