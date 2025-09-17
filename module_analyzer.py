#!/usr/bin/env python3
"""
Cache-first module analysis with Claude Code agent fallback.
"""

import json
import os
from typing import Dict, Any, Optional
from module_analyzer_agent import ModuleAnalyzerAgent


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

    def _module_to_key(self, module_path: str) -> str:
        """Convert module path to database key."""
        return module_path.replace(".", "__")

    def is_cached(self, module_path: str) -> bool:
        """Check if module is in cache."""
        db = self._load_db()
        key = self._module_to_key(module_path)
        return key in db.get("modules", {})

    def get_cached(self, module_path: str) -> Optional[Dict[str, Any]]:
        """Get cached module analysis."""
        if not self.is_cached(module_path):
            return None

        db = self._load_db()
        key = self._module_to_key(module_path)
        return db["modules"][key]

    def analyze_module(self, module_path: str) -> Dict[str, Any]:
        """Analyze module (cache-first)."""
        print(f"Analyzing {module_path}...")

        # Check cache first
        cached = self.get_cached(module_path)
        if cached:
            print(f"✓ Found in cache")
            return cached

        # Use agent
        print(f"⚡ Not cached, using agent...")
        result = self.agent.analyze_module_with_agent(module_path)

        # Cache result
        db = self._load_db()
        key = self._module_to_key(module_path)
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
        "module_path",
        help="Module path to analyze (e.g., torch.nn.Linear)"
    )
    parser.add_argument(
        "--db-path",
        default="module_db.json",
        help="Path to module database file (default: module_db.json)"
    )

    args = parser.parse_args()

    analyzer = ModuleAnalyzer(db_path=args.db_path)

    try:
        result = analyzer.analyze_module(args.module_path)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())