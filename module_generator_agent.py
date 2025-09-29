#!/usr/bin/env python3
"""
Claude Code Agent for generating Python module files from analysis responses.
This module uses Claude Code in headless mode to create callable module classes.
"""

import json
import os
import subprocess
import sys
import argparse
from typing import Dict, Any
from pathlib import Path
from dotenv import dotenv_values


class ModuleGeneratorAgent:
    """Claude Code agent for generating Python module files from analysis responses."""

    def __init__(self, module_db_file: str = "module_db.json"):
        """Initialize the agent with module database file."""
        self.claude_command = "claude"
        self.module_db_file = module_db_file

    def query_module_database(self, full_class_name: str) -> Dict[str, Any] | None:
        """
        Query the module database for a specific full class name.

        Args:
            full_class_name: The full class name to search for (e.g., 'transformers.models.llama.modeling_llama.LlamaMLP')

        Returns:
            Module data dictionary if found, None if not found
        """
        try:
            if not os.path.exists(self.module_db_file):
                return None

            with open(self.module_db_file, 'r') as f:
                db = json.load(f)

            # Search through all modules in the database
            for module_key, module_data in db.get("modules", {}).items():
                if module_data.get("full_class_name") == full_class_name:
                    return module_data

            return None
        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            print(f"Error reading module database: {e}")
            return None

    def generate_module_with_agent(self, full_class_name: str) -> Dict[str, str]:
        """
        Use Claude Code to generate a Python module file from module database.

        Args:
            full_class_name: Full class name to generate module for (e.g., 'transformers.models.llama.modeling_llama.LlamaMLP')

        Returns:
            Dictionary with generation results including file paths
        """
        # Query the module database first
        module_data = self.query_module_database(full_class_name)

        if module_data is None:
            return {
                "status": "error",
                "error": f"Module database does not contain '{full_class_name}'"
            }

        prompt = self._create_generation_prompt(module_data)

        print("=== PROMPT ===")
        print(prompt)
        print("=== END PROMPT ===")

        try:
            # Remove generation_result.json if it exists
            if os.path.exists("generation_result.json"):
                os.remove("generation_result.json")

            # Remove diagnostic file if it exists
            if os.path.exists("generation_result_diagnostics.json"):
                os.remove("generation_result_diagnostics.json")

            # Prepare environment (current env + any from .env file)
            env = os.environ.copy()

            # Load .env variables only for this subprocess
            env_vars = dotenv_values(".env")
            env.update(env_vars)

            # Run Claude Code in headless mode
            result = subprocess.run([
                self.claude_command, "-p", prompt,
                "--dangerously-skip-permissions"
            ], capture_output=True, text=True, timeout=600, env=env)

            if result.returncode != 0:
                raise RuntimeError(f"Claude Code failed: {result.stderr}")

            print("=== CLAUDE RESPONSE ===")
            print(result.stdout)
            print("=== END CLAUDE RESPONSE ===")

            # Read generation result
            if not os.path.exists("generation_result.json"):
                raise FileNotFoundError("Claude did not create generation_result.json file")

            with open("generation_result.json", 'r') as f:
                generation_result = json.load(f)

            # Read diagnostic information (for debugging purposes)
            diagnostic_file = "generation_result_diagnostics.json"
            if os.path.exists(diagnostic_file):
                with open(diagnostic_file, 'r') as f:
                    diagnostics = json.load(f)
                print(f"Diagnostics: {diagnostics}")

            return generation_result

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Claude Code generation timed out for {full_class_name}")
        except json.JSONDecodeError as e:
            print(f"Claude stdout: {result.stdout}")
            print(f"Claude stderr: {result.stderr}")
            raise RuntimeError(f"Failed to parse Claude response: {e}")
        except Exception as e:
            raise RuntimeError(f"Generation failed for {full_class_name}: {e}")

    def _create_generation_prompt(self, module_data: Dict[str, Any]) -> str:
        """Create the prompt for Claude Code to generate module files."""
        full_class_name = module_data.get("full_class_name", "unknown")
        return f"""
ultrathink: Generate Python module file from agent analysis response

GOAL: Create a callable Python module file from the agent analysis response using flat directory structure

Available Resources:
- Module analysis data for {full_class_name} with FLOP/memory formulas
- generated_modules/ - Directory for generated modules (flat structure)
- generated_modules/base.py - BaseModule abstract class
- DESIGN.md - System architecture reference

Module Data:
{json.dumps(module_data, indent=2)}

IMPORTANT:
1. Follow the naming conventions from updated DESIGN.md exactly
2. Generate real executable Python code, NOT template formulas
3. Use new syntax: ${{}} for modules, {{}} for parameters
4. Write generation result to generation_result.json (NOT as final step)
5. **FINAL STEP**: Write diagnostic information to generation_result_diagnostics.json

TODO List:

ultrathink 1. **Process Data**: Examine the provided module analysis data for {full_class_name}

ultrathink 2. **Determine Paths**: Convert full_class_name to required paths (flat structure):
   - For torch.nn.Linear: class_name="TorchLinear", file_name="torch_linear.py"
   - For transformers.models.llama.modeling_llama.LlamaMLP: class_name="TransformersLlamaMLP", file_name="transformers_llama_mlp.py"

ultrathink 3. **Generate Module File**: Create Python class following BaseModule pattern:
   - Import BaseModule from .base (since we're in flat structure)
   - Import any required module classes (e.g., from .torch_linear import TorchLinear)
   - Implement all abstract methods with REAL Python code
   - **CRITICAL**: Convert formulas to executable Python code:
     - Replace `{{parameter}}` with `params['parameter']`
     - Replace `${{torch.nn.Linear}}(...)` with actual instantiation: `TorchLinear().compute_flops(...)`
     - Replace `${{transformers.LlamaMLP}}(...)` with `TransformersLlamaMLP().compute_flops(...)`

ultrathink 4. **Write Result**: Create generation_result.json with:
   {{
     "status": "success" | "error",
     "module_file": "path/to/generated/file.py",
     "class_name": "GeneratedClassName",
     "full_class_name": "original.full.class.Name",
     "error": "error message if failed"
   }}

ultrathink 5. **Write Diagnostics**: Create generation_result_diagnostics.json with:
   If successful:
   {{
     "module_generated": "{full_class_name}",
     "status": "success"
   }}
   If failed:
   {{
     "module_generated": "{full_class_name}",
     "status": "fail",
     "reason": "Description of what went wrong"
   }}

## Key Requirements:

### Naming Convention (Flat Structure):
- Class names: LibraryClassName (e.g., TorchLinear, TransformersLlamaMLP)
- File names: library_class_name.py (e.g., torch_linear.py, transformers_llama_mlp.py)
- CamelCase to snake_case conversion for file names

### Formula to Code Conversion:
Transform template formulas into real Python code:

**Example Input Formula:**
`"2 * ${{torch.nn.Linear}}({{B}} * {{S}}, {{hidden_size}}, {{intermediate_size}}) + 5 * {{B}} * {{S}} * {{intermediate_size}}"`

**Example Output Python Code:**
```python
def compute_flops(self, **params: Dict[str, Any]) -> int:
    torch_linear = TorchLinear()
    linear_flops = torch_linear.compute_flops(
        B=params['B'] * params['S'],
        input_features=params['hidden_size'],
        output_features=params['intermediate_size']
    )
    return 2 * linear_flops + 5 * params['B'] * params['S'] * params['intermediate_size']
```

### Module File Structure:
```python
from typing import Dict, Any
from .base import BaseModule
# Import required modules
from .torch_linear import TorchLinear

class ClassName(BaseModule):
    \"\"\"Auto-generated FLOP/memory calculator for full.class.name\"\"\"

    def get_required_parameters(self) -> Dict[str, str]:
        # Return parameter types from analysis

    def compute_flops(self, **params: Dict[str, Any]) -> int:
        # Real Python code with direct method calls

    def compute_memory_reads(self, **params: Dict[str, Any]) -> int:
        # Real Python code with direct method calls

    def compute_memory_writes(self, **params: Dict[str, Any]) -> int:
        # Real Python code with direct method calls

    def compute_intermediates(self, **params: Dict[str, Any]) -> int:
        # Real Python code with direct method calls
```

"""


def main():
    """Command-line interface for the module generator agent."""
    parser = argparse.ArgumentParser(description='Generate Python module files from module database')
    parser.add_argument('full_class_name', type=str,
                       help='Full class name to generate module for (e.g., transformers.models.llama.modeling_llama.LlamaMLP)')
    parser.add_argument('--module-db', type=str, default='module_db.json',
                       help='Path to module database file (default: module_db.json)')

    args = parser.parse_args()

    try:
        agent = ModuleGeneratorAgent(module_db_file=args.module_db)

        result = agent.generate_module_with_agent(full_class_name=args.full_class_name)

        if result["status"] == "success":
            print(f"✅ Module generation successful!")
            print(f"Generated file: {result['module_file']}")
            print(f"Class name: {result['class_name']}")
        else:
            print(f"❌ Module generation failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()