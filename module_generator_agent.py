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

    def __init__(self, claude_command: str = "claude", output_dir: str = "generated_modules"):
        """Initialize the agent with Claude Code command."""
        self.claude_command = claude_command
        self.output_dir = Path(output_dir)

    def generate_module_with_agent(self, response_file: str) -> Dict[str, str]:
        """
        Use Claude Code to generate a Python module file from analysis response.

        Args:
            response_file: Path to JSON response file from agent analysis

        Returns:
            Dictionary with generation results including file paths
        """
        prompt = self._create_generation_prompt(response_file)

        print("=== PROMPT ===")
        print(prompt)
        print("=== END PROMPT ===")

        try:
            # Remove generation_result.json if it exists
            if os.path.exists("generation_result.json"):
                os.remove("generation_result.json")

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

            return generation_result

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Claude Code generation timed out for {response_file}")
        except json.JSONDecodeError as e:
            print(f"Claude stdout: {result.stdout}")
            print(f"Claude stderr: {result.stderr}")
            raise RuntimeError(f"Failed to parse Claude response: {e}")
        except Exception as e:
            raise RuntimeError(f"Generation failed for {response_file}: {e}")

    def _create_generation_prompt(self, response_file: str) -> str:
        """Create the prompt for Claude Code to generate module files."""
        return f"""
ultrathink: Generate Python module file from agent analysis response

GOAL: Create a callable Python module file from the agent analysis response using flat directory structure

Available Resources:
- {response_file} - Agent analysis response with FLOP/memory formulas
- generated_modules/ - Directory for generated modules (flat structure)
- generated_modules/base.py - BaseModule abstract class
- DESIGN.md - System architecture reference

IMPORTANT:
1. Follow the naming conventions from updated DESIGN.md exactly
2. Generate real executable Python code, NOT template formulas
3. Your final step must be to write the generation result to generation_result.json

TODO List:

1. **Read Response**: Load and examine the agent analysis from {response_file}

2. **Determine Paths**: Convert full_class_name to required paths (flat structure):
   - For torch.nn.Linear: class_name="TorchLinear", file_name="torch_linear.py"
   - For transformers.models.llama.modeling_llama.LlamaMLP: class_name="TransformersLlamaMLP", file_name="transformers_llama_mlp.py"

3. **Generate Module File**: Create Python class following BaseModule pattern:
   - Import BaseModule from .base (since we're in flat structure)
   - Import any required module classes (e.g., from .torch_linear import TorchLinear)
   - Implement all abstract methods with REAL Python code
   - **CRITICAL**: Convert formulas to executable Python code:
     - Replace `${{parameter}}` with `params['parameter']`
     - Replace `{{torch.nn.Linear}}(...)` with actual instantiation: `TorchLinear().compute_flops(...)`
     - Replace `{{transformers.LlamaMLP}}(...)` with `TransformersLlamaMLP().compute_flops(...)`

4. **Write Result**: Create generation_result.json with:
   {{
     "status": "success" | "error",
     "module_file": "path/to/generated/file.py",
     "class_name": "GeneratedClassName",
     "full_class_name": "original.full.class.Name",
     "error": "error message if failed"
   }}

## Key Requirements:

### Naming Convention (Flat Structure):
- Class names: LibraryClassName (e.g., TorchLinear, TransformersLlamaMLP)
- File names: library_class_name.py (e.g., torch_linear.py, transformers_llama_mlp.py)
- CamelCase to snake_case conversion for file names

### Formula to Code Conversion:
Transform template formulas into real Python code:

**Example Input Formula:**
`"2 * {{torch.nn.Linear}}(${{B}} * ${{S}}, ${{hidden_size}}, ${{intermediate_size}}) + 5 * ${{B}} * ${{S}} * ${{intermediate_size}}"`

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

Your final response must contain ONLY: SUCCESS or FAIL
"""


def main():
    """Command-line interface for the module generator agent."""
    parser = argparse.ArgumentParser(description='Generate Python module files from analysis responses')
    parser.add_argument('response_file', type=str,
                       help='Path to JSON response file from agent analysis')
    parser.add_argument('--output-dir', type=str, default='generated_modules',
                       help='Output directory for generated modules (default: generated_modules)')
    parser.add_argument('--claude-command', type=str, default='claude',
                       help='Claude Code command (default: claude)')

    args = parser.parse_args()

    # Validate input file exists
    if not os.path.exists(args.response_file):
        print(f"Error: Response file not found: {args.response_file}", file=sys.stderr)
        sys.exit(1)

    try:
        agent = ModuleGeneratorAgent(
            claude_command=args.claude_command,
            output_dir=args.output_dir
        )

        result = agent.generate_module_with_agent(response_file=args.response_file)

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