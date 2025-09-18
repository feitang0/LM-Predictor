#!/usr/bin/env python3
"""
Claude Code Agent for analyzing PyTorch modules' FLOP and memory characteristics.
This module uses Claude Code in headless mode to understand forward function implementations.
"""

import json
import os
import subprocess
import sys
import argparse
from typing import Dict, Any, Optional
from dotenv import dotenv_values
from jsonschema import validate, ValidationError


class ModuleAnalyzerAgent:
    """Claude Code agent for analyzing PyTorch module forward functions."""

    def __init__(self, claude_command: str = "claude", schema_path: str = "module_db_schema.json", examples_path: str = "module_db_examples.json"):
        """Initialize the agent with Claude Code command."""
        self.claude_command = claude_command
        self.schema_path = schema_path
        self.examples_path = examples_path

    def _load_schema(self) -> Dict[str, Any]:
        """Load the module database schema."""
        if not os.path.exists(self.schema_path):
            raise FileNotFoundError(f"Schema file not found: {self.schema_path}")

        with open(self.schema_path, 'r') as f:
            return json.load(f)

    def _load_examples(self) -> list[Dict[str, Any]]:
        """Load the module database examples."""
        if not os.path.exists(self.examples_path):
            raise FileNotFoundError(f"Examples file not found: {self.examples_path}")

        with open(self.examples_path, 'r') as f:
            return json.load(f)

    def analyze_module_with_agent(self, module_spec: str) -> Dict[str, Any]:
        """
        Use Claude Code to analyze a module's forward function for FLOPs and memory.

        Args:
            module_spec: Module specification - can be class name or full path
                        (e.g., 'Linear', 'torch.nn.Linear', 'LlamaAttention')

        Returns:
            Dictionary with analysis results including thinking process, formulas, and functions
        """
        prompt = self._create_analysis_prompt(module_spec)

        print("=== PROMPT ===")
        print(prompt)
        print("=== END PROMPT ===")

        try:
            # Remove response.json if it exists
            if os.path.exists("response.json"):
                os.remove("response.json")

            # Prepare environment (current env + any from .env file)
            env = os.environ.copy()

            # Load .env variables only for this subprocess (doesn't affect system env)
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

            # Read analysis from response.json
            if not os.path.exists("response.json"):
                raise FileNotFoundError("Claude did not create response.json file")

            with open("response.json", 'r') as f:
                analysis = json.load(f)

            # Validate the analysis structure
            self._validate_analysis(analysis)

            return analysis

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Claude Code analysis timed out for {module_spec}")
        except json.JSONDecodeError as e:
            print(f"Claude stdout: {result.stdout}")
            print(f"Claude stderr: {result.stderr}")
            raise RuntimeError(f"Failed to parse Claude response: {e}")
        except Exception as e:
            raise RuntimeError(f"Analysis failed for {module_spec}: {e}")

    def _create_analysis_prompt(self, module_spec: str) -> str:
        """Create the prompt for Claude Code to analyze a module."""
        # Load examples to get template structure
        examples = self._load_examples()
        example = examples[1]  # Use the LlamaAttention example as template
        example_json = json.dumps(example, indent=2)

        return f"""
ultrathink: Analyze the module: {module_spec}

GOAL: Calculate the FLOPs and memory read/write volumes when calling forward() on this module

Available Resources:
- transformers/ - Hugging Face transformers source code
- pytorch/ - PyTorch source code
- module_db_schema.json - Output schema specification

IMPORTANT:
1. Your final step must be to write the complete JSON analysis to response.json
2. Use {{}} to represent a module and ${{}} to represent a parameter

TODO List:

1. **Locate Module**: Find {module_spec} in the codebase and read its forward() method

2. **Analyze FLOPs**: Break down the forward() operations into:
   - **Module calls**: Such as {{torch.nn.Linear}}(${{B}} * ${{S}}, ${{input_dim}}, ${{output_dim}})
   - **Direct calculations**: For all primitive operations (e.g., "4 * ${{B}} * ${{S}} * ${{hidden_size}}" for SiLU, "${{B}} * ${{S}} * ${{hidden_size}}" for element-wise ops, "7 * ${{B}} * ${{S}} * ${{hidden_size}}" for LayerNorm, etc.)

3. **Analyze Memory**: Calculate data movement:
   - **Reads**: Parameter weights + input activations
   - **Writes**: Output activations
   - **Intermediates**: Temporary tensors created during computation
   - **Note**: Memory analysis focuses on data flow, not computation - may have empty module_depends

4. **Write Result**: Create complete JSON matching the schema and write to response.json

## Example Reference:
Here's an example of the expected output format:

{example_json}

Your final response must contain ONLY: SUCCESS or FAIL
"""

    def _parse_claude_response(self, response_text: str) -> Dict[str, Any]:
        """Extract JSON analysis from Claude's response text."""
        import re

        # First try: Look for JSON code block (```json ... ```)
        json_block_pattern = r'```json\s*\n(.*?)\n```'
        json_match = re.search(json_block_pattern, response_text, re.DOTALL)

        if json_match:
            json_text = json_match.group(1).strip()
        else:
            # Second try: Assume the entire response is JSON
            json_text = response_text.strip()

        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from Claude's response: {e}")


    def _validate_analysis(self, analysis: Dict[str, Any]) -> None:
        """Validate the structure of the analysis result using JSON Schema."""
        try:
            schema = self._load_schema()
            validate(instance=analysis, schema=schema)
        except ValidationError as e:
            # Provide more helpful error messages
            error_path = " -> ".join(str(p) for p in e.absolute_path) if e.absolute_path else "root"
            raise ValueError(f"Schema validation failed at {error_path}: {e.message}")
        except Exception as e:
            raise ValueError(f"Validation failed: {e}")


def main():
    """Command-line interface for the module analyzer agent."""
    parser = argparse.ArgumentParser(description='Analyze PyTorch module with Claude Code agent')
    parser.add_argument('module', type=str,
                       help='Module specification: class name or full path (e.g., Linear, torch.nn.Linear)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (default: print to stdout)')
    parser.add_argument('--claude_command', type=str, default='claude',
                       help='Claude Code command (default: claude)')
    parser.add_argument('--schema_path', type=str, default='module_db_schema.json',
                       help='Path to module database schema file (default: module_db_schema.json)')
    parser.add_argument('--examples_path', type=str, default='module_db_examples.json',
                       help='Path to module database examples file (default: module_db_examples.json)')

    args = parser.parse_args()

    try:
        agent = ModuleAnalyzerAgent(claude_command=args.claude_command, schema_path=args.schema_path, examples_path=args.examples_path)
        analysis = agent.analyze_module_with_agent(args.module)

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(analysis, f, indent=2)
            print(f"Analysis saved to {args.output}")
        else:
            print(json.dumps(analysis, indent=2))

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
