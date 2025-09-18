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
            # Prepare environment (current env + any from .env file)
            env = os.environ.copy()

            # Load .env variables only for this subprocess (doesn't affect system env)
            env_vars = dotenv_values(".env")
            env.update(env_vars)

            # Run Claude Code in headless mode
            result = subprocess.run([
                self.claude_command, "-p", prompt,
                "--output-format", "json"
            ], capture_output=True, text=True, timeout=300, env=env)

            if result.returncode != 0:
                raise RuntimeError(f"Claude Code failed: {result.stderr}")

            print("=== CLAUDE RESPONSE ===")
            print(result.stdout)
            print("=== END CLAUDE RESPONSE ===")

            # Parse Claude's response
            claude_response = json.loads(result.stdout)
            response_text = claude_response.get("result", "")

            # Extract JSON from Claude's response
            analysis = self._parse_claude_response(response_text)

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
ultrathink: Build reusable analysis for LLM module: {module_spec}

GOAL: Create a modular FLOP and memory analysis that can be reused across different model configurations. This analysis will be stored in a database and composed with other modules.

CRITICAL PRINCIPLE: Maximize reusability through module dependencies rather than hardcoded calculations.

## Available Resources:
- transformers/ - Hugging Face transformers source code
- pytorch/ - PyTorch source code
- module_db_schema.json - Output schema specification

## Analysis Steps:

1. **Locate Module**: Find {module_spec} in the codebase and read its forward() method

2. **Decompose Operations**: Break down the forward() method into:
   - **Reusable modules** (any class with its own forward method) → Use {{ModuleName}}() syntax
   - **Primitive operations** (activations, element-wise math, tensor ops) → Calculate FLOPs directly

3. **Build Modular Formulas**:
   - **Modules**: {{torch.nn.Linear}}(${{batch_size}}, ${{input_dim}}, ${{output_dim}})
   - **Primitives**: Direct FLOP counts (e.g., "4 * ${{B}} * ${{S}} * ${{hidden_size}}" for SiLU activation)
   - **Parameters**: Use descriptive names (${{B}}, ${{S}}, ${{hidden_size}}, ${{intermediate_size}})

4. **Memory Analysis**: Calculate reads/writes/intermediates using same modular approach

## Example Reference:
Here's a good example of the expected output format:

{example_json}

## Output Requirements:
Return ONLY a JSON object matching module_db_schema.json schema. Use:
- ${{parameter}} for variables
- {{ModuleName}}(args) for module dependencies
- Full class names (e.g., "transformers.models.llama.modeling_llama.LlamaMLP")
- "human_validated": false in validation section
"""

    def _parse_claude_response(self, response_text: str) -> Dict[str, Any]:
        """Extract JSON analysis from Claude's response text."""
        # Look for JSON block in the response
        lines = response_text.strip().split('\n')
        json_start = -1
        json_end = -1

        for i, line in enumerate(lines):
            if line.strip().startswith('```json'):
                json_start = i + 1
            elif line.strip() == '```' and json_start != -1:
                json_end = i
                break

        if json_start == -1 or json_end == -1:
            # Try to find JSON object directly
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')
            if start_idx != -1 and end_idx != -1:
                json_text = response_text[start_idx:end_idx+1]
            else:
                raise ValueError("No JSON found in Claude's response")
        else:
            json_text = '\n'.join(lines[json_start:json_end])

        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            # Fallback: try to extract key information manually
            return self._manual_parse_response(response_text)

    def _manual_parse_response(self, response_text: str) -> Dict[str, Any]:
        """Fallback parser for when JSON extraction fails."""
        # This is a simple fallback - in practice, you'd want more robust parsing
        return {
            "full_class_name": "unknown",
            "code_location": {
                "file": "unknown",
                "line_start": 0,
                "line_end": 0
            },
            "flop_analysis": {
                "thinking_process": f"Failed to parse response: {response_text[:200]}...",
                "parameters": [
                    {"name": "B", "type": "int", "description": "batch size"},
                    {"name": "S", "type": "int", "description": "sequence length"}
                ],
                "calculation_formula": "0",
                "module_depends": [],
                "breakdown": {"unknown": "0"}
            },
            "memory_analysis": {
                "thinking_process": "Parse failed",
                "parameters": [
                    {"name": "B", "type": "int", "description": "batch size"},
                    {"name": "S", "type": "int", "description": "sequence length"},
                    {"name": "dtype_bytes", "type": "int", "description": "bytes per data type element"}
                ],
                "reads_calculation_formula": "0",
                "writes_calculation_formula": "0",
                "intermediates_calculation_formula": "0",
                "module_depends": []
            },
            "validation": {
                "human_validated": False
            }
        }

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