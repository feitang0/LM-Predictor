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

    def __init__(self, claude_command: str = "claude", schema_path: str = "module_db_schema_v2.json", examples_path: str = "module_db_examples.json"):
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
ultrathink: I need to analyze the FLOPs and memory access patterns of the PyTorch module: {module_spec}

This could be a class name (e.g., 'Linear') or a full module path (e.g., 'torch.nn.Linear'). Please find the module in the codebase.

You have access to:
- transformers/ - Transformers library source code
- pytorch/ - PyTorch source code (if needed)

TODO List:

1. **Find the Module**: Locate {module_spec} in the codebase using search tools
2. **Read Forward Method**: Extract and read the complete forward() method implementation
3. **Analyze Operations**: Step by step, identify all computational operations

4. **Calculate FLOPs**: For each operation, determine:
   - Input/output tensor shapes
   - Number of floating point operations
   - Dependencies on batch_size (B), sequence_length (S), and model dimensions

5. **Memory Analysis**: Analyze memory access patterns:
   - Parameter reads (weight matrices, biases)
   - Activation reads/writes
   - Intermediate tensor storage

6. **Generate Templates**: Create formula templates using the template syntax

**Template Syntax Requirements**:
- Parameters: Use ${{param}} syntax (e.g., ${{B}}, ${{S}}, ${{hidden_size}})
- Module calls: Use {{Module}}() syntax (e.g., {{torch.nn.Linear}}(${{B}} * ${{S}}, ${{input_dim}}, ${{output_dim}}))

**Your response must be ONLY the JSON object below with your actual analysis data:**

```json
{example_json}
```
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
                "formula_template": "0",
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
                "reads_template": "0",
                "writes_template": "0",
                "intermediates_template": "0",
                "module_depends": []
            },
            "validation": {
                "status": "failed",
                "validator": None,
                "date": None,
                "notes": "Agent response parsing failed"
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
    parser.add_argument('--schema_path', type=str, default='module_db_schema_v2.json',
                       help='Path to module database schema file (default: module_db_schema_v2.json)')
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