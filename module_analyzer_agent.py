#!/usr/bin/env python3
"""
Claude Code Agent for analyzing PyTorch modules' FLOP and memory characteristics.
This module uses Claude Code in headless mode to understand forward function implementations.
"""

import json
import subprocess
import sys
import argparse
from typing import Dict, Any, Optional


class ModuleAnalyzerAgent:
    """Claude Code agent for analyzing PyTorch module forward functions."""

    def __init__(self, claude_command: str = "claude", schema_path: str = "module_db_schema.json"):
        """Initialize the agent with Claude Code command."""
        self.claude_command = claude_command
        self.schema_path = schema_path

    def _load_schema(self) -> Dict[str, Any]:
        """Load the module database schema."""
        if not os.path.exists(self.schema_path):
            raise FileNotFoundError(f"Schema file not found: {self.schema_path}")

        with open(self.schema_path, 'r') as f:
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

        try:
            # Run Claude Code in headless mode
            result = subprocess.run([
                self.claude_command, "-p", prompt,
                "--output-format", "json",
                "--allowedTools", "Read,Grep,Glob"
            ], capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                raise RuntimeError(f"Claude Code failed: {result.stderr}")

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
            raise RuntimeError(f"Failed to parse Claude response: {e}")
        except Exception as e:
            raise RuntimeError(f"Analysis failed for {module_spec}: {e}")

    def _create_analysis_prompt(self, module_spec: str) -> str:
        """Create the prompt for Claude Code to analyze a module."""
        # Load schema to get the structure
        schema = self._load_schema()
        example = schema["example_entries"][1]  # Use the LlamaAttention example as template

        # Generate example JSON structure from schema
        example_json = json.dumps({
            "full_class_name": "module.path.ClassName",
            "code_location": example["code_location"],
            "flop_analysis": {
                "thinking_process": "Step-by-step explanation of how you calculated FLOPs for each operation",
                "parameters": example["flop_analysis"]["parameters"],
                "formula_template": "Template using ${param} and {Module}() syntax like: 2 * ${B} * ${S} * ${input_features} * ${output_features}",
                "module_depends": ["torch__nn__Linear"],
                "breakdown": {
                    "operation_name1": "${B} * ${S} * ${param}",
                    "operation_name2": "{torch__nn__Linear}(${B} * ${S}, ${input_dim}, ${output_dim})"
                }
            },
            "memory_analysis": {
                "thinking_process": "Explanation of memory access patterns",
                "parameters": example["memory_analysis"]["parameters"],
                "reads_template": "${param1} * ${param2} * ${dtype_bytes}",
                "writes_template": "${param1} * ${param2} * ${dtype_bytes}",
                "intermediates_template": "${param1} * ${param2} * ${dtype_bytes}",
                "module_depends": []
            },
            "validation": {
                "status": "pending",
                "validator": None,
                "date": None,
                "notes": f"Agent-generated analysis for {module_spec}"
            }
        }, indent=2)

        return f"""
I need to analyze the FLOPs and memory access patterns of the PyTorch module: {module_spec}

This could be a class name (e.g., 'Linear') or a full module path (e.g., 'torch.nn.Linear'). Please find the module in the codebase.

You have access to:
- transformers/ - Transformers library source code
- pytorch/ - PyTorch source code (if needed)

Please perform the following analysis:

1. **Find the Module**: Locate {module_class} in the codebase using search tools
2. **Read Forward Method**: Extract and read the complete forward() method implementation
3. **Analyze Operations**: Step by step, identify all computational operations:
   - Matrix multiplications (Linear layers, einsum, etc.)
   - Element-wise operations (activations, normalization, etc.)
   - Attention computations (if applicable)
   - Any other FLOP-intensive operations

4. **Calculate FLOPs**: For each operation, determine:
   - Input/output tensor shapes
   - Number of floating point operations
   - Dependencies on batch_size (B), sequence_length (S), and model dimensions

5. **Memory Analysis**: Analyze memory access patterns:
   - Parameter reads (weight matrices, biases)
   - Activation reads/writes
   - Intermediate tensor storage

6. **Generate Templates**: Create formula templates using the template syntax

**IMPORTANT**: Use template syntax in your formulas:
- Parameters: Use ${{param}} syntax (e.g., ${{B}}, ${{S}}, ${{hidden_size}})
- Module calls: Use {{Module}}() syntax (e.g., {{torch__nn__Linear}}(${{B}} * ${{S}}, ${{input_dim}}, ${{output_dim}}))
- Module names use double underscores: torch.nn.Linear becomes torch__nn__Linear

**Return your analysis as a JSON object with this exact structure:**

```json
{example_json}
```

Be extremely careful with your FLOP calculations. For matrix multiplication of shapes [A, B] x [B, C], the FLOPs are 2*A*B*C (multiply-accumulate). For attention mechanisms, calculate QK^T, softmax, and attention*V separately.

Make sure your templates use standard variable names: B (batch_size), S (seq_len), and actual parameter names from the module.

The parameters array should include objects with "name", "type", and "description" fields for each parameter used in the templates.
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
        """Validate the structure of the analysis result."""
        required_keys = [
            "full_class_name", "code_location",
            "flop_analysis", "memory_analysis", "validation"
        ]

        for key in required_keys:
            if key not in analysis:
                raise ValueError(f"Missing required key: {key}")

        # Validate flop_analysis structure
        flop_keys = ["thinking_process", "parameters", "formula_template", "module_depends", "breakdown"]
        for key in flop_keys:
            if key not in analysis["flop_analysis"]:
                raise ValueError(f"Missing flop_analysis key: {key}")

        # Validate memory_analysis structure
        memory_keys = ["thinking_process", "parameters", "reads_template", "writes_template", "intermediates_template", "module_depends"]
        for key in memory_keys:
            if key not in analysis["memory_analysis"]:
                raise ValueError(f"Missing memory_analysis key: {key}")

        # Validate parameters structure
        for section in ["flop_analysis", "memory_analysis"]:
            params = analysis[section]["parameters"]
            if not isinstance(params, list):
                raise ValueError(f"{section}.parameters must be a list")

            for param in params:
                if not isinstance(param, dict):
                    raise ValueError(f"{section}.parameters items must be objects")

                param_keys = ["name", "type", "description"]
                for key in param_keys:
                    if key not in param:
                        raise ValueError(f"Missing parameter key {key} in {section}.parameters")


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

    args = parser.parse_args()

    try:
        agent = ModuleAnalyzerAgent(claude_command=args.claude_command, schema_path=args.schema_path)
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