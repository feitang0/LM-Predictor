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

    def __init__(self, claude_command: str = "claude"):
        """Initialize the agent with Claude Code command."""
        self.claude_command = claude_command

    def analyze_module_with_agent(self, module_class: str, module_path: str) -> Dict[str, Any]:
        """
        Use Claude Code to analyze a module's forward function for FLOPs and memory.

        Args:
            module_class: PyTorch module class name (e.g., 'LlamaAttention')
            module_path: Python import path (e.g., 'transformers.models.llama.modeling_llama')

        Returns:
            Dictionary with analysis results including thinking process, formulas, and functions
        """
        prompt = self._create_analysis_prompt(module_class, module_path)

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
            raise RuntimeError(f"Claude Code analysis timed out for {module_class}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse Claude response: {e}")
        except Exception as e:
            raise RuntimeError(f"Analysis failed for {module_class}: {e}")

    def _create_analysis_prompt(self, module_class: str, module_path: str) -> str:
        """Create the prompt for Claude Code to analyze a module."""
        return f"""
I need to analyze the FLOPs and memory access patterns of the PyTorch module {module_class}.

The module is from: {module_path}

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

6. **Generate Functions**: Create Python functions to calculate FLOPs and memory

**Return your analysis as a JSON object with this exact structure:**

```json
{{
  "module_class": "{module_class}",
  "module_path": "{module_path}",
  "code_location": {{
    "file": "path/to/file.py",
    "line_start": 123,
    "line_end": 200
  }},
  "forward_method": {{
    "line_start": 145,
    "line_end": 180,
    "code_snippet": "def forward(self, ...):\\n    ..."
  }},
  "flop_analysis": {{
    "thinking_process": "Step-by-step explanation of how you calculated FLOPs for each operation",
    "parameters": ["batch_size", "seq_len", "hidden_size", "num_heads", "head_dim"],
    "formula": "Mathematical formula like: 8 * B * S * hidden_size^2 + 4 * B * num_heads * S^2 * head_dim",
    "function_code": "def count_{module_class.lower()}_flops(B, S, hidden_size, num_heads, head_dim):\\n    # Implementation\\n    return total_flops",
    "breakdown": {{
      "operation_name1": "2 * B * S * hidden_size^2",
      "operation_name2": "4 * B * num_heads * S^2 * head_dim"
    }}
  }},
  "memory_analysis": {{
    "thinking_process": "Explanation of memory access patterns",
    "parameters": ["batch_size", "seq_len", "hidden_size", "dtype_bytes"],
    "reads_formula": "Formula for total bytes read",
    "writes_formula": "Formula for total bytes written",
    "intermediates": "Formula for peak intermediate memory"
  }},
  "validation": {{
    "status": "pending",
    "validator": null,
    "date": null,
    "notes": "Agent-generated analysis for {module_class}"
  }},
  "dependencies": ["List", "of", "dependency", "modules"]
}}
```

Be extremely careful with your FLOP calculations. For matrix multiplication of shapes [A, B] x [B, C], the FLOPs are 2*A*B*C (multiply-accumulate). For attention mechanisms, calculate QK^T, softmax, and attention*V separately.

Make sure your formulas use standard variable names: B (batch_size), S (seq_len), and actual parameter names from the module.
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
            "module_class": "Unknown",
            "module_path": "unknown",
            "code_location": {
                "file": "unknown",
                "line_start": 0,
                "line_end": 0
            },
            "forward_method": {
                "line_start": 0,
                "line_end": 0,
                "code_snippet": "# Parse failed"
            },
            "flop_analysis": {
                "thinking_process": f"Failed to parse response: {response_text[:200]}...",
                "parameters": ["batch_size", "seq_len"],
                "formula": "0",
                "function_code": "def count_flops(): return 0",
                "breakdown": {"unknown": "0"}
            },
            "memory_analysis": {
                "thinking_process": "Parse failed",
                "parameters": ["batch_size", "seq_len"],
                "reads_formula": "0",
                "writes_formula": "0",
                "intermediates": "0"
            },
            "validation": {
                "status": "failed",
                "validator": None,
                "date": None,
                "notes": "Agent response parsing failed"
            },
            "dependencies": []
        }

    def _validate_analysis(self, analysis: Dict[str, Any]) -> None:
        """Validate the structure of the analysis result."""
        required_keys = [
            "module_class", "module_path", "code_location", "forward_method",
            "flop_analysis", "memory_analysis", "validation", "dependencies"
        ]

        for key in required_keys:
            if key not in analysis:
                raise ValueError(f"Missing required key: {key}")

        # Validate flop_analysis structure
        flop_keys = ["thinking_process", "parameters", "formula", "function_code", "breakdown"]
        for key in flop_keys:
            if key not in analysis["flop_analysis"]:
                raise ValueError(f"Missing flop_analysis key: {key}")

        # Validate memory_analysis structure
        memory_keys = ["thinking_process", "parameters", "reads_formula", "writes_formula", "intermediates"]
        for key in memory_keys:
            if key not in analysis["memory_analysis"]:
                raise ValueError(f"Missing memory_analysis key: {key}")


def main():
    """Command-line interface for the module analyzer agent."""
    parser = argparse.ArgumentParser(description='Analyze PyTorch module with Claude Code agent')
    parser.add_argument('--module_class', type=str, required=True,
                       help='PyTorch module class name (e.g., LlamaAttention)')
    parser.add_argument('--module_path', type=str, required=True,
                       help='Python import path (e.g., transformers.models.llama.modeling_llama)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (default: print to stdout)')
    parser.add_argument('--claude_command', type=str, default='claude',
                       help='Claude Code command (default: claude)')

    args = parser.parse_args()

    try:
        agent = ModuleAnalyzerAgent(claude_command=args.claude_command)
        analysis = agent.analyze_module_with_agent(args.module_class, args.module_path)

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