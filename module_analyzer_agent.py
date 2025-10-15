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
from jsonschema import validate, ValidationError


class ModuleAnalyzerAgent:
    """Claude Code agent for analyzing PyTorch module forward functions."""

    def __init__(self, claude_command: str = "claude", schema_path: str = "module_db_schema.json", examples_path: str = "module_db_examples.json", output_file: str = "module_analysis.json"):
        """Initialize the agent with Claude Code command."""
        self.claude_command = claude_command
        self.schema_path = schema_path
        self.examples_path = examples_path
        self.output_file = output_file

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

    def analyze_module_with_agent(self, module_spec: str, output_file: Optional[str] = None, sub_layers: list = None) -> Dict[str, Any]:
        """
        Use Claude Code to analyze a module's forward function for FLOPs and memory.

        Args:
            module_spec: Module specification - can be class name or full path
                        (e.g., 'Linear', 'torch.nn.Linear', 'LlamaAttention')
            output_file: Override default output file for this analysis
            sub_layers: Optional sub-layers from architecture (for composite modules)

        Returns:
            Dictionary with analysis results including thinking process, formulas, and functions
        """
        analysis_output_file = output_file or self.output_file
        prompt = self._create_analysis_prompt(module_spec, analysis_output_file, sub_layers)

        print("=== PROMPT ===")
        print(prompt)
        print("=== END PROMPT ===")

        try:
            # Remove output file if it exists
            if os.path.exists(analysis_output_file):
                os.remove(analysis_output_file)

            # Remove diagnostic file if it exists
            diagnostic_file = analysis_output_file.replace('.json', '_diagnostics.json')
            if os.path.exists(diagnostic_file):
                os.remove(diagnostic_file)

            # Run Claude Code in headless mode
            result = subprocess.run([
                self.claude_command, "-p", prompt,
                "--dangerously-skip-permissions"
            ], capture_output=True, text=True, timeout=600)

            print(f"=== SUBPROCESS RESULT ===")
            print(f"Return code: {result.returncode}")
            print(f"=== STDOUT ===")
            print(result.stdout)
            print(f"=== STDERR ===")
            print(result.stderr)
            print(f"=== END SUBPROCESS RESULT ===")

            if result.returncode != 0:
                raise RuntimeError(f"Claude Code failed with return code {result.returncode}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")

            print("=== CLAUDE RESPONSE ===")
            print(result.stdout)
            print("=== END CLAUDE RESPONSE ===")

            # Read analysis from output file
            if not os.path.exists(analysis_output_file):
                raise FileNotFoundError(f"Claude did not create {analysis_output_file} file")

            with open(analysis_output_file, 'r') as f:
                analysis = json.load(f)

            # Read diagnostic information (for debugging purposes)
            diagnostic_file = analysis_output_file.replace('.json', '_diagnostics.json')
            if os.path.exists(diagnostic_file):
                with open(diagnostic_file, 'r') as f:
                    diagnostics = json.load(f)
                print(f"Diagnostics: {diagnostics}")

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

    def _create_analysis_prompt(self, module_spec: str, output_file: str, sub_layers: list = None) -> str:
        """Create the prompt for Claude Code to analyze a module.

        Args:
            module_spec: Module specification
            output_file: Output file path
            sub_layers: Optional sub-layers from architecture (for composite modules)
        """
        # Load examples to get template structure
        examples = self._load_examples()
        example = examples[1]  # Use the LlamaAttention example as template
        example_json = json.dumps(example, indent=2)

        # Build composite module context if sub_layers provided
        composite_context = ""
        if sub_layers:
            # Extract class names from sub_layers
            sub_layer_classes = [sl.get('class', 'unknown') for sl in sub_layers]
            composite_context = f"""
⚠️ COMPOSITE MODULE DETECTED ⚠️

This module contains these sub-layers (already analyzed separately):
{json.dumps(sub_layer_classes, indent=2)}

CRITICAL RULE: Count ONLY operations that occur BETWEEN sub-layer calls.
- ✅ Include: Matmuls, softmax, element-wise ops, residual connections
- ❌ Exclude: Any computation done by the sub-layers listed above
- Do NOT use ${{torch.nn.Linear}} or other ${{...}} references for sub-layers above

Example: For LlamaSdpaAttention, count Q@K^T matmul, softmax, scores@V matmul.
Do NOT count q_proj/k_proj/v_proj/o_proj Linear layers - they're sub-layers.

"""

        return f"""
ultrathink: Analyze the module: {module_spec}

{composite_context}

GOAL: Calculate the FLOPs and memory read/write volumes when calling forward() on this module

Available Resources:
- transformers/ - Hugging Face transformers source code
- pytorch/ - PyTorch source code
- module_db_schema.json - Output schema specification

CRITICAL PRINCIPLES:
1. Each module counts ONLY operations that occur within its own forward() method
2. Do NOT count FLOPs that created the input tensors - those belong to other modules
3. Use ${{}} to represent a module reference and {{}} to represent a parameter
4. Example: ${{torch.nn.Linear}}({{B}} * {{S}}, {{input_dim}}, {{output_dim}})

TODO List:

ultrathink 1. **Locate Module**: Find {module_spec} in the codebase and read its forward() method

ultrathink 2. **Determine Operations**: Identify what computations happen WITHIN this module:
   - **Module calls**: When this module calls other modules (e.g., self.linear(...)), use references like ${{torch.nn.Linear}}({{B}} * {{S}}, {{input_dim}}, {{output_dim}})
   - **Direct calculations**: Operations performed directly (e.g., element-wise ops, activations, normalizations)
   - **DO NOT COUNT**: Operations that produced the input tensors

ultrathink 3. **Analyze FLOPs**: Count operations following these rules:
   - For modules that internally call Linear layers: Use ${{torch.nn.Linear}} references
   - For direct computations: Use explicit formulas like "4 * {{B}} * {{S}} * {{hidden_size}}" for SiLU
   - Example for attention: Count RoPE application + SDPA operations, but delegate Linear projections

ultrathink 4. **Analyze Memory**: Calculate data movement within this module:
   - **Reads**: Parameter weights + input activations (what this module reads)
   - **Writes**: Output activations (what this module produces)
   - **Intermediates**: Temporary tensors created during computation

ultrathink 5. **Decide Naming**: Determine the generated file name and class name using semantic rules:
   - Extract library and class name from full_class_name
   - File naming: Convert to snake_case with library prefix
     Examples:
     * torch.nn.modules.activation.SiLU → file: "torch_silu.py", class: "TorchSilu"
     * torch.nn.modules.linear.Linear → file: "torch_linear.py", class: "TorchLinear"
     * transformers.models.llama.modeling_llama.LlamaRMSNorm → file: "transformers_llama_rms_norm.py", class: "TransformersLlamaRMSNorm"
     * transformers.models.llama.modeling_llama.LlamaSdpaAttention → file: "transformers_llama_sdpa_attention.py", class: "TransformersLlamaSdpaAttention"
   - Use semantic understanding to handle acronyms (RMS, SDPA, MLP) and word boundaries
   - Class naming: PascalCase with library prefix (e.g., TorchSilu, TransformersLlamaRMSNorm)

ultrathink 6. **Write Analysis**: Create complete JSON matching the schema (including generated_file_name and generated_class_name) and write to {output_file}

ultrathink 7. **Write Diagnostics**: Create diagnostic information and write to {output_file.replace('.json', '_diagnostics.json')} with:
   If successful:
   {{
     "module_analyzed": "{module_spec}",
     "status": "success"
   }}
   If failed:
   {{
     "module_analyzed": "{module_spec}",
     "status": "fail",
     "reason": "Description of what went wrong"
   }}

## Example Reference:
Here's an example of the expected output format (NOTE: Add generated_file_name and generated_class_name fields):

{example_json}

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
    parser.add_argument('--output_file', type=str, default='module_analysis.json',
                       help='Analysis output file (default: module_analysis.json)')

    args = parser.parse_args()

    try:
        agent = ModuleAnalyzerAgent(claude_command=args.claude_command, schema_path=args.schema_path, examples_path=args.examples_path, output_file=args.output_file)
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
