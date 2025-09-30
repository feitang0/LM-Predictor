#!/usr/bin/env python3
"""
Claude Code Agent for generating standardized model architecture JSON from enhanced model string representation.
This module uses Claude Code in headless mode to transform PyTorch model structure into hierarchical representation.
"""

import json
import os
import subprocess
from typing import Dict, Any, Optional
from jsonschema import validate, ValidationError


class ModelArchitectureAgent:
    """Claude Code agent for generating standardized model architecture JSON."""

    def __init__(self, claude_command: str = "claude", schema_path: str = "model_representation_schema.json", examples_path: str = "model_representation_examples.json", output_file: str = "model_architecture.json"):
        """Initialize the agent with Claude Code command and schema paths."""
        self.claude_command = claude_command
        self.schema_path = schema_path
        self.examples_path = examples_path
        self.output_file = output_file

    def _load_schema(self) -> Dict[str, Any]:
        """Load the model representation schema."""
        if not os.path.exists(self.schema_path):
            raise FileNotFoundError(f"Schema file not found: {self.schema_path}")

        with open(self.schema_path, 'r') as f:
            return json.load(f)

    def _load_examples(self) -> Dict[str, Any]:
        """Load the model representation examples."""
        if not os.path.exists(self.examples_path):
            raise FileNotFoundError(f"Examples file not found: {self.examples_path}")

        with open(self.examples_path, 'r') as f:
            return json.load(f)

    def generate_architecture_with_agent(self, model_id: str, model_structure: str, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Use Claude Code to generate standardized model architecture JSON from enhanced model string.

        Args:
            model_id: Model identifier (e.g., 'meta-llama/Llama-2-7b-hf')
            model_structure: Enhanced model structure string with full class names
            output_file: Override default output file for this analysis

        Returns:
            Dictionary with standardized architecture representation
        """
        architecture_output_file = output_file or self.output_file
        prompt = self._create_architecture_prompt(model_id, model_structure, architecture_output_file)

        print("=== PROMPT ===")
        print(prompt)
        print("=== END PROMPT ===")

        try:
            # Remove output file if it exists
            if os.path.exists(architecture_output_file):
                os.remove(architecture_output_file)

            # Remove diagnostic file if it exists
            diagnostic_file = architecture_output_file.replace('.json', '_diagnostics.json')
            if os.path.exists(diagnostic_file):
                os.remove(diagnostic_file)

            # Run Claude Code in headless mode
            result = subprocess.run([
                self.claude_command, "-p", prompt,
                "--dangerously-skip-permissions",
                "--verbose"
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

            # Read architecture from output file
            if not os.path.exists(architecture_output_file):
                raise FileNotFoundError(f"Claude did not create {architecture_output_file} file")

            with open(architecture_output_file, 'r') as f:
                architecture = json.load(f)

            # Read diagnostic information (for debugging purposes)
            diagnostic_file = architecture_output_file.replace('.json', '_diagnostics.json')
            if os.path.exists(diagnostic_file):
                with open(diagnostic_file, 'r') as f:
                    diagnostics = json.load(f)
                print(f"Diagnostics: {diagnostics}")

            # Validate against schema
            schema = self._load_schema()
            validate(instance=architecture, schema=schema)
            print("✓ Architecture JSON validated against schema")

            return architecture

        except ValidationError as e:
            raise ValueError(f"Architecture validation failed: {e.message}")
        except subprocess.TimeoutExpired:
            raise RuntimeError("Claude Code agent timed out")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Required file not found: {e}")
        except Exception as e:
            raise RuntimeError(f"Agent execution failed: {e}")

    def _create_architecture_prompt(self, model_id: str, model_structure: str, output_file: str) -> str:
        """Create the prompt for Claude Code to generate model architecture."""

        # Load example and extract just the architecture (without metadata)
        examples = self._load_examples()
        example_architecture = examples["examples"]["llama-2-7b-hf"]
        example_json = json.dumps(example_architecture, indent=2)

        return f"""
ultrathink: Generate standardized model architecture JSON for: {model_id}

GOAL: Transform enhanced model structure string into standardized hierarchical JSON architecture representation

Available Resources:
- model_representation_schema.json - Output schema specification
- model_representation_examples.json - Example architectures

PRINCIPLES:
1. Follow the exact format specified in model_representation_schema.json
2. Generate structure-only JSON: layers with name, class, optional repeat, optional sub_layers
3. Basic layers (leaf nodes) = no sub_layers field
4. Composite layers (containers) = have sub_layers field
5. Use full Python class paths from the enhanced structure
6. Detect repeated patterns and use "repeat" field instead of duplicating

INPUT DATA:

Model ID: {model_id}

Enhanced Model Structure (with full class names):
{model_structure}

TODO List:

ultrathink 1. **Parse Structure**: Analyze the enhanced model structure string
   - Extract the hierarchy from indentation levels
   - Identify module names and their full class paths
   - Understand parent-child relationships between modules

ultrathink 2. **Classify Layers**: For each module, determine its type:
   - Basic layers: Leaf nodes without children (e.g., Linear, Embedding, RMSNorm)
   - Composite layers: Containers with children (e.g., LlamaModel, LlamaDecoderLayer, LlamaSdpaAttention)
   - Look for patterns like "(0-31): 32 x LayerName" indicating repetition

ultrathink 3. **Detect Repetition**: Identify repeated structures:
   - Look for patterns like "32 x LlamaDecoderLayer" or "(0-31)" ranges
   - Extract single template and add "repeat": <count> field
   - Ensure we don't duplicate the same layer 32 times

ultrathink 4. **Build Hierarchy**: Create the standardized JSON structure:
   - Start with model_id
   - Build layers array with proper nesting using sub_layers
   - Map each module to its layer representation
   - Preserve hierarchy depth and relationships

ultrathink 5. **Write Architecture**: Create complete JSON matching the schema and write to {output_file}
   - Use exact class paths from enhanced structure
   - Ensure proper basic vs composite layer distinction
   - Follow the example format (DO NOT include $schema, description, or other metadata fields)
   - Output ONLY: model_id and layers fields
   - Validate against schema requirements

ultrathink 6. **Write Diagnostics**: Create diagnostic information and write to {output_file.replace('.json', '_diagnostics.json')} with:
   If successful:
   {{
     "model_analyzed": "{model_id}",
     "status": "success"
   }}
   If failed:
   {{
     "model_analyzed": "{model_id}",
     "status": "fail",
     "reason": "Description of what went wrong"
   }}

## Example Output Format:
{example_json}

IMPORTANT: Output ONLY model_id and layers fields. NO $schema, title, description, or other metadata fields.

"""