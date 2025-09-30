#!/usr/bin/env python3
"""
Claude Code Agent for populating module parameters from model configuration.
This module uses Claude Code in headless mode to map required parameters to actual values.
"""

import json
import os
import subprocess
import sys
import argparse
from typing import Dict, Any, Optional


class PopulateParametersAgent:
    """Claude Code agent for populating module parameters from model configuration."""

    def __init__(self, claude_command: str = "claude", output_file: str = "populated_parameters.json"):
        """
        Initialize the agent with Claude Code command and output file.

        Args:
            claude_command: Command to invoke Claude Code (default: "claude")
            output_file: Path to output file for populated parameters
        """
        self.claude_command = claude_command
        self.output_file = output_file

    def populate_architecture_with_agent(
        self,
        architecture_json: Dict[str, Any],
        model_config: Dict[str, Any],
        enhanced_model_structure: str
    ) -> Dict[str, Any]:
        """
        Use Claude Code to populate parameters for all basic layers in architecture.

        Runtime parameters (batch_size, seq_len, dtype_bytes) are populated as template
        placeholders (e.g., "{batch_size}") for later substitution.

        Args:
            architecture_json: Standardized hierarchical architecture representation
            model_config: Model configuration from AutoConfig.from_pretrained().to_dict()
            enhanced_model_structure: PyTorch model string representation with full class names

        Returns:
            Architecture JSON with 'parameters' field added to each basic layer.
            Runtime-dependent parameters use template format: "{batch_size}", "{seq_len}", etc.

        Raises:
            ValueError: If any basic layer cannot have parameters populated
            RuntimeError: If agent execution fails
        """
        prompt = self._create_population_prompt(
            architecture_json,
            model_config,
            enhanced_model_structure
        )

        print("=== PROMPT ===")
        print(prompt)
        print("=== END PROMPT ===")

        try:
            # Remove output file if it exists
            if os.path.exists(self.output_file):
                os.remove(self.output_file)

            # Remove diagnostic file if it exists
            diagnostic_file = self.output_file.replace('.json', '_diagnostics.json')
            if os.path.exists(diagnostic_file):
                os.remove(diagnostic_file)

            # Run Claude Code in headless mode with 300s timeout (shorter than other agents)
            result = subprocess.run([
                self.claude_command, "-p", prompt,
                "--dangerously-skip-permissions",
                "--verbose"
            ], capture_output=True, text=True, timeout=300)

            print(f"=== SUBPROCESS RESULT ===")
            print(f"Return code: {result.returncode}")
            print(f"=== STDOUT ===")
            print(result.stdout)
            print(f"=== STDERR ===")
            print(result.stderr)
            print(f"=== END SUBPROCESS RESULT ===")

            if result.returncode != 0:
                raise RuntimeError(
                    f"Claude Code failed with return code {result.returncode}\n"
                    f"STDOUT: {result.stdout}\n"
                    f"STDERR: {result.stderr}"
                )

            print("=== CLAUDE RESPONSE ===")
            print(result.stdout)
            print("=== END CLAUDE RESPONSE ===")

            # Read populated architecture from output file
            if not os.path.exists(self.output_file):
                raise FileNotFoundError(f"Claude did not create {self.output_file} file")

            with open(self.output_file, 'r') as f:
                populated_architecture = json.load(f)

            # Read diagnostic information (for debugging purposes)
            if os.path.exists(diagnostic_file):
                with open(diagnostic_file, 'r') as f:
                    diagnostics = json.load(f)
                print(f"Diagnostics: {diagnostics}")

            # Validate that all basic layers have parameters
            self._validate_populated_architecture(populated_architecture)

            return populated_architecture

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Claude Code parameter population timed out for architecture")
        except json.JSONDecodeError as e:
            print(f"Claude stdout: {result.stdout}")
            print(f"Claude stderr: {result.stderr}")
            raise RuntimeError(f"Failed to parse Claude response: {e}")
        except Exception as e:
            raise RuntimeError(f"Parameter population failed: {e}")

    def _validate_populated_architecture(self, architecture: Dict[str, Any]) -> None:
        """
        Validate that all basic layers in architecture have parameters populated.

        Args:
            architecture: Populated architecture JSON

        Raises:
            ValueError: If any basic layer is missing parameters
        """
        def check_layer(layer: Dict[str, Any], path: str = "") -> None:
            layer_path = f"{path}/{layer.get('name', '?')}"

            # Check if this is a basic layer (no sub_layers)
            if "sub_layers" not in layer:
                # Basic layer must have parameters
                if "parameters" not in layer:
                    raise ValueError(f"Basic layer missing parameters: {layer_path} (class: {layer.get('class')})")
                if not isinstance(layer["parameters"], dict):
                    raise ValueError(f"Invalid parameters for layer {layer_path}: must be a dict")
            else:
                # Composite layer - recurse into sub_layers
                for sub_layer in layer.get("sub_layers", []):
                    check_layer(sub_layer, layer_path)

        # Check all top-level layers
        for layer in architecture.get("layers", []):
            check_layer(layer)

    def _create_population_prompt(
        self,
        architecture_json: Dict[str, Any],
        model_config: Dict[str, Any],
        enhanced_model_structure: str
    ) -> str:
        """Create the prompt for Claude Code to populate architecture parameters."""

        # Format inputs for clarity
        architecture_json_str = json.dumps(architecture_json, indent=2)
        model_config_json = json.dumps(model_config, indent=2)

        return f"""
ultrathink: Populate parameters for all basic layers in model architecture

GOAL: Add 'parameters' field to each basic layer with all required parameter values populated

ARCHITECTURE (structure only - needs parameters):
{architecture_json_str}

INPUT SOURCES:

1. Model Configuration (static parameters):
{model_config_json}

2. Enhanced Model Structure:
{enhanced_model_structure}

RUNTIME TEMPLATE VARIABLES (use as string placeholders):

These parameters vary at runtime and should be populated as template strings:
- "{{batch_size}}" - Batch size dimension (B)
- "{{seq_len}}" - Sequence length dimension (S)
- "{{w_dtype_bytes}}" - Weight precision in bytes (1/2/4/8)
- "{{a_dtype_bytes}}" - Activation precision in bytes (1/2/4/8)

For compound expressions, use Python format string syntax:
- N (total tokens) = "{{{{batch_size}}}} * {{{{seq_len}}}}"

KEY CONCEPTS:

**Basic Layer**: Layer WITHOUT 'sub_layers' field - performs actual computation
  - Examples: torch.nn.modules.linear.Linear, torch.nn.modules.sparse.Embedding
  - These layers MUST have 'parameters' field populated

**Composite Layer**: Layer WITH 'sub_layers' field - organizational container
  - Examples: transformers.models.llama.modeling_llama.LlamaModel
  - These layers do NOT need parameters (no computation)

PARAMETER MAPPING RULES:

**1. Runtime-Dependent Parameters** (use templates):
   - B, batch_size → "{{batch_size}}"
   - S, seq_len → "{{seq_len}}"
   - N (total tokens) → "{{batch_size}} * {{seq_len}}"
   - dtype_bytes (for weights) → "{{w_dtype_bytes}}"
   - dtype_bytes (for activations) → "{{a_dtype_bytes}}"

**2. Static Config Mappings** (use concrete values):
   - hidden_size → model_config["hidden_size"] (e.g., 4096)
   - intermediate_size → model_config["intermediate_size"] (e.g., 11008)
   - num_attention_heads → model_config["num_attention_heads"] (e.g., 32)
   - num_key_value_heads → model_config["num_key_value_heads"] (e.g., 32)
   - vocab_size → model_config["vocab_size"] (e.g., 32000)
   - max_position_embeddings → model_config["max_position_embeddings"] (e.g., 4096)
   - num_hidden_layers → model_config["num_hidden_layers"] (e.g., 32)
   - rms_norm_eps → model_config["rms_norm_eps"] (e.g., 1e-05)

**3. Derived Calculations** (compute from config):
   - head_dim = hidden_size // num_attention_heads (e.g., 4096 // 32 = 128)
   - If num_key_value_heads missing: use num_attention_heads

**4. Enhanced Structure Extraction**:
   - Parse structure string for module-specific parameters (in_features, out_features)
   - Extract tensor shapes and dimensions from enhanced structure

TODO List:

ultrathink 1. **Read Module Registry**: Access generated_modules/registry.py to look up required parameters for each module class
   - Use get_required_parameters(full_class_name) to get parameter requirements
   - Example: get_required_parameters('torch.nn.modules.linear.Linear') returns {{"N": "int", "in_features": "int", ...}}

ultrathink 2. **Traverse Architecture**: Walk through the architecture JSON recursively
   - For each layer, check if it has 'sub_layers' field
   - If NO sub_layers → Basic layer (needs parameters)
   - If HAS sub_layers → Composite layer (recurse into sub_layers)

ultrathink 3. **Populate Each Basic Layer**: For each basic layer found:
   a. Get the layer's 'class' field (full Python class path)
   b. Look up required parameters from registry for that class
   c. Populate parameters using priority order:
      - Runtime parameters → Direct config → Derived → Structure → Defaults
   d. Add 'parameters' field to the layer with populated values

ultrathink 4. **Handle Common Parameters**:
   - For Linear layers: N="{{{{batch_size}}}} * {{{{seq_len}}}}", in_features=4096, out_features=4096, dtype_bytes="{{{{w_dtype_bytes}}}}"
   - For Embedding layers: vocab_size=32000, embedding_dim=4096, dtype_bytes="{{{{w_dtype_bytes}}}}"
   - For RMSNorm: hidden_size=4096, eps=1e-05, dtype_bytes="{{{{a_dtype_bytes}}}}"
   - Use templates for runtime params, concrete values for static params

ultrathink 5. **Write Populated Architecture**: Write to {self.output_file}
   Format: Same architecture structure with 'parameters' added to basic layers
   Example:
   {{
     "model_id": "meta-llama/Llama-2-7b-hf",
     "layers": [
       {{
         "name": "embed_tokens",
         "class": "torch.nn.modules.sparse.Embedding",
         "parameters": {{
           "vocab_size": 32000,
           "embedding_dim": 4096,
           "dtype_bytes": "{{{{w_dtype_bytes}}}}"
         }}
       }},
       {{
         "name": "decoder_layer",
         "class": "transformers.models.llama.modeling_llama.LlamaDecoderLayer",
         "repeat": 32,
         "sub_layers": [
           {{
             "name": "q_proj",
             "class": "torch.nn.modules.linear.Linear",
             "parameters": {{
               "N": "{{{{batch_size}}}} * {{{{seq_len}}}}",
               "in_features": 4096,
               "out_features": 4096,
               "dtype_bytes": "{{{{w_dtype_bytes}}}}"
             }}
           }}
         ]
       }}
     ]
   }}

   CRITICAL: Runtime parameters MUST be strings with template placeholders, NOT numbers!

ultrathink 6. **Write Diagnostics**: Write to {self.output_file.replace('.json', '_diagnostics.json')}
   If successful:
   {{
     "status": "success",
     "basic_layers_populated": <count of basic layers>,
     "composite_layers_skipped": <count of composite layers>
   }}
   If failed:
   {{
     "status": "fail",
     "reason": "Description of what went wrong",
     "problematic_layers": [list of layers that failed]
   }}

CRITICAL:
- You MUST add 'parameters' field to EVERY basic layer (no sub_layers)
- You must NOT add 'parameters' to composite layers (with sub_layers)
- Use the module registry to get required parameters for each layer class

"""


def main():
    """Command-line interface for the populate parameters agent."""
    parser = argparse.ArgumentParser(description='Populate parameters for all basic layers in model architecture')
    parser.add_argument('--architecture-json', type=str, required=True,
                       help='JSON file containing architecture representation')
    parser.add_argument('--model-config', type=str, required=True,
                       help='JSON file containing model configuration')
    parser.add_argument('--model-structure', type=str, required=True,
                       help='Text file containing enhanced model structure')
    parser.add_argument('--output', type=str, default='populated_architecture.json',
                       help='Output file path (default: populated_architecture.json)')
    parser.add_argument('--claude-command', type=str, default='claude',
                       help='Claude Code command (default: claude)')

    args = parser.parse_args()

    try:
        # Load architecture JSON
        with open(args.architecture_json, 'r') as f:
            architecture_json = json.load(f)

        # Load model config
        with open(args.model_config, 'r') as f:
            model_config = json.load(f)

        # Load model structure
        with open(args.model_structure, 'r') as f:
            enhanced_model_structure = f.read()

        agent = PopulateParametersAgent(
            claude_command=args.claude_command,
            output_file=args.output
        )

        result = agent.populate_architecture_with_agent(
            architecture_json=architecture_json,
            model_config=model_config,
            enhanced_model_structure=enhanced_model_structure
        )

        print(f"✅ Architecture population successful!")
        print(f"Populated architecture saved to: {args.output}")
        print(f"\nRuntime parameters use template format:")
        print(f"  {{batch_size}}, {{seq_len}}, {{w_dtype_bytes}}, {{a_dtype_bytes}}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()