# Populate Parameters Agent Specification

## Overview

The Populate Parameters Agent is a critical component in the LM-Predictor pipeline that bridges the gap between module analysis and FLOP/memory computation. It takes required parameters from generated modules and populates them with actual values extracted from model configuration, enhanced model structure, and standard JSON representation.

**Pipeline Position:**
```
Module Analysis → Parameter Population → FLOP/Memory Computation
```

## Agent Interface

### Class Definition
```python
class PopulateParametersAgent:
    def __init__(self, claude_command: str = "claude", output_file: str = "populated_parameters.json")

    def populate_parameters_with_agent(
        self,
        module_class: str,
        required_parameters: Dict[str, str],
        model_config: Dict[str, Any],
        enhanced_model_structure: str,
        architecture_json: Dict[str, Any],
        runtime_parameters: Optional[Dict[str, int]] = None
    ) -> Dict[str, Any]
```

### File Location
- **Implementation:** `populate_parameters_agent.py`
- **Output file:** Configurable, defaults to `populated_parameters.json`
- **Diagnostics:** `populated_parameters_diagnostics.json` (only on failure)

## Input Specification

### 1. module_class (str)
**Purpose:** Full Python class path of the module being analyzed
**Format:** `"library.module.submodule.ClassName"`
**Examples:**
- `"torch.nn.modules.linear.Linear"`
- `"transformers.models.llama.modeling_llama.LlamaSdpaAttention"`
- `"torch.nn.modules.sparse.Embedding"`

### 2. required_parameters (Dict[str, str])
**Purpose:** Parameter requirements from module's `get_required_parameters()` method
**Format:** `{"param_name": "param_type", ...}`
**Constraint:** All values must be valid Python type names
**Examples:**
```json
{
  "B": "int",
  "S": "int",
  "hidden_size": "int",
  "num_attention_heads": "int",
  "num_key_value_heads": "int",
  "head_dim": "int",
  "dtype_bytes": "int"
}
```

### 3. model_config (Dict[str, Any])
**Purpose:** Model configuration from `AutoConfig.from_pretrained().to_dict()`
**Format:** Standard transformers configuration dictionary
**Required Fields:** Must contain at least one mappable parameter
**Example:**
```json
{
  "vocab_size": 32000,
  "max_position_embeddings": 4096,
  "hidden_size": 4096,
  "intermediate_size": 11008,
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  "num_key_value_heads": 32,
  "torch_dtype": "float16",
  "rms_norm_eps": 1e-05
}
```

### 4. enhanced_model_structure (str)
**Purpose:** PyTorch model string representation with full class names
**Format:** Multi-line string with full class paths and tensor shapes
**Example:**
```
LlamaForCausalLM(
  (model): transformers.models.llama.modeling_llama.LlamaModel(
    (embed_tokens): torch.nn.modules.sparse.Embedding(32000, 4096)
    (layers): torch.nn.modules.container.ModuleList(
      (0-31): 32 x transformers.models.llama.modeling_llama.LlamaDecoderLayer(
        (self_attn): transformers.models.llama.modeling_llama.LlamaSdpaAttention(
          (q_proj): torch.nn.modules.linear.Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): torch.nn.modules.linear.Linear(in_features=4096, out_features=4096, bias=False)
        )
      )
    )
  )
)
```

### 5. architecture_json (Dict[str, Any])
**Purpose:** Standardized hierarchical representation following model_representation_schema.json
**Format:** JSON object with model_id and layers array
**Example:**
```json
{
  "model_id": "meta-llama/Llama-2-7b-hf",
  "layers": [
    {
      "name": "model",
      "class": "transformers.models.llama.modeling_llama.LlamaModel",
      "sub_layers": [
        {
          "name": "embed_tokens",
          "class": "torch.nn.modules.sparse.Embedding"
        },
        {
          "name": "decoder_layer",
          "class": "transformers.models.llama.modeling_llama.LlamaDecoderLayer",
          "repeat": 32
        }
      ]
    }
  ]
}
```

### 6. runtime_parameters (Optional[Dict[str, int]])
**Purpose:** User-provided runtime values for batch size and sequence length
**Format:** `{"param_name": value, ...}`
**Default:** `None` (agent will use defaults)
**Example:**
```json
{
  "B": 16,
  "S": 512
}
```

## Output Specification

### Success Output
**Type:** `Dict[str, Any]`
**Format:** All required parameters populated with actual values
**Constraint:** Must contain all parameters from `required_parameters` keys

**Example:**
```json
{
  "B": 16,
  "S": 512,
  "hidden_size": 4096,
  "num_attention_heads": 32,
  "num_key_value_heads": 32,
  "head_dim": 128,
  "intermediate_size": 11008,
  "vocab_size": 32000,
  "dtype_bytes": 2
}
```

### Error Handling
**Missing Required Parameter:** Raise `ValueError` with specific parameter name
**Invalid Type:** Raise `TypeError` with parameter name and expected type
**Agent Failure:** Raise `RuntimeError` with agent error details

## Parameter Mapping Rules

### Priority Order
1. **Runtime Parameters** - Direct user input (highest priority)
2. **Direct Config Mapping** - Exact key match in model_config
3. **Derived Calculations** - Computed from other config values
4. **Enhanced Structure Extraction** - Parsed from model structure
5. **Default Values** - Fallback defaults (lowest priority)

### Direct Config Mappings
```python
DIRECT_MAPPINGS = {
    "hidden_size": "hidden_size",
    "intermediate_size": "intermediate_size",
    "num_attention_heads": "num_attention_heads",
    "num_key_value_heads": "num_key_value_heads",
    "vocab_size": "vocab_size",
    "max_position_embeddings": "max_position_embeddings",
    "num_hidden_layers": "num_hidden_layers",
    "rms_norm_eps": "rms_norm_eps"
}
```

### Derived Calculations
```python
DERIVED_CALCULATIONS = {
    "head_dim": lambda config: config["hidden_size"] // config["num_attention_heads"],
    "dtype_bytes": lambda config: {
        "float16": 2, "bfloat16": 2, "float32": 4, "float64": 8
    }.get(config.get("torch_dtype", "float32"), 4)
}
```

### Default Values
```python
DEFAULT_VALUES = {
    "B": 1,           # Batch size
    "S": 512,         # Sequence length
    "dtype_bytes": 4  # float32 fallback
}
```

### Special Parameter Handling

#### Runtime Dimensions (B, S)
1. Check `runtime_parameters` first
2. For `S`: Use `max_position_embeddings` if available
3. Use defaults as last resort

#### Multi-Source Parameters
Some parameters may need values from multiple sources:
- `num_key_value_heads`: Check config, fallback to `num_attention_heads`
- `head_dim`: Always calculate from `hidden_size / num_attention_heads`

## Complete Example

### Input
```python
module_class = "transformers.models.llama.modeling_llama.LlamaSdpaAttention"
required_parameters = {
    "B": "int",
    "S": "int",
    "hidden_size": "int",
    "num_attention_heads": "int",
    "num_key_value_heads": "int",
    "head_dim": "int",
    "dtype_bytes": "int"
}
model_config = {
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 32,
    "torch_dtype": "float16",
    "max_position_embeddings": 4096
}
runtime_parameters = {"B": 8, "S": 1024}
```

### Expected Output
```python
{
    "B": 8,              # From runtime_parameters
    "S": 1024,           # From runtime_parameters
    "hidden_size": 4096, # From model_config
    "num_attention_heads": 32,     # From model_config
    "num_key_value_heads": 32,     # From model_config
    "head_dim": 128,     # Calculated: 4096 // 32
    "dtype_bytes": 2     # Calculated: float16 → 2
}
```

## Implementation Requirements

### Required Imports
```python
import json
import os
import subprocess
from typing import Dict, Any, Optional
from dotenv import dotenv_values
```

### Claude Code Integration
- Use headless mode with `--dangerously-skip-permissions`
- Timeout: 300 seconds (shorter than other agents)
- Environment: Include `.env` variables

### Error Handling
- Validate all inputs before processing
- Provide specific error messages for missing parameters
- Include parameter name and expected type in errors
- Log all agent communications for debugging

### Logging Requirements
- Log all parameter mappings applied
- Log any fallback values used
- Log calculation formulas for derived parameters
- Include timing information

### Output File Requirements
- Write results to configurable output file
- Diagnostics only on failure
- Clean up output file before each run
- Validate output against required_parameters

## Agent Prompt Structure

The agent prompt should include:
1. Clear goal statement
2. All input data formatted clearly
3. Mapping rules and priority order
4. Expected output format
5. Error handling instructions
6. Examples for complex calculations

**Key Instruction:** The agent must populate ALL parameters from `required_parameters` - no missing values allowed.