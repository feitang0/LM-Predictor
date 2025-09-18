# Model FLOP/Memory Analysis System

## Overview
A comprehensive system to analyze FLOPs (Floating Point Operations) and memory read/write volumes for PyTorch models by building a reusable database of module computation characteristics, leveraging Claude Code agent for complex forward function analysis.

## Architecture

**Pipeline:**
```
Model → Analyze Architecture → Recursive Layer-by-Layer Computation → Results
  ↓              ↓                           ↓                           ↓
[Input]   Extract Modules           For Each Module:                   Report
          (layer-by-layer)              ↓
                                   Known Module?
                                   ├── Yes: Call generated_modules/ functions
                                   └── No: Agent Analysis
                                           ├── Output analyzed JSON
                                           ├── Store to database
                                           ├── Generate callable module
                                           └── Continue computation
```

### 1. Module Analysis Database
A JSON database storing computational characteristics for each PyTorch module type.

**Naming Convention:**
- **Database Keys**: `library_ClassName` format (e.g., `transformers_LlamaMLP`, `torch_Linear`)
- **Full Class Names**: Complete Python paths (e.g., `transformers.models.llama.modeling_llama.LlamaMLP`)
- **Generated Classes**: `LibraryClassName` format (e.g., `TransformersLlamaMLP`, `TorchLinear`)

**Formula Syntax:**
- **Parameters**: `${param_name}` - represent parameters substituted with actual values
- **Module Calls**: `{ModuleName}` - represent dependent module names for FLOP and memory calculation

```json
{
  "transformers_LlamaAttention": {
    "full_class_name": "transformers.models.llama.modeling_llama.LlamaAttention",
    "code_location": {
      "file": "transformers/src/transformers/models/llama/modeling_llama.py",
      "line_start": 224,
      "line_end": 265
    },
    "flop_analysis": {
      "thinking_process": "Step-by-step reasoning: 1) Q,K,V projections each do matrix multiply of [B,S,H] x [H,H] = 2*B*S*H^2 FLOPs...",
      "parameters": [
        {"name": "B", "type": "int", "description": "batch size"},
        {"name": "S", "type": "int", "description": "sequence length"},
        {"name": "hidden_size", "type": "int", "description": "model hidden dimension"},
        {"name": "num_heads", "type": "int", "description": "number of attention heads"}
      ],
      "calculation_formula": "3 * {torch__nn__Linear}(${B} * ${S}, ${hidden_size}, ${hidden_size}) + 2 * ${B} * ${num_heads} * ${S} * ${S} * (${hidden_size} // ${num_heads}) + {torch__nn__Linear}(${B} * ${S}, ${hidden_size}, ${hidden_size})",
      "module_depends": ["torch__nn__Linear"],
      "breakdown": {
        "q_proj": "{torch__nn__Linear}(${B} * ${S}, ${hidden_size}, ${hidden_size})",
        "k_proj": "{torch__nn__Linear}(${B} * ${S}, ${hidden_size}, ${hidden_size})",
        "v_proj": "{torch__nn__Linear}(${B} * ${S}, ${hidden_size}, ${hidden_size})",
        "attention_scores": "2 * ${B} * ${num_heads} * ${S} * ${S} * (${hidden_size} // ${num_heads})",
        "o_proj": "{torch__nn__Linear}(${B} * ${S}, ${hidden_size}, ${hidden_size})"
      }
    },
    "memory_analysis": {
      "thinking_process": "Memory access pattern: Weight matrices are read once, input activations read once...",
      "parameters": [
        {"name": "B", "type": "int", "description": "batch size"},
        {"name": "S", "type": "int", "description": "sequence length"},
        {"name": "hidden_size", "type": "int", "description": "model hidden dimension"},
        {"name": "dtype_bytes", "type": "int", "description": "bytes per data type element"}
      ],
      "reads_calculation_formula": "4 * ${hidden_size} * ${hidden_size} * ${dtype_bytes} + ${B} * ${S} * ${hidden_size} * ${dtype_bytes}",
      "writes_calculation_formula": "${B} * ${S} * ${hidden_size} * ${dtype_bytes}",
      "intermediates_calculation_formula": "${B} * ${num_heads} * ${S} * ${S} * ${dtype_bytes}",
      "module_depends": ["torch__nn__Linear"]
    },
    "validation": {
      "human_validated": false
    }
  },

  "torch_Linear": {
    "full_class_name": "torch.nn.Linear",
    "code_location": {
      "file": "pytorch/torch/nn/modules/linear.py",
      "line_start": 124,
      "line_end": 125
    },
    "flop_analysis": {
      "thinking_process": "Standard matrix multiplication: input @ weight.T",
      "parameters": [
        {"name": "B", "type": "int", "description": "batch size"},
        {"name": "S", "type": "int", "description": "sequence length"},
        {"name": "input_features", "type": "int", "description": "input feature dimension"},
        {"name": "output_features", "type": "int", "description": "output feature dimension"}
      ],
      "calculation_formula": "2 * ${B} * ${S} * ${input_features} * ${output_features}",
      "module_depends": [],
      "breakdown": {
        "matrix_multiply": "2 * ${B} * ${S} * ${input_features} * ${output_features}"
      }
    },
    "memory_analysis": {
      "thinking_process": "Memory access pattern: Weight matrix read once, input activations read once",
      "parameters": [
        {"name": "B", "type": "int", "description": "batch size"},
        {"name": "S", "type": "int", "description": "sequence length"},
        {"name": "input_features", "type": "int", "description": "input feature dimension"},
        {"name": "output_features", "type": "int", "description": "output feature dimension"},
        {"name": "dtype_bytes", "type": "int", "description": "bytes per data type element"}
      ],
      "reads_calculation_formula": "${input_features} * ${output_features} * ${dtype_bytes} + ${B} * ${S} * ${input_features} * ${dtype_bytes}",
      "writes_calculation_formula": "${B} * ${S} * ${output_features} * ${dtype_bytes}",
      "intermediates_calculation_formula": "0",
      "module_depends": []
    },
    "validation": {
      "human_validated": true
    }
  }
}
```

### 2. Core Components

#### A. Model Analyzer (`model_analyzer.py`)
- Main entry point for model FLOP/memory analysis
- Extract model architecture as nested dict with fully qualified class names
- Build module inventory with library.path.to.ClassName format
- Extract module parameters (hidden_size, num_heads, etc.) for formula substitution
- Orchestrate the entire analysis pipeline and generate comprehensive reports

#### B. Module Analyzer (`module_analyzer.py`)
- Cache-first module analysis: check database before using agent
- For known modules: use cached formula templates from database
- For unknown modules: analyze using Claude Code agent → cache to database
- Generate safe JSON formula templates (not executable code)
- Focus on template generation and database management

#### C. Module Database (`module_db.json`)
- Store formula templates using `${param}` and `{Module}()` syntax (same for FLOP and memory)
- Include thinking process documentation for transparency
- Track original module paths and code locations
- Maintain validation status and dependency graphs

#### D. Module Registry (`generated_modules/registry.py`)
- Auto-discover generated module classes using library namespacing
- Resolve module dependencies with circular detection
- Handle path-to-name conversion: `torch.nn.Linear` → `torch_Linear` (database key)
- Provide user-friendly interface: `compute_flops("torch.nn.Linear", ...)`

#### E. Generated Modules (`generated_modules/`)
- Python classes auto-generated from JSON formula templates
- Flat directory structure with descriptive class names prevents conflicts
- Clean class names: `TorchLinear`, `TransformersLlamaMlp`
- Recursive formula evaluation with parameter substitution

**File Organization:**
- One file per module type for clear separation and maintainability
- Flat structure in generated_modules/ directory
- Library prefix in class names prevents naming conflicts
- CamelCase to snake_case file naming: `TorchLinear` → `torch_linear.py`, `TransformersLlamaMLP` → `transformers_llama_mlp.py`

**Class Structure:**
```python
# generated_modules/torch_linear.py
from .base import BaseModule
from typing import Dict, Any, List

class TorchLinear(BaseModule):
    """Auto-generated FLOP/memory calculator for torch.nn.Linear"""

    def get_required_parameters(self) -> Dict[str, str]:
        """Return parameter names mapped to their types"""
        # e.g., {"B": "int", "S": "int", "input_features": "int", "output_features": "int"}
        # Generated from JSON template parameters field

    def compute_flops(self, **params: Dict[str, Any]) -> int:
        """Calculate FLOPs using agent-analyzed formula"""
        # Generated from JSON template formula_template

    def compute_memory_reads(self, **params: Dict[str, Any]) -> int:
        """Calculate memory reads using agent-analyzed formula"""

    def compute_memory_writes(self, **params: Dict[str, Any]) -> int:
        """Calculate memory writes using agent-analyzed formula"""

    def compute_intermediates(self, **params: Dict[str, Any]) -> int:
        """Calculate intermediate memory using agent-analyzed formula"""
```

## File Structure

```
LM-Predictor/
├── model_analyzer.py        # Main entry point for model FLOP/memory analysis
├── module_analyzer.py       # Cache-first module analysis with Claude Code agent fallback
├── module_generator_agent.py # Claude Code agent for generating module files
├── module_db.json          # Module analysis database (formula templates only)
├── generated_modules/       # Generated Python modules (flat structure)
│   ├── __init__.py         # Registry and convenience functions
│   ├── registry.py         # ModuleRegistry with auto-discovery
│   ├── base.py            # BaseModule abstract class
│   ├── torch_linear.py     # torch.nn.Linear → TorchLinear (database key: torch_Linear)
│   ├── torch_layer_norm.py # torch.nn.LayerNorm → TorchLayerNorm (database key: torch_LayerNorm)
│   ├── transformers_llama_mlp.py # transformers.LlamaMLP → TransformersLlamaMLP (database key: transformers_LlamaMLP)
│   ├── transformers_bert_attention.py # transformers.BertAttention → TransformersBertAttention
│   └── ...
├── transformers/          # Submodule for source code analysis
├── pytorch/               # Submodule for source code analysis
├── DESIGN.md             # This documentation
├── PROGRESS.md           # Implementation tracking
├── SCRATCHPAD.md         # Working notes and issues
└── CLAUDE.md            # Development environment and commands
```