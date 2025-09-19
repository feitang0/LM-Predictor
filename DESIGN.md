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

**Schema & Examples:**
- JSON Schema: See `module_db_schema.json` for the complete specification
- Example Entries: See `module_db_examples.json` for torch.nn.Linear and LlamaAttention examples

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
- Focus on calculation formula generation and database management

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
└── CLAUDE.md            # Development environment and commands
```