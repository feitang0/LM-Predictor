# Model FLOP/Memory Analysis System

## Overview
A comprehensive system to analyze FLOPs (Floating Point Operations) and memory read/write volumes for PyTorch models by building a reusable database of module computation characteristics, leveraging Claude Code agent for complex forward function analysis.

## Motivation
- **Accuracy**: Analyze actual forward pass code to get precise FLOP and memory counts
- **Efficiency**: Cache validated analysis functions to avoid re-analyzing common modules
- **Scalability**: Build once, reuse across all models with same module types
- **Intelligence**: Use Claude Code agent to understand complex forward functions

## Architecture

### 1. Module Analysis Database
A JSON database storing computational characteristics for each PyTorch module type:

```json
{
  "transformers__models__llama__modeling__llama__LlamaAttention": {
    "module_path": "transformers.models.llama.modeling_llama.LlamaAttention",
    "code_location": {
      "file": "transformers/src/transformers/models/llama/modeling_llama.py",
      "line_start": 197,
      "line_end": 250
    },
    "flop_analysis": {
      "thinking_process": "Step-by-step reasoning: 1) Q,K,V projections each do matrix multiply of [B,S,H] x [H,H] = 2*B*S*H^2 FLOPs...",
      "parameters": ["B", "S", "hidden_size", "num_heads"],
      "formula_template": "3 * {torch__nn__Linear}(${B} * ${S}, ${hidden_size}, ${hidden_size}) + 2 * ${B} * ${num_heads} * ${S} * ${S} * (${hidden_size} // ${num_heads}) + {torch__nn__Linear}(${B} * ${S}, ${hidden_size}, ${hidden_size})",
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
      "parameters": ["B", "S", "hidden_size", "dtype_bytes"],
      "reads_template": "4 * ${hidden_size} * ${hidden_size} * ${dtype_bytes} + ${B} * ${S} * ${hidden_size} * ${dtype_bytes}",
      "writes_template": "${B} * ${S} * ${hidden_size} * ${dtype_bytes}",
      "intermediates_template": "${B} * ${num_heads} * ${S} * ${S} * ${dtype_bytes}",
      "module_depends": ["torch__nn__Linear"]
    },
    "validation": {
      "status": "pending",
      "validator": null,
      "date": null,
      "notes": "Agent-generated, awaiting human validation"
    }
  },

  "torch__nn__Linear": {
    "module_path": "torch.nn.Linear",
    "code_location": {
      "file": "torch/nn/modules/linear.py",
      "line_start": 103,
      "line_end": 120
    },
    "flop_analysis": {
      "thinking_process": "Standard matrix multiplication: input @ weight.T",
      "parameters": ["B", "S", "input_features", "output_features"],
      "formula_template": "2 * ${B} * ${S} * ${input_features} * ${output_features}",
      "module_depends": [],
      "breakdown": {
        "matrix_multiply": "2 * ${B} * ${S} * ${input_features} * ${output_features}"
      }
    },
    "memory_analysis": {
      "thinking_process": "Memory access pattern: Weight matrix read once, input activations read once",
      "parameters": ["B", "S", "input_features", "output_features", "dtype_bytes"],
      "reads_template": "${input_features} * ${output_features} * ${dtype_bytes} + ${B} * ${S} * ${input_features} * ${dtype_bytes}",
      "writes_template": "${B} * ${S} * ${output_features} * ${dtype_bytes}",
      "intermediates_template": "0",
      "module_depends": []
    },
    "validation": {
      "status": "validated",
      "validator": "manual",
      "date": "2024-01-15",
      "notes": "Standard linear layer - well understood"
    }
  }
}
```

### 2. Nested Formula System

The system uses recursive formula evaluation to handle complex module dependencies:

**Formula Syntax:**
- **Parameters**: `${param_name}` - substituted with actual values
- **Module Calls**: `{ModuleName}(args)` - recursively evaluated sub-modules (context determines FLOPs vs memory)
- **Operations**: Standard mathematical expressions

**FLOP Evaluation Flow:**
```
TransformerBlock Formula
├── {MultiHeadAttention}(B, S, hidden_size, num_heads)
│   ├── 3x {Linear}(B*S, hidden_size, hidden_size)     [Q,K,V projections]
│   ├── 2 * B * num_heads * S^2 * head_dim            [Attention scores]
│   └── {Linear}(B*S, hidden_size, hidden_size)       [Output projection]
├── {LayerNorm}(B*S, hidden_size)
├── {MLP}(B, S, hidden_size, intermediate_size)
│   ├── {Linear}(B*S, hidden_size, intermediate_size)  [Up projection]
│   ├── B * S * intermediate_size                     [Activation FLOPs]
│   └── {Linear}(B*S, intermediate_size, hidden_size)  [Down projection]
└── {LayerNorm}(B*S, hidden_size)
```

**Memory Evaluation Flow:**
```
TransformerBlock Memory
├── {MultiHeadAttention}(B, S, hidden_size, num_heads)
│   ├── 3x {Linear}(B*S, hidden_size, hidden_size)  [Q,K,V weight reads]
│   ├── B * num_heads * S^2 * dtype_bytes                  [Attention matrix intermediates]
│   └── {Linear}(B*S, hidden_size, hidden_size)     [Output weight reads]
├── {LayerNorm}(B*S, hidden_size)                   [Norm parameters]
├── {MLP}(B, S, hidden_size, intermediate_size)
│   ├── {Linear}(B*S, hidden_size, intermediate_size)  [Up weight reads]
│   └── {Linear}(B*S, intermediate_size, hidden_size)  [Down weight reads]
└── {LayerNorm}(B*S, hidden_size)                     [Norm parameters]
```

**Pipeline:**
```
Model → Extract Modules → Formula Resolution → Generated Python → Results
  ↓            ↓                ↓                    ↓              ↓
[Fixed]   named_modules()  Recursive Eval      Auto-Generated   Report
                               ↓
                        Known: Use Cached Database
                        Unknown: Agent Analysis → Cache to Database
```

### 3. Core Components

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
- Handle path-to-name conversion: `torch.nn.Linear` → `torch__nn__Linear`
- Provide user-friendly interface: `compute_flops("torch.nn.Linear", ...)`

#### E. Generated Modules (`generated_modules/*/`)
- Python classes auto-generated from JSON formula templates
- Library-organized directory structure prevents naming conflicts
- Clean class names: `TorchLinear`, `TransformersLlamaAttention`
- Recursive formula evaluation with parameter substitution

## Usage Flow

### 1. Simple Usage
```python
# User-friendly interface hides complexity
from generated_modules import compute_flops

# Analyze single module
flops = compute_flops("torch.nn.Linear",
                     batch_elements=2048, input_features=4096, output_features=4096)

# Analyze composite module (automatic recursion)
flops = compute_flops("transformers.models.llama.modeling_llama.LlamaAttention",
                     B=1, S=2048, hidden_size=4096, num_heads=32)
```

### 2. Full Model Analysis
```python
# Extract all modules from model
model = load_model_on_meta_device(model_id)
analyzer = FlopAnalyzer()

total_flops = 0
for module_name, module in model.named_modules():
    if is_leaf_module(module):
        # Get fully qualified module path
        module_path = f"{module.__class__.__module__}.{module.__class__.__name__}"

        # Registry automatically handles dependencies
        params = extract_module_parameters(module)
        flops = analyzer.compute_flops(module_path, **params)
        total_flops += flops
```

### 3. Recursive Formula Resolution
```python
# Example: TransformerBlock evaluation
# 1. Load "TransformerBlock" template from JSON
# 2. Parse: "{MultiHeadAttention}(B, S, hidden_size, num_heads) + ..."
# 3. Recursively resolve MultiHeadAttention:
#    - Load "MultiHeadAttention" template
#    - Parse: "3 * {Linear}(B*S, hidden_size, hidden_size) + ..."
#    - Resolve Linear (leaf module): "2 * batch_elements * input_features * output_features"
# 4. Substitute parameters and compute final result
```

### 4. Complete Evaluation Example

**Input:** `compute_flops("TransformerBlock", B=1, S=2048, hidden_size=4096, num_heads=32, intermediate_size=11008)`

**Step-by-step Resolution:**
```
TransformerBlock(1, 2048, 4096, 32, 11008):
├── {MultiHeadAttention}(1, 2048, 4096, 32):
│   ├── 3x {Linear}(2048, 4096, 4096):
│   │   └── 3 × (2 * 2048 * 4096 * 4096) = 206,158,430,208
│   ├── Attention: 2 * 1 * 32 * 2048 * 2048 * 128 = 34,359,738,368
│   └── {Linear}(2048, 4096, 4096):
│       └── 2 * 2048 * 4096 * 4096 = 68,719,476,736
│   Total: 309,237,645,312
├── {LayerNorm}(2048, 4096):
│   └── 5 * 2048 * 4096 = 41,943,040
├── {MLP}(1, 2048, 4096, 11008):
│   ├── {Linear}(2048, 4096, 11008): 184,885,575,680
│   ├── Activation: 1 * 2048 * 11008 = 22,544,384
│   └── {Linear}(2048, 11008, 4096): 184,885,575,680
│   Total: 369,793,695,744
└── {LayerNorm}(2048, 4096): 41,943,040

Final: 679,115,227,136 FLOPs
```

### 5. Agent Analysis for Unknown Modules
```python
# When encountering unknown module:
# 1. Agent searches for source code in transformers/pytorch
# 2. Analyzes forward() method implementation
# 3. Generates formula template (not code!) for safety
# 4. Outputs JSON entry: {"formula_template": "...", "parameters": [...]}
# 5. Registry generates Python module from template
# 6. Result cached for future use
```

## Benefits

1. **Accuracy**: Based on actual forward pass code analysis for both FLOPs and memory
2. **Efficiency**: O(1) lookup for analyzed modules vs O(n) code analysis
3. **Intelligence**: Claude Code understands complex implementations
4. **Transparency**: Thinking process documented for each analysis
5. **Extensibility**: Automatically handles new module types
6. **Reproducibility**: Same analysis across different researchers

## File Structure

```
LM-Predictor/
├── model_inspect.py          # Model architecture extraction
├── compute_analyzer.py       # FLOP/memory computation with cache
├── module_analyzer_agent.py  # Claude Code agent integration
├── module_db.json           # Module analysis database (templates only)
├── generated_modules/        # Generated Python modules
│   ├── __init__.py          # Registry and convenience functions
│   ├── registry.py          # ModuleRegistry with auto-discovery
│   ├── base.py             # BaseModule abstract class
│   ├── torch/              # PyTorch core modules
│   │   ├── nn_linear.py    # torch.nn.Linear → TorchLinear
│   │   ├── nn_layer_norm.py # torch.nn.LayerNorm → TorchLayerNorm
│   │   └── ...
│   └── transformers/       # Transformers library modules
│       ├── models_llama_modeling_llama_llama_attention.py
│       ├── models_bert_modeling_bert_bert_attention.py
│       └── ...
├── transformers/          # Submodule for source code analysis
├── DESIGN.md             # This documentation
├── PROGRESS.md           # Implementation tracking
├── SCRATCHPAD.md         # Working notes and issues
└── CLAUDE.md            # Development environment and commands
```

**Key Features:**
- **Library namespacing**: Separate directories prevent naming conflicts
- **Auto-generation**: Python modules generated from JSON templates
- **Registry system**: Automatic discovery and dependency resolution
- **Clean imports**: Simple `from generated_modules import compute_flops`

## Commands

- `uv run python model_inspect.py --model_id <model>` - Extract model architecture with full paths
- `uv run python model_inspect.py --model_id <model> --flops --batch_size <B> --seq_len <S>` - Complete FLOP/memory analysis using registry
- `uv run python module_analyzer_agent.py --module <ModuleName>` - Generate formula template for unknown module

**Usage Examples:**
```bash
# Basic model analysis
uv run python model_inspect.py --model_id meta-llama/Llama-2-7b-hf

# Full FLOP/memory analysis with nested evaluation
uv run python model_inspect.py --model_id meta-llama/Llama-2-7b-hf --flops --batch_size 1 --seq_len 2048

# Analyze specific unknown module (agent will search for it)
uv run python module_analyzer_agent.py --module LlamaRMSNorm
```

This system combines **formula templates** for safety, **nested evaluation** for composability, **library namespacing** for conflict resolution, and **Claude Code agent intelligence** for automatic FLOP/memory analysis of unknown modules.

## Related Documentation

- **[CLAUDE.md](CLAUDE.md)** - Environment setup, commands, and coding guidelines for development
- **[PROGRESS.md](PROGRESS.md)** - Implementation milestones, current progress, and completion tracking
- **[SCRATCHPAD.md](SCRATCHPAD.md)** - Active issues, debugging notes, and experimental ideas