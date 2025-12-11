# LM-Predictor Design Documentation

This document provides comprehensive design and architecture documentation for the LM-Predictor project. For quick start and essential guidance, refer to the main [CLAUDE.md](../CLAUDE.md) file.

## Project Overview

LM-Predictor is a Model FLOP/Memory Analysis System that uses AI agents (Claude Agent SDK) to automatically analyze neural network modules and compute their computational costs (FLOPs) and memory access patterns. The system performs hierarchical analysis of PyTorch and Transformers modules to predict inference costs.

## Architecture

### Two-Stage Analysis Pipeline

1. **Module Analysis** (`module_analyzer_agent_sdk.py`):
   - Analyzes individual module `forward()` methods line-by-line
   - Extracts computational kernels (basic operations and composite module calls)
   - Uses `${{ClassName}}(params)` notation to reference sub-modules that need separate analysis
   - Outputs: `module_analysis.json` following `module_analysis_schema.json`
   - Uses Claude Agent SDK to autonomously navigate code and perform analysis

2. **Model Analysis** (`model_analyzer_agent_sdk.py`):
   - Expands module references recursively to create fully-resolved computational graphs
   - Replaces all `${{ClassName}}(params)` references with actual kernels from referenced modules
   - Creates nested structure showing hierarchical decomposition of operations
   - Outputs: `model_analysis.json` with fully expanded nested kernels

### Core Components

- **Module Analyzer**: Analyzes a single module's forward pass, identifying both:
  - **Basic kernels** (`kernel_type: "basic"`): Direct tensor operations with explicit FLOPs formulas (e.g., `torch.matmul`, element-wise ops)
  - **Composite kernels** (`kernel_type: "composite"`): Module calls that reference other modules using `${{...}}` syntax

- **Model Analyzer** (`model_analyzer.py`): Legacy script that loads model architectures on meta device to inspect structure without loading weights

- **Prompt Templates**: Located in `prompts/` directory:
  - `module_analyzer.txt`: Instructs agent how to analyze module forward methods
  - `model_analyzer.txt`: Instructs agent how to expand module references recursively

### Schema System

`module_analysis_schema.json` defines the structure for analysis output:
- `class_name`: Fully qualified Python class name
- `kernels[]`: Array of computational kernels with:
  - `kernel_type`: "basic" or "composite"
  - `operation`: Human-readable description
  - `analysis`: Detailed explanation of computation
  - `flops`: Formula using standardized variables (`{batch_size}`, `{seq_len}`, etc.)
  - `memory_access`: {read, write} formulas in bytes

### Standardized Variable Names

All formulas use these canonical variable names:
- `batch_size`: Batch size
- `seq_len`: Sequence length
- `cache_len`: KV cache length
- `w_bytes`: Weight precision in bytes
- `a_bytes`: Activation precision in bytes
- `hidden_size`, `num_heads`, `head_dim`: Model architecture parameters

## Running Analysis

### Analyze a Single Module

```bash
uv run python module_analyzer_agent_sdk.py \
  --module "transformers.models.llama.modeling_llama.LlamaRMSNorm" \
  --transformers ./transformers \
  --pytorch ./pytorch
```

This analyzes the module's forward method and produces `module_analysis.json`.

### Expand Module References

```bash
uv run python model_analyzer_agent_sdk.py \
  --model "transformers.models.llama.modeling_llama.LlamaMLP" \
  --transformers ./transformers \
  --pytorch ./pytorch \
  --module-analysis ./analysis_results
```

This reads the module's analysis and expands all `${{...}}` references using analyses from `./analysis_results/`.

### Inspect Model Architecture

```bash
uv run python model_analyzer.py \
  --model_id "meta-llama/Llama-3.1-8B"
```

Loads model structure on meta device and prints architecture with fully qualified class names.

## Development Guidelines

- **Analysis Accuracy**: The agent must analyze EVERY computational line in the forward method, using actual line numbers from source code
- **Inference Path Only**: Analyze only the default inference execution path (skip training-specific branches)
- **No Guessing**: If source code cannot be found, STOP and document in SCRATCHPAD.md - never make assumptions
- **Formula Precision**: All FLOPs formulas must account for every operation; memory access must be in bytes
- **Module References**: Use `${{fully.qualified.ClassName}}(params)` for module calls that will be analyzed separately

## Key Files

- `module_analyzer_agent_sdk.py`: Claude Agent SDK script for module analysis
- `model_analyzer_agent_sdk.py`: Claude Agent SDK script for reference expansion
- `model_analyzer.py`: Legacy model structure inspection tool
- `module_analysis_schema.json`: JSON schema for analysis output
- `prompts/analyze_module_forward.txt`: Module analysis prompt template
- `prompts/expand_module_references.txt`: Reference expansion prompt template
- `SCRATCHPAD.md`: Working document for agent's analysis process (generated)

## External Dependencies

- **transformers/** submodule: HuggingFace Transformers library source code for analysis
- **pytorch/** submodule: PyTorch library source code for low-level operation analysis
- Uses `uv` for Python environment management (see `pyproject.toml`)

## Analysis Workflow

1. Identify target module class (e.g., `torch.nn.Linear`, `LlamaAttention`)
2. Run module analyzer to produce initial analysis with `${{...}}` references
3. Analyze all referenced sub-modules (recursively)
4. Run model analyzer to expand references into fully resolved computational graph
5. Result: Complete hierarchical breakdown of FLOPs and memory access per operation

## Related Documentation

- `compiled-noodling-kettle.md`: Session summary discussing template population for model analysis and issues with context-dependent variables.