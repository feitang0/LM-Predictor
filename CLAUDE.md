# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@docs/DESIGN.md

## Important Instructions

- Never write or modify code unless explicitly requested by the user.

## Quick Start

LM-Predictor is a Model FLOP/Memory Analysis System that uses AI agents to automatically analyze neural network modules and compute computational costs (FLOPs) and memory access patterns.

### Essential Commands

**Analyze a single module:**
```bash
uv run python module_analyzer_agent_sdk.py \
  --module "transformers.models.llama.modeling_llama.LlamaRMSNorm" \
  --transformers ./transformers \
  --pytorch ./pytorch
```

**Expand module references:**
```bash
uv run python model_analyzer_agent_sdk.py \
  --model "transformers.models.llama.modeling_llama.LlamaMLP" \
  --transformers ./transformers \
  --pytorch ./pytorch \
  --module-analysis ./analysis_results
```

**Inspect model architecture:**
```bash
uv run python model_analyzer.py \
  --model_id "meta-llama/Llama-3.1-8B"
```

## Key Files

- `module_analyzer_agent_sdk.py`: Module analysis script
- `model_analyzer_agent_sdk.py`: Reference expansion script
- `model_analyzer.py`: Model structure inspection
- `module_analysis_schema.json`: Analysis output schema
- `prompts/`: Prompt templates for analysis

## External Dependencies

- **transformers/** submodule: HuggingFace Transformers source
- **pytorch/** submodule: PyTorch source
- Uses `uv` for Python environment management

For detailed architecture, development guidelines, and complete workflow, see the design documentation via the import above.