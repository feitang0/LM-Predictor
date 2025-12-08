# Session Summary: Template Population for Model Analysis

**Date**: 2024-12-08
**Status**: In Progress - Need to fix model JSON generation

---

# Part 1: Completed - populate_template() Function

## What Was Implemented

Added `populate_template()` function to `model_analyzer.py` that:
1. Takes model JSON with template variables like `{batch_size}`, `{hidden_size}`
2. Substitutes runtime parameters (batch_size, seq_len, cache_len, w_bytes, a_bytes)
3. Substitutes architecture parameters from model config
4. Evaluates formulas to numeric values

### Key Functions Added

```python
# Extracts standardized params from any HuggingFace config
def extract_model_params(config) -> Dict[str, Any]

# Recursively populates template variables in JSON
def _populate_value(self, value, params, is_formula=False) -> Any

# Main API
def populate_template(self, model_json, batch_size, seq_len, ...) -> Dict[str, Any]
```

### CLI Added
```bash
python model_analyzer.py --populate \
    --model_id openai-community/gpt2 \
    --model_json models/openai-community/gpt2.json \
    --batch_size 1 --seq_len 2048 --w_bit 16 --a_bit 16
```

### Config Attribute Mapping
Handles different HuggingFace naming conventions:
- GPT-2: `n_embd` → `hidden_size`, `n_head` → `num_heads`
- Llama: `hidden_size`, `num_attention_heads`
- etc.

---

# Part 2: Outstanding Issue - Context-Dependent Variables

## Problem
When regenerating model JSON, got error:
```
Error: Unresolved template variables: ['output_size', 'output_size']
```

## Root Cause

### Two-Level Template System

1. **Module JSONs** (`modules/*.json`): Use generic variables like `{output_size}` because module doesn't know its context

2. **Model JSONs** (`models/*.json`): Should expand these to global variables based on context

### The Bug
`model_analyzer_agent_sdk.py` copies module templates without expanding context-dependent variables.

---

# Plan: Standardize Template Variables for Model Analysis

## Problem

When running `--populate`, unresolved variables like `{output_size}` cause errors because they are **context-dependent** and cannot be resolved globally.

## Root Cause Analysis

### Two-Level Template System

1. **Module-level JSONs** (`modules/*.json`): Use generic variables like `{output_size}`, `{input_size}`, `{num_elements}` because the module doesn't know its context.

2. **Model-level JSONs** (`models/*.json`): Should expand context-dependent variables to **global** variables based on how each module is used.

### The Bug
`model_analyzer_agent_sdk.py` is copying module templates without expanding context-dependent variables.

## Standardized Global Variables

Model JSONs should ONLY use these variables (resolvable at populate time):

| Variable | Source | Description |
|----------|--------|-------------|
| `{batch_size}` | Runtime | Batch size |
| `{seq_len}` | Runtime | Sequence length |
| `{cache_len}` | Runtime | KV cache length |
| `{hidden_size}` | Config | Model hidden dimension |
| `{num_heads}` | Config | Number of attention heads |
| `{head_dim}` | Derived | hidden_size / num_heads |
| `{intermediate_size}` | Config | MLP intermediate size |
| `{vocab_size}` | Config | Vocabulary size |
| `{num_layers}` | Config | Number of transformer layers |
| `{w_bytes}` | Runtime | Weight precision bytes |
| `{a_bytes}` | Runtime | Activation precision bytes |
| `{has_bias}` | Layer-specific | Boolean for bias presence |

### Context-Dependent Variables (NOT allowed in model JSON)

These must be expanded during model JSON generation:

| Module Variable | Context → Expansion |
|-----------------|---------------------|
| `{output_size}` | lm_head → `{vocab_size}` |
| `{output_size}` | c_attn → `3 * {hidden_size}` |
| `{output_size}` | c_proj → `{hidden_size}` |
| `{output_size}` | c_fc → `{intermediate_size}` |
| `{num_elements}` | MLP activation → `{batch_size} * {seq_len} * {intermediate_size}` |
| `{input_size}` | → context-specific dimension |

## Solution: Fix model_analyzer_agent_sdk.py

The agent prompt must instruct:
1. When expanding a module reference, substitute context-dependent variables
2. Only use global variables in the final model JSON
3. Verify no `{output_size}`, `{input_size}`, `{num_elements}` remain

## Files to Modify

1. **`prompts/model_analyzer.txt`** - Add instructions to expand context-dependent variables
2. **`model_analyzer.py`** - Add validation to check for disallowed variables (optional)

## Validation (Optional Enhancement)

Add to `populate_template()`:
```python
ALLOWED_VARIABLES = {
    'batch_size', 'seq_len', 'cache_len',
    'hidden_size', 'num_heads', 'head_dim',
    'intermediate_size', 'vocab_size', 'num_layers',
    'w_bytes', 'a_bytes', 'has_bias'
}

def _validate_variables(self, formula: str) -> List[str]:
    """Return list of disallowed variables in formula."""
    variables = re.findall(r'\{(\w+)\}', formula)
    return [v for v in variables if v not in ALLOWED_VARIABLES]
```
