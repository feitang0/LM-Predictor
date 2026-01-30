# LM-Predictor Design Documentation

This document provides comprehensive design and architecture documentation for the LM-Predictor project.

## Project Overview

LM-Predictor is a Model FLOP/Memory Analysis System that uses a **two-stage kernel dependency graph approach** to analyze neural network computational costs. The system breaks down complex models into a dependency graph of kernels, then computes costs bottom-up from atomic kernels (with known formulas) to the model root.

## Architecture: Two-Stage Kernel Dependency Graph

---

## Stage 1: Kernel Dependency Graph Extraction

### Goal
Build a pure structural dependency graph from model root to atomic kernels. No FLOPs/memory analysis in this stage.

### Input
Model class name (e.g., `transformers.models.llama.modeling_llama.LlamaForCausalLM`)

### Output
Nested JSON tree showing kernel dependencies (see `kernel_dependency_schema.json` for schema):

```json
{
  "kernel_name": "LlamaForCausalLM",
  "kernel_type": "composite",
  "sub_kernels": [
    {"kernel_name": "LlamaModel", "kernel_type": "composite", "sub_kernels": [...]},
    {"kernel_name": "torch.nn.Linear", "kernel_type": "atomic"}
  ]
}
```

### Agent Task
The agent's task is to **identify what modules/functions are called in `forward()`**.

- Read the module's `forward()` method
- List all sub-module calls and tensor operations (both compute AND memory operations)
- Track line ranges where each operation occurs
- Classify each as `composite` (calls other modules) or `atomic` (indivisible operation)
- No FLOPs formulas, no memory analysis, no parameter mapping

**What to track:**
- **Compute operations**: Module calls (self.layer), functions (F.linear, torch.matmul), element-wise ops (x + y)
- **Memory operations**: Layout ops (.contiguous(), .transpose()), shape ops (.view(), .reshape()), copies (.clone()), device transfers (.to(), .cuda())
- **Line ranges**: Single line ("120") or multi-line ("120-125") for each operation

### Kernel Taxonomy

```
Model (root)
  └── Composite Kernel (module that calls other modules/ops)
        └── Composite Kernel
              └── Atomic Kernel (indivisible op with known formula)
```

**Composite Kernel**: A module whose `forward()` calls other modules or operations. Has `sub_kernels` array.

**Atomic Kernel**: An indivisible operation with a known, hardcoded FLOPs/memory formula. No `sub_kernels`.

---

## Stage 2: Bottom-Up FLOPs/Memory Analysis

### Goal
Starting from atomic kernels, compute FLOPs/memory and propagate costs upward to the model root.

### Key Principle: Hardcoded Atomic Formulas
Atomic kernels have **human-maintained formulas**, not agent-generated ones. This ensures accuracy.

### Atomic Kernel Catalog

The `atomic_kernels.json` file contains formulas for all atomic operations:

**Compute Operations:**

| Kernel | Parameters | FLOPs | Memory Read | Memory Write |
|--------|------------|-------|-------------|--------------|
| `torch.matmul` | M, K, N | `2*M*K*N` | `(M*K + K*N) * bytes` | `M*N * bytes` |
| `torch.nn.functional.linear` | M, K, N, has_bias | `2*M*K*N + M*N*has_bias` | `(M*K + K*N + N*has_bias) * bytes` | `M*N * bytes` |
| `torch.nn.functional.softmax` | num_elements | `3*num_elements` | `num_elements * bytes` | `num_elements * bytes` |
| `torch.add` | num_elements | `num_elements` | `2*num_elements * bytes` | `num_elements * bytes` |
| `torch.mul` | num_elements | `num_elements` | `2*num_elements * bytes` | `num_elements * bytes` |
| `torch.nn.Embedding` | batch_size, seq_len, embed_dim | `0` | `batch_size * seq_len * sizeof(index) + batch_size * seq_len * embed_dim * bytes` | `batch_size * seq_len * embed_dim * bytes` |
| `torch.nn.LayerNorm` | num_elements, hidden_size | `5*num_elements` | `num_elements * bytes + 2 * hidden_size * bytes` | `num_elements * bytes` |
| `torch.nn.RMSNorm` | num_elements, hidden_size | `3*num_elements` | `num_elements * bytes + hidden_size * bytes` | `num_elements * bytes` |

**Memory Operations:**

| Kernel | Parameters | FLOPs | Memory Read | Memory Write |
|--------|------------|-------|-------------|--------------|
| `torch.Tensor.contiguous` | num_elements (if not already contiguous) | `0` | `num_elements * bytes` | `num_elements * bytes` |
| `torch.Tensor.view` | - | `0` | `0` | `0` (no-op, just changes view) |
| `torch.Tensor.reshape` | num_elements (if copy needed) | `0` | `num_elements * bytes` | `num_elements * bytes` |
| `torch.Tensor.transpose` | - | `0` | `0` | `0` (changes stride, no copy) |
| `torch.Tensor.clone` | num_elements | `0` | `num_elements * bytes` | `num_elements * bytes` |

### Parameter Resolution (Bottom-Up)

The agent resolves parameters from **child to parent** context:

1. **Atomic kernel declares what it needs**: e.g., `torch.matmul` needs `M`, `K`, `N`
2. **Agent analyzes parent context**: How is this kernel called? What are the tensor shapes?
3. **Agent maps symbolic params to concrete expressions**

**Example**:
```python
# In LlamaAttention.forward():
attn_output = torch.matmul(attn_weights, value_states)
# attn_weights shape: (batch_size, num_heads, seq_len, seq_len)
# value_states shape: (batch_size, num_heads, seq_len, head_dim)

# Agent determines:
# M = batch_size * num_heads * seq_len
# K = seq_len
# N = head_dim

# Resolved FLOPs = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
```

### Aggregation

Once all atomic kernels have resolved formulas:
1. Sum atomic costs to get parent composite kernel cost
2. Sum parent costs to get grandparent cost
3. Continue until reaching model root
4. Handle `multiplier` fields by multiplying (e.g., `config.num_hidden_layers`)

---

## Schema Definitions

### kernel_dependency_schema.json (Stage 1 Output)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["kernel_name", "kernel_type"],
  "properties": {
    "kernel_name": {
      "type": "string",
      "description": "Fully qualified kernel name (e.g., torch.nn.Linear)"
    },
    "kernel_type": {
      "type": "string",
      "enum": ["composite", "atomic"],
      "description": "composite has sub_kernels, atomic is indivisible"
    },
    "multiplier": {
      "type": "string",
      "description": "Expression for multiplier (e.g., config.num_hidden_layers)"
    },
    "sub_kernels": {
      "type": "array",
      "items": { "$ref": "#" },
      "description": "Child kernels (only for composite type)"
    }
  }
}
```

### atomic_kernels.json (Atomic Kernel Catalog)

```json
{
  "torch.matmul": {
    "parameters": ["M", "K", "N"],
    "flops": "2 * M * K * N",
    "memory_read": "(M * K + K * N) * bytes",
    "memory_write": "M * N * bytes",
    "description": "Matrix multiplication of (M, K) x (K, N) -> (M, N)"
  },
  "torch.nn.functional.linear": {
    "parameters": ["M", "K", "N", "has_bias"],
    "flops": "2 * M * K * N + M * N * has_bias",
    "memory_read": "(M * K + K * N + N * has_bias) * bytes",
    "memory_write": "M * N * bytes",
    "description": "Linear transformation: output = input @ weight.T + bias"
  }
}
```

---

## Standardized Variable Names

All formulas use these canonical variable names:

### Runtime Variables
- `batch_size`: Batch size
- `seq_len`: Sequence length (input)
- `cache_len`: KV cache length (for cached attention)
- `bytes`: Precision in bytes (e.g., 2 for fp16, 4 for fp32)

### Model Config Variables
- `config.hidden_size`: Hidden dimension
- `config.num_attention_heads`: Number of attention heads
- `config.num_key_value_heads`: Number of KV heads (for GQA)
- `config.intermediate_size`: FFN intermediate dimension
- `config.vocab_size`: Vocabulary size
- `config.num_hidden_layers`: Number of transformer layers

### Derived Variables
- `head_dim`: `config.hidden_size / config.num_attention_heads`
- `num_elements`: Total elements in tensor (product of dimensions)

---

## Agent Task Simplification

### Stage 1 Agent: Graph Extractor
**Single question**: "What does this module call in its `forward()` method?"

- Read source code
- Identify sub-module calls and tensor operations
- Output nested structure
- No formulas, no parameter analysis

### Stage 2 Agent: Cost Analyzer
**Single question**: "What are M, K, N (etc.) in this context?"

- Given an atomic kernel and its parent context
- Determine tensor shapes at call site
- Map to atomic kernel's required parameters
- No structure discovery, no formula creation

---

## File Structure

```
LM-Predictor/
├── CLAUDE.md                      # Quick reference
├── docs/
│   └── DESIGN.md                  # This file
├── atomic_kernels.json            # Hardcoded atomic kernel formulas
├── kernel_dependency_schema.json  # Schema for Stage 1 output
├── model_analyzer.py              # Model architecture inspection
├── graph_extractor.py             # Stage 1: Kernel dependency graph builder
├── prompts/
│   ├── kernel_analyzer.txt        # Stage 1: Kernel analyzer prompt
│   └── cost_analyzer.txt          # Stage 2: Cost analyzer prompt (future)
├── utils/
│   ├── __init__.py
│   └── message_printer.py         # Pretty-print Claude Agent SDK messages
├── graphs/                        # Stage 1 outputs (dependency graphs)
│   └── kernels/                   # Individual kernel analyses
├── analysis/                      # Stage 2 outputs (cost analysis)
├── transformers/                  # HuggingFace Transformers submodule
└── pytorch/                       # PyTorch submodule
```

---

## Utilities

### message_printer.py

Pretty-printing utilities for Claude Agent SDK messages during graph extraction.

**Functions:**
- `print_pretty_message(message)`: Color-coded formatted output with icons (📝, 💭, 🔧, ✅)
  - Used by `graph_extractor.py` to display agent progress
  - Shows text blocks, thinking blocks, and tool use with proper formatting
- `print_simple_message(message)`: Plain text output without colors (for logging or non-terminal output)

**Features:**
- Visual separation of message types (System, User, Assistant, Result)
- Syntax highlighting for different content blocks
- Compact display for tool inputs using pprint

---

## Workflow

1. **Inspect Model Architecture**
   ```bash
   uv run python model_analyzer.py --model_id "meta-llama/Llama-3.1-8B"
   ```
   Shows model structure with fully qualified class names.

2. **Extract Kernel Dependency Graph (Stage 1)**
   ```bash
   # Future: graph_extractor.py
   ```
   Produces `graphs/{ModelName}.json` with pure structure.

3. **Compute FLOPs/Memory (Stage 2)**
   ```bash
   # Future: cost_analyzer.py
   ```
   Produces `analysis/{ModelName}.json` with resolved formulas.

---

## Advantages of This Approach

1. **Simpler prompts**: Each stage has a single, focused task
2. **Human-verified formulas**: Atomic kernels are hardcoded, not agent-generated
3. **Compositional**: Dependency graphs are reusable across models sharing modules
4. **Guaranteed completeness**: Can't compute parent cost without all children resolved
5. **Easier debugging**: Structure errors (Stage 1) vs. formula errors (Stage 2) are separated
6. **Incremental updates**: Adding new atomic kernels doesn't require re-analyzing models

---

## External Dependencies

- **transformers/** submodule: HuggingFace Transformers source code
- **pytorch/** submodule: PyTorch source code for operation details
- **uv**: Python environment management
