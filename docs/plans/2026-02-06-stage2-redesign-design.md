# Two-Phase Cost Analysis

## Overview

Two-phase design for FLOPs/Memory cost analysis:

- **Phase 1 (Top-Down BFS)**: Self-driven BFS starting from a root module. Agent analyzes each kernel — discovers children (module calls + inline ops), writes bindings — and outputs an intermediate cost JSON. The agent's output drives the BFS: each new child kernel reference that doesn't already have a cost JSON is added to the queue. No pre-existing dependency graph required.
- **Phase 2 (Bottom-Up Substitution)**: Pure Python resolves the tree from leaves to root, producing a final recursive cost tree.

Key principles:
- **No leaf/composite distinction** — every kernel is just a kernel. A kernel with an existing cost JSON is skipped; one without is analyzed.
- **Hand-written cost JSONs** for basic ops (in `kernels/` directory) are seeded before BFS starts
- **Self-driven BFS** — Phase 1 discovers the kernel graph and writes cost JSONs in one pass (no separate graph extraction step needed)
- **Per-sub-kernel costs** — costs stay broken down by child, never rolled into one big formula
- **Inline ops as children** — operations within `forward()` (like `a = b + c`) are modeled as synthetic child entries
- **config.xxx is implicit** — shared global context, like batch_size/seq_len
- **count field for multipliers** — `"count": "config.num_hidden_layers"` instead of formula-level `* N`

---

## Intermediate Cost JSON Schema

Two variants enforced by `oneOf`. Used for hand-written leaf kernels and agent-written composite kernels.

### Leaf kernel (hand-written, has formulas, no children)

```json
{
  "kernel_name": "torch.addmm",
  "init_params": [],
  "forward_params": ["M", "K", "N"],
  "flops": "2 * M * K * N + M * N",
  "memory_read": "(M * N + M * K + K * N) * bytes",
  "memory_write": "M * N * bytes"
}
```

### Composite kernel (agent-written, has children, no formulas)

```json
{
  "kernel_name": "GPT2Attention",
  "init_params": [],
  "forward_params": [],
  "children": {
    "c_attn": {
      "kernel": "Conv1D",
      "bindings": { "nf": "3*config.n_embd", "nx": "config.n_embd" },
      "count": 1
    },
    "scale_query": {
      "kernel": "torch.mul",
      "bindings": { "num_elements": "batch_size*seq_len*config.n_embd" },
      "count": 1
    },
    "qk_matmul": {
      "kernel": "torch.matmul",
      "bindings": { "M": "batch_size*config.n_head*seq_len", "K": "config.n_embd//config.n_head", "N": "cache_len" },
      "count": 1
    }
  }
}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `kernel_name` | string | Fully qualified kernel name |
| `init_params` | string[] | Constructor arg names (excluding `self`, `config`, `**kwargs`). Empty for functions or config-only modules |
| `forward_params` | string[] | Input tensor dimensions not in init_params or implicit vars |
| `children` | object | Map of label -> {kernel, bindings, count}. Each label is a unique name for one child usage |
| `children[].kernel` | string | Child kernel's name (matches its cost JSON's `kernel_name`) |
| `children[].bindings` | object | Map from child's param names to expressions in the parent's namespace |
| `children[].count` | int or string | Number of times this child is invoked. Default 1. Can be expression like `"config.num_hidden_layers"` |
| `flops` | string | (Leaf only) Symbolic formula for FLOPs |
| `memory_read` | string | (Leaf only) Symbolic formula for memory read bytes |
| `memory_write` | string | (Leaf only) Symbolic formula for memory write bytes |

### Implicit variables (never in parameters, never passed)

- **Runtime**: `batch_size`, `seq_len`, `cache_len`, `bytes`
- **Config**: `config.xxx` (e.g. `config.n_embd`, `config.n_head`)

These are shared global context available to all kernels.

### Cost computation for composite kernels

Composite kernels have no formula fields. Their total cost is always:

```
total_flops = sum(child.flops * child.count for child in children)
total_memory_read = sum(child.memory_read * child.count for child in children)
total_memory_write = sum(child.memory_write * child.count for child in children)
```

This is computed mechanically in Phase 2, not by the agent.

### Inline operations as children

Operations within `forward()` that aren't child module calls are modeled as synthetic children referencing leaf kernels:

```python
# In GPT2Attention.forward():
query = query * (head_dim ** -0.5)  # element-wise multiply
```

Becomes:
```json
"scale_query": {
  "kernel": "torch.mul",
  "bindings": { "num_elements": "batch_size*seq_len*config.n_embd" },
  "count": 1
}
```

---

## JSON Schemas

### Intermediate schema (`kernel_cost_schema.json`)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Kernel Cost (Intermediate)",
  "type": "object",
  "required": ["kernel_name", "init_params", "forward_params"],
  "properties": {
    "kernel_name": { "type": "string" },
    "init_params": { "type": "array", "items": { "type": "string" } },
    "forward_params": { "type": "array", "items": { "type": "string" } }
  },
  "oneOf": [
    {
      "title": "Leaf kernel",
      "required": ["flops", "memory_read", "memory_write"],
      "not": { "required": ["children"] },
      "properties": {
        "flops": { "type": "string" },
        "memory_read": { "type": "string" },
        "memory_write": { "type": "string" }
      }
    },
    {
      "title": "Composite kernel",
      "required": ["children"],
      "not": {
        "anyOf": [
          { "required": ["flops"] },
          { "required": ["memory_read"] },
          { "required": ["memory_write"] }
        ]
      },
      "properties": {
        "children": {
          "type": "object",
          "minProperties": 1,
          "additionalProperties": {
            "type": "object",
            "required": ["kernel", "bindings"],
            "properties": {
              "kernel": { "type": "string" },
              "bindings": {
                "type": "object",
                "additionalProperties": { "type": "string" }
              },
              "count": {
                "oneOf": [
                  { "type": "integer", "minimum": 1 },
                  { "type": "string" }
                ],
                "default": 1
              }
            },
            "additionalProperties": false
          }
        }
      }
    }
  ]
}
```

### Final resolved schema (`resolved_cost_schema.json`)

Recursive schema for the Phase 2 output tree.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Resolved Cost Tree",
  "type": "object",
  "required": ["kernel_name", "flops", "memory_read", "memory_write"],
  "properties": {
    "kernel_name": { "type": "string" },
    "flops": { "type": "string" },
    "memory_read": { "type": "string" },
    "memory_write": { "type": "string" },
    "children": {
      "type": "object",
      "additionalProperties": { "$ref": "#/$defs/resolved_child" }
    }
  },
  "$defs": {
    "resolved_child": {
      "type": "object",
      "required": ["kernel", "count", "flops", "memory_read", "memory_write"],
      "properties": {
        "kernel": { "type": "string" },
        "count": {
          "oneOf": [
            { "type": "integer", "minimum": 1 },
            { "type": "string" }
          ]
        },
        "flops": { "type": "string" },
        "memory_read": { "type": "string" },
        "memory_write": { "type": "string" },
        "children": {
          "type": "object",
          "additionalProperties": { "$ref": "#/$defs/resolved_child" }
        }
      },
      "additionalProperties": false
    }
  }
}
```

---

## Phase 1: Self-Driven Top-Down BFS

### Processing order

Self-driven BFS starting from a root module name (`--module`). No pre-existing dependency graph required.

1. Seed hand-written leaf kernel JSONs from `kernels/` into `analysis/costs/`
2. Add root module to BFS queue
3. For each kernel in queue:
   - If a cost JSON already exists in `costs/` (hand-written or previously computed) -> **skip**
   - Otherwise -> run agent
   - Read agent's output `children` -> add any child kernel not yet in `costs/` to queue
4. Continue until queue is empty

### Agent's task (per kernel)

1. **Read own source**: Find `__init__` and `forward()` for this kernel
2. **Determine own parameters**:
   - `init_params`: constructor args from `__init__` signature (excluding `self`, `config`, `**kwargs`)
   - `forward_params`: input tensor dimensions in `forward()` that aren't implicit (runtime/config) or init_params
3. **For each child call in `forward()`**:
   - Identify child module calls (e.g. `self.c_attn(x)`) and inline ops (e.g. `query * scale`)
   - Read child's `__init__` and `forward()` **source code** (always available on disk)
   - Trace tensor shapes at the call site
   - Write named bindings mapping child's params to parent's expressions
   - For inline ops, create synthetic child entries referencing leaf kernels
4. **Write intermediate cost JSON** with `children` object

### What the agent reads

| What | Why | Available? |
|------|-----|------------|
| Own `__init__` + `forward()` | Determine parameters, trace shapes | Always (source) |
| Child's `__init__` + `forward()` | Know child's param names, write bindings | Always (source) |
| Child's cost JSON | NOT needed | N/A |

Source code is always available in `transformers/` and `pytorch/` directories, regardless of BFS order. The agent never needs to read any child's cost JSON.

### Inference assumptions

- KV cache is available (conditionals checking for cache -> TRUE)
- `batch_size` for batch dimension
- `seq_len` for current sequence length (1 during decode, full during prefill)
- `cache_len` for KV cache sequence length
- Boolean params (has_bias): resolve to 1 or 0

---

## Phase 2: Bottom-Up Substitution (Python)

Pure Python, deterministic, no agent. Takes all intermediate cost JSONs from Phase 1 and produces one final resolved tree.

### Process (bottom-up)

1. **Order kernels** leaf-to-root by reading `children` references from cost JSONs
2. **For each kernel**, build its resolved template:
   - **Leaf**: template = its own formulas (uses only its params + implicit vars)
   - **Composite**: for each child entry, take the child's resolved template, apply bindings (substitute child param names with bound expressions). The kernel's own cost for each metric = sum of (child cost * count).
3. **Store** the resolved template (still contains the kernel's own param variables like `nx`, `nf`)

### Building the final tree

Start from root. For each child reference:
1. Take the child kernel's resolved template
2. Apply the parent's bindings to eliminate the child's params
3. Recurse into the child's own children, cascading the same substitutions
4. Attach as a resolved child node with `flops`, `memory_read`, `memory_write`, and nested `children`

### Substitution example

**torch.addmm** (leaf, template):
```
flops = "2*M*K*N + M*N"
```

**Conv1D** (resolved template, params: nf, nx):
```
children.addmm:
  flops = "2*(batch_size*seq_len)*nx*nf + (batch_size*seq_len)*nf"
  (M->batch_size*seq_len, K->nx, N->nf applied)
```

**GPT2Attention** (final output, c_attn branch):
```
children.c_attn:
  flops = "2*(batch_size*seq_len)*(config.n_embd)*(3*config.n_embd) + ..."
  (nx->config.n_embd, nf->3*config.n_embd applied)
  children.addmm:
    flops = "2*(batch_size*seq_len)*(config.n_embd)*(3*config.n_embd) + ..."
    (same substitution cascaded down)
```

At the root level, no param variables remain — only implicit vars (`batch_size`, `seq_len`, `cache_len`, `bytes`, `config.xxx`).

---

## End-to-End Data Flow

### Phase 1 BFS (top-down)

**Level 0 — GPT2Attention** (agent runs):
```json
{
  "kernel_name": "GPT2Attention",
  "init_params": [],
  "forward_params": [],
  "children": {
    "c_attn": {
      "kernel": "Conv1D",
      "bindings": { "nf": "3*config.n_embd", "nx": "config.n_embd" },
      "count": 1
    },
    "c_proj": {
      "kernel": "Conv1D",
      "bindings": { "nf": "config.n_embd", "nx": "config.n_embd" },
      "count": 1
    },
    "scale_query": {
      "kernel": "torch.mul",
      "bindings": { "num_elements": "batch_size*seq_len*config.n_embd" },
      "count": 1
    },
    "qk_matmul": {
      "kernel": "torch.matmul",
      "bindings": { "M": "batch_size*config.n_head*seq_len", "K": "config.n_embd//config.n_head", "N": "cache_len" },
      "count": 1
    },
    "softmax": {
      "kernel": "F.softmax",
      "bindings": { "num_elements": "batch_size*config.n_head*seq_len*cache_len" },
      "count": 1
    },
    "av_matmul": {
      "kernel": "torch.matmul",
      "bindings": { "M": "batch_size*config.n_head*seq_len", "K": "cache_len", "N": "config.n_embd//config.n_head" },
      "count": 1
    }
  }
}
```

**Level 1 — Conv1D** (agent runs):
```json
{
  "kernel_name": "Conv1D",
  "init_params": ["nf", "nx"],
  "forward_params": [],
  "children": {
    "addmm": {
      "kernel": "torch.addmm",
      "bindings": { "M": "batch_size*seq_len", "K": "nx", "N": "nf" },
      "count": 1
    }
  }
}
```

**Level 2 — torch.addmm, torch.matmul, etc.**: hand-written, **skip**.

---

## Relationship to graph_extractor.py

`graph_extractor.py` (Stage 1) is kept as-is for cases where you only want the structural dependency graph without costs. Phase 1 (`cost_analyzer.py`) is self-contained and does not require a pre-existing graph — it discovers the kernel tree and writes cost JSONs in one pass.

Phase 2 uses the intermediate cost JSONs from Phase 1 directly. It can determine bottom-up ordering by reading the `children` references in each cost JSON.

---

## Implementation Changes

### Removed
- `atomic_kernels.json` catalog and all catalog lookup logic
- `NAME_MAP` normalization table
- `prepare_sub_kernels` function
- Namespace remapping logic in prompt
- `--single` flag, `--graph` flag
- atomic/composite distinction
- Top-level `flops`/`memory_read`/`memory_write` formula strings on composite kernels
- Dependency on Stage 1 graph output for cost analysis

### New
- `kernels/` directory: hand-written cost JSONs for basic ops
  - Seeded into `analysis/costs/` before Phase 1 starts
  - BFS skips any kernel that already has a cost JSON
- `kernel_name_map.json`: maps fully-qualified names to short leaf kernel names
- `kernel_cost_schema.json`: intermediate schema with `oneOf` (leaf vs composite)
- `resolved_cost_schema.json`: recursive schema for final output tree (Phase 2)

### Changed
- **`cost_analyzer.py`**: Major rewrite
  - `--module` flag replaces `--graph` — self-driven BFS from root module
  - Agent discovers children AND writes bindings in one pass
  - Agent output drives BFS: new child kernels added to queue
- **`prompts/cost_analyzer.txt`**: New prompt — agent reads own + child source, discovers children, traces shapes, writes cost JSON with structured children/bindings

### Hand-written cost JSONs needed

Basic operations that serve as leaves:
- `torch.matmul` / `torch.bmm`
- `torch.addmm`
- `F.linear`
- `F.softmax`
- `F.scaled_dot_product_attention`
- `F.silu` / `F.gelu` / `F.relu` (activation functions)
- `F.dropout`
- `F.layer_norm` / `F.batch_norm`
- `F.embedding`
- `torch.add` / `torch.mul` (element-wise)
- `torch.Tensor.contiguous` / `.clone` (memory ops)
- `torch.Tensor.view` / `.reshape` / `.transpose` (zero-cost ops)

---

## Edge Cases

### Zero-cost operations
Hand-written with `flops: "0"`, `memory_read: "0"`, `memory_write: "0"`. Agent can include or skip them.

### Unknown operations
If an operation has no cost JSON and can't be expressed as a child call, agent writes `"unknown"` for the formula.

### Inheritance (__init__ chains)
Agent reads source code to trace self.xxx -> config.xxx through the __init__ chain. For most HuggingFace models, this is 1-2 hops. Config fields are implicit, so the agent only needs to identify non-config constructor args.

### Config as implicit context
All `config.xxx` fields are shared global context. Modules that take only `config` as constructor arg have `init_params: []`. Only modules with non-config constructor args (like Conv1D's `nf`, `nx`) have non-empty init_params.
