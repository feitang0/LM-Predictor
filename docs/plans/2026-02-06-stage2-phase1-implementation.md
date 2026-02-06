# Stage 2 Phase 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement Phase 1 (top-down BFS agent analysis) of the Stage 2 redesign, producing intermediate cost JSONs for each kernel in the dependency graph.

**Architecture:** Rewrite `cost_analyzer.py` to BFS through the Stage 1 dependency graph top-down. For each composite kernel, run a Claude agent that reads source code, identifies children (module calls + inline ops), writes bindings, and outputs an intermediate cost JSON. Hand-written leaf kernel JSONs are pre-seeded and skipped during BFS.

**Tech Stack:** Python 3.12, claude-agent-sdk, jsonschema, existing utils/message_printer.py

---

### Task 1: Create hand-written leaf kernel cost JSONs

Create the `kernels/` directory with cost JSONs for basic operations. These serve as pre-seeded leaves that BFS will skip.

**Files:**
- Create: `kernels/torch.matmul.json`
- Create: `kernels/torch.mm.json`
- Create: `kernels/torch.bmm.json`
- Create: `kernels/torch.addmm.json`
- Create: `kernels/torch.baddbmm.json`
- Create: `kernels/F.linear.json`
- Create: `kernels/F.softmax.json`
- Create: `kernels/F.scaled_dot_product_attention.json`
- Create: `kernels/F.silu.json`
- Create: `kernels/F.gelu.json`
- Create: `kernels/F.relu.json`
- Create: `kernels/F.dropout.json`
- Create: `kernels/F.layer_norm.json`
- Create: `kernels/F.embedding.json`
- Create: `kernels/torch.add.json`
- Create: `kernels/torch.mul.json`
- Create: `kernels/torch.div.json`

**Step 1: Create the kernels directory**

Run: `mkdir -p /Users/tangfei/Dev/LM-Predictor/kernels`

**Step 2: Write each leaf kernel JSON**

Each file follows the intermediate leaf schema. Convert from `atomic_kernels.json` (which uses `parameters`) to the new format (which uses `forward_params`). Leaf kernels have `init_params: []` (functions don't have constructors) and no `children`.

Example — `kernels/torch.matmul.json`:
```json
{
  "kernel_name": "torch.matmul",
  "init_params": [],
  "forward_params": ["M", "K", "N"],
  "flops": "2 * M * K * N",
  "memory_read": "(M * K + K * N) * bytes",
  "memory_write": "M * N * bytes"
}
```

Example — `kernels/F.linear.json`:
```json
{
  "kernel_name": "F.linear",
  "init_params": [],
  "forward_params": ["M", "K", "N", "has_bias"],
  "flops": "2 * M * K * N + M * N * has_bias",
  "memory_read": "(M * K + K * N + N * has_bias) * bytes",
  "memory_write": "M * N * bytes"
}
```

Example — `kernels/F.dropout.json`:
```json
{
  "kernel_name": "F.dropout",
  "init_params": [],
  "forward_params": [],
  "flops": "0",
  "memory_read": "0",
  "memory_write": "0"
}
```

Convert ALL entries from `atomic_kernels.json` categories: matrix_operations, activations, normalizations, element_wise, special_operations. Skip pooling/convolutions/upsample/einsum for now (not needed for transformer models). Also add `torch.div` (element-wise division, same pattern as `torch.mul`).

**Step 3: Validate all kernel JSONs against the intermediate schema**

Run: `uv run python -c "import json, jsonschema, glob; schema=json.load(open('kernel_cost_schema.json')); [jsonschema.validate(json.load(open(f)), schema) for f in glob.glob('kernels/*.json')]; print('All valid')"`

Expected: "All valid" (this will fail until Task 2 writes the schema)

**Step 4: Commit**

```bash
git add kernels/
git commit -m "feat: add hand-written leaf kernel cost JSONs"
```

---

### Task 2: Write the intermediate cost JSON schema

Replace `kernel_cost_schema.json` with the new schema from the design doc.

**Files:**
- Modify: `kernel_cost_schema.json` (full rewrite)

**Step 1: Write the schema file**

Replace `kernel_cost_schema.json` with the schema from the design document (Section "JSON Schemas" > "Intermediate schema"). This is the `oneOf` schema with leaf vs composite variants.

Full content:
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

**Step 2: Validate leaf kernel JSONs against the new schema**

Run: `uv run python -c "import json, jsonschema, glob; schema=json.load(open('kernel_cost_schema.json')); [jsonschema.validate(json.load(open(f)), schema) for f in sorted(glob.glob('kernels/*.json'))]; print(f'All {len(list(glob.glob(\"kernels/*.json\")))} files valid')"`

Expected: "All N files valid"

**Step 3: Commit**

```bash
git add kernel_cost_schema.json
git commit -m "feat: replace kernel cost schema with intermediate oneOf schema"
```

---

### Task 3: Create the kernel name mapping

The Stage 1 dependency graph uses fully qualified names (e.g., `torch.nn.functional.softmax`, `torch.nn.Linear`) while our leaf kernel JSONs use short names (e.g., `F.softmax`, `F.linear`). We need a mapping to resolve these.

**Files:**
- Create: `kernel_name_map.json`

**Step 1: Write the mapping file**

This consolidates the current `NAME_MAP` from `cost_analyzer.py` into a standalone JSON file. The mapping goes from fully-qualified name -> leaf kernel JSON name.

```json
{
  "torch.nn.functional.softmax": "F.softmax",
  "torch.nn.functional.linear": "F.linear",
  "torch.nn.functional.silu": "F.silu",
  "torch.nn.functional.gelu": "F.gelu",
  "torch.nn.functional.relu": "F.relu",
  "torch.nn.functional.dropout": "F.dropout",
  "torch.nn.functional.layer_norm": "F.layer_norm",
  "torch.nn.functional.embedding": "F.embedding",
  "torch.nn.functional.scaled_dot_product_attention": "F.scaled_dot_product_attention",
  "torch.nn.Dropout": "F.dropout",
  "torch.nn.Linear": "F.linear",
  "torch.nn.Embedding": "F.embedding",
  "torch.nn.LayerNorm": "F.layer_norm",
  "torch.nn.SiLU": "F.silu",
  "torch.nn.GELU": "F.gelu",
  "torch.nn.ReLU": "F.relu"
}
```

**Step 2: Commit**

```bash
git add kernel_name_map.json
git commit -m "feat: add kernel name mapping from fully-qualified to short names"
```

---

### Task 4: Write the new agent prompt

Replace `prompts/cost_analyzer.txt` with the new Phase 1 prompt. The agent's job is now different: read own + child source, determine parameters, trace shapes, write children with bindings.

**Files:**
- Modify: `prompts/cost_analyzer.txt` (full rewrite)

**Step 1: Write the new prompt**

The prompt should instruct the agent to:

1. Read its own `__init__` and `forward()` source code
2. Determine `init_params` (constructor args excluding self/config/**kwargs) and `forward_params`
3. For each child call in `forward()`:
   - **Module calls** (`self.xxx(...)`): look up `self.xxx` type from `__init__`, read child source to understand its parameter names, write bindings
   - **Inline operations** (`a + b`, `x * scale`, `torch.matmul(a, b)`): create synthetic child entries referencing leaf kernels (torch.add, torch.mul, etc.) with bindings
4. Write the intermediate cost JSON with `children` object (no formula strings)
5. Use `count` field for repeated children (loops)

Key differences from old prompt:
- No `sub_kernels` array — uses `children` object with named labels
- No formula strings on composite kernels — just children with bindings
- Agent must identify inline ops and model them as children
- Agent reads child source to understand parameter names (init_params/forward_params) for binding

The prompt should include:
- The intermediate cost JSON schema (embedded or referenced)
- Inference assumptions (KV cache available, batch_size, seq_len, cache_len, bytes)
- Example of a correct output (GPT2Attention or Conv1D from design doc)
- Instructions to write a memory/reasoning file and the final JSON

```
<role_definition>
You are a kernel cost analyzer. Your job is to analyze a composite kernel and write an intermediate cost JSON describing its children and bindings.

You receive a kernel name to analyze. Your task:
1. Read its source code (__init__ and forward())
2. Identify all children: module calls (self.xxx) and inline operations (tensor ops like +, *, torch.xxx)
3. For each child, determine its parameter names and write bindings mapping those params to expressions in the current kernel's namespace
4. Output an intermediate cost JSON

CRITICAL RULES:
- Composite kernels have ONLY a `children` object — NO `flops`, `memory_read`, `memory_write` fields
- Every cost-producing operation must be a child entry, including inline ops like `a + b`
- Each child has a unique label, a kernel reference, bindings, and optional count
- Use `config.xxx` for model config attributes (these are implicit, shared globally)
- Use `batch_size`, `seq_len`, `cache_len`, `bytes` as implicit runtime variables
- Do NOT use `self.xxx` in bindings — resolve self.xxx to config.xxx or constructor args

INFERENCE ASSUMPTIONS:
- KV cache is available (conditionals checking for cache → TRUE)
- batch_size for batch dimension
- seq_len for current sequence length (1 during decode, full during prefill)
- cache_len for KV cache sequence length
- Boolean params (has_bias): resolve to 1 or 0
- Inference mode (skip training-only branches, dropout = no-op)
</role_definition>

<task>
Kernel to analyze: {kernel_name}

Source directories:
- Transformers: {transformers_dir}
- PyTorch: {pytorch_dir}

Output directory for cost JSONs: {costs_dir}

Leaf kernels directory (hand-written, read-only): {kernels_dir}

Steps:
1. Locate {kernel_name}'s source code. Read __init__ and forward() methods.
2. Determine own parameters:
   - `init_params`: constructor args from __init__ (exclude self, config, **kwargs)
   - `forward_params`: input tensor dimensions in forward() not in init_params or implicit vars
3. For EACH child call in forward():
   a. Module calls (self.xxx(...)):
      - Look up self.xxx's class from __init__
      - Read the child class's __init__ and forward() to understand its parameters
      - Check if a leaf kernel JSON exists for this child in {kernels_dir}/ (using {name_map_path} for name resolution)
      - Write bindings: map child's init_params + forward_params to expressions using {kernel_name}'s variables
   b. Inline operations (a + b, x * scale, torch.matmul(a, b)):
      - Create synthetic child entry referencing the appropriate leaf kernel
      - Write bindings based on tensor shapes at the call site
4. For repeated children (loops), set count to the loop expression (e.g., "config.num_hidden_layers")
5. Write the intermediate cost JSON to {costs_dir}/{kernel_name}.json
6. Write your reasoning/analysis to {costs_dir}/{kernel_name}.md
</task>

<output_format>
Write the cost JSON to {costs_dir}/{kernel_name}.json

The file MUST follow this structure for a composite kernel:
{{
  "kernel_name": "{kernel_name}",
  "init_params": [...],
  "forward_params": [...],
  "children": {{
    "<label>": {{
      "kernel": "<child kernel name>",
      "bindings": {{ "<child_param>": "<expression in parent's namespace>" }},
      "count": 1
    }}
  }}
}}

Rules:
- "kernel" in each child entry must match a kernel_name from either:
  - A hand-written leaf JSON in {kernels_dir}/ (check {name_map_path} for name resolution)
  - Another composite kernel that will be analyzed separately
- Labels should be natural names: attribute names (c_attn, c_proj) or descriptive (qk_matmul, scale_query, add_residual)
- Bindings map from the CHILD's parameter names to expressions using the PARENT's variables
- count defaults to 1; use string expression for dynamic counts (e.g., "config.num_hidden_layers")
- Do NOT include `flops`, `memory_read`, or `memory_write` on composite kernels
- Expressions should use: init_params names, config.xxx, batch_size, seq_len, cache_len, bytes

Example output for Conv1D:
{{
  "kernel_name": "transformers.pytorch_utils.Conv1D",
  "init_params": ["nf", "nx"],
  "forward_params": [],
  "children": {{
    "addmm": {{
      "kernel": "torch.addmm",
      "bindings": {{ "M": "batch_size * seq_len", "K": "nx", "N": "nf" }},
      "count": 1
    }}
  }}
}}

Example output for a kernel with inline ops:
{{
  "kernel_name": "SomeAttention",
  "init_params": [],
  "forward_params": [],
  "children": {{
    "q_proj": {{
      "kernel": "F.linear",
      "bindings": {{ "M": "batch_size * seq_len", "K": "config.hidden_size", "N": "config.hidden_size", "has_bias": "1" }},
      "count": 1
    }},
    "scale_query": {{
      "kernel": "torch.mul",
      "bindings": {{ "num_elements": "batch_size * seq_len * config.hidden_size" }},
      "count": 1
    }},
    "qk_matmul": {{
      "kernel": "torch.matmul",
      "bindings": {{ "M": "batch_size * config.num_heads * seq_len", "K": "config.hidden_size // config.num_heads", "N": "cache_len" }},
      "count": 1
    }}
  }}
}}
</output_format>

<memory_file_guidance>
Write your reasoning to {costs_dir}/{kernel_name}.md as you work.

Include:
- Source file path and line numbers for __init__ and forward()
- Relevant code sections
- For each child: how you determined bindings (shape tracing)
- For inline ops: what operation, what tensors, what shapes
- Uncertainties or assumptions
</memory_file_guidance>
```

**Step 2: Commit**

```bash
git add prompts/cost_analyzer.txt
git commit -m "feat: rewrite cost analyzer prompt for Phase 1 children/bindings design"
```

---

### Task 5: Rewrite cost_analyzer.py for Phase 1

Replace the entire `cost_analyzer.py` with the new Phase 1 logic.

**Files:**
- Modify: `cost_analyzer.py` (full rewrite)

**Step 1: Write the new cost_analyzer.py**

The new script should:

1. **Parse args**: `--graph` (Stage 1 graph JSON), `--transformers`, `--pytorch`, `--output`
2. **Load the Stage 1 dependency graph** from the `--graph` file
3. **Load leaf kernel names**: scan `kernels/` directory + load `kernel_name_map.json` for name resolution
4. **Seed leaf kernels**: copy hand-written JSONs from `kernels/` to `analysis/costs/`
5. **BFS top-down** through the dependency graph:
   - Extract unique composite kernels and their BFS order (dedup by kernel_name)
   - For each composite kernel:
     - Skip if cost JSON already exists in `analysis/costs/`
     - Otherwise, build prompt from template and run Claude agent
     - Agent writes output to `analysis/costs/{kernel_name}.json`
6. **Validate** each output against `kernel_cost_schema.json`

Key differences from the old code:
- Top-down BFS (not bottom-up)
- No `atomic_kernels.json` loading, no `NAME_MAP` constant, no `prepare_sub_kernels`
- No `--single` flag
- Agent writes children/bindings (not sub_kernels with formulas)
- Prompt template uses different format variables

```python
#!/usr/bin/env python3
"""
Cost Analyzer - Stage 2 Phase 1 of LM-Predictor

Top-down BFS through kernel dependency graph. For each composite kernel,
runs a Claude agent to analyze source code and write an intermediate cost
JSON with children references and bindings.

Usage:
    uv run python cost_analyzer.py \
        --graph graphs/transformers.models.gpt2.modeling_gpt2.GPT2Attention.json \
        --transformers ./transformers \
        --pytorch ./pytorch \
        --output ./analysis
"""

import argparse
import asyncio
import json
import shutil
from collections import deque
from pathlib import Path
from typing import Final

import jsonschema
from claude_agent_sdk import query, ClaudeAgentOptions
from utils.message_printer import print_pretty_message


DISALLOWED_TOOLS: Final[list[str]] = [
    "Bash",
    "NotebookEdit",
    "WebFetch",
    "WebSearch",
    "BashOutput",
    "KillShell",
    "Skill",
    "SlashCommand",
    "AgentOutputTool",
]


def load_prompt_template(filename: str) -> str:
    """Load a prompt template from the prompts directory."""
    prompt_path = Path(__file__).parent / "prompts" / filename
    return prompt_path.read_text()


def load_leaf_kernel_names(kernels_dir: Path, name_map_path: Path) -> set[str]:
    """Load all known leaf kernel names (both short and fully-qualified)."""
    names: set[str] = set()

    # Short names from kernels/ directory (e.g., "torch.matmul", "F.linear")
    for json_file in kernels_dir.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)
        names.add(data["kernel_name"])

    # Fully-qualified names from name map
    if name_map_path.exists():
        with open(name_map_path) as f:
            name_map = json.load(f)
        names.update(name_map.keys())

    return names


def seed_leaf_kernels(kernels_dir: Path, costs_dir: Path, name_map_path: Path) -> None:
    """Copy hand-written leaf kernel JSONs to costs dir. Also create copies under mapped names."""
    # Copy originals
    for json_file in kernels_dir.glob("*.json"):
        dest = costs_dir / json_file.name
        if not dest.exists():
            shutil.copy2(json_file, dest)

    # Create copies under fully-qualified names
    if name_map_path.exists():
        with open(name_map_path) as f:
            name_map = json.load(f)
        for fq_name, short_name in name_map.items():
            src = costs_dir / f"{short_name}.json"
            dest = costs_dir / f"{fq_name}.json"
            if src.exists() and not dest.exists():
                shutil.copy2(src, dest)


def collect_composite_kernels_bfs(graph: dict, leaf_names: set[str]) -> list[str]:
    """BFS through graph, return composite kernel names in top-down order (deduped)."""
    order: list[str] = []
    seen: set[str] = set()
    queue: deque[dict] = deque([graph])

    while queue:
        node = queue.popleft()
        name = node.get("kernel_name", "")
        ktype = node.get("kernel_type", "")

        if ktype == "composite" and name not in seen and name not in leaf_names:
            seen.add(name)
            order.append(name)

        for child in node.get("sub_kernels", []):
            queue.append(child)

    return order


async def analyze_kernel(
    kernel_name: str,
    transformers_dir: str,
    pytorch_dir: str,
    costs_dir: Path,
    kernels_dir: Path,
    name_map_path: Path,
    schema: dict,
    working_dir: Path,
) -> bool:
    """Run the agent for one composite kernel. Returns True if successful."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {kernel_name}")
    print(f"{'='*60}")

    template = load_prompt_template("cost_analyzer.txt")
    prompt = template.format(
        kernel_name=kernel_name,
        transformers_dir=transformers_dir,
        pytorch_dir=pytorch_dir,
        costs_dir=str(costs_dir),
        kernels_dir=str(kernels_dir),
        name_map_path=str(name_map_path),
    )

    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            disallowed_tools=DISALLOWED_TOOLS,
            permission_mode="acceptEdits",
            cwd=str(working_dir),
            add_dirs=[transformers_dir, pytorch_dir],
        ),
    ):
        print_pretty_message(message)

    # Validate output
    output_file = costs_dir / f"{kernel_name}.json"
    if not output_file.exists():
        print(f"  WARNING: Agent did not produce output for {kernel_name}")
        return False

    with open(output_file) as f:
        output = json.load(f)

    try:
        jsonschema.validate(output, schema)
        print(f"  Schema validation PASSED for {kernel_name}")
        return True
    except jsonschema.ValidationError as e:
        print(f"  Schema validation FAILED for {kernel_name}: {e.message}")
        return False


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2 Phase 1: Top-down BFS cost analysis"
    )
    parser.add_argument(
        "-g", "--graph",
        required=True,
        help="Path to Stage 1 dependency graph JSON",
    )
    parser.add_argument(
        "--transformers",
        default="./transformers",
        help="Transformers source directory",
    )
    parser.add_argument(
        "--pytorch",
        default="./pytorch",
        help="PyTorch source directory",
    )
    parser.add_argument(
        "--output",
        default="./analysis",
        help="Output directory for cost analysis",
    )
    args = parser.parse_args()

    working_dir = Path(__file__).parent
    output_dir = Path(args.output)
    costs_dir = output_dir / "costs"
    costs_dir.mkdir(parents=True, exist_ok=True)

    kernels_dir = working_dir / "kernels"
    name_map_path = working_dir / "kernel_name_map.json"
    schema_path = working_dir / "kernel_cost_schema.json"

    # Load graph
    with open(args.graph) as f:
        graph = json.load(f)

    # Load schema
    with open(schema_path) as f:
        schema = json.load(f)

    # Load leaf kernel names
    leaf_names = load_leaf_kernel_names(kernels_dir, name_map_path)

    # Seed leaf kernels
    seed_leaf_kernels(kernels_dir, costs_dir, name_map_path)

    print(f"Graph: {args.graph}")
    print(f"Transformers dir: {args.transformers}")
    print(f"PyTorch dir: {args.pytorch}")
    print(f"Output dir: {output_dir}")
    print(f"Leaf kernels: {len(leaf_names)}")

    # BFS top-down
    composite_kernels = collect_composite_kernels_bfs(graph, leaf_names)
    print(f"Composite kernels to analyze: {len(composite_kernels)}")
    for i, name in enumerate(composite_kernels):
        print(f"  {i+1}. {name}")

    results: dict[str, bool] = {}
    for kernel_name in composite_kernels:
        # Skip if already computed
        if (costs_dir / f"{kernel_name}.json").exists():
            print(f"\nSkipping {kernel_name} (already exists)")
            results[kernel_name] = True
            continue

        success = await analyze_kernel(
            kernel_name=kernel_name,
            transformers_dir=args.transformers,
            pytorch_dir=args.pytorch,
            costs_dir=costs_dir,
            kernels_dir=kernels_dir,
            name_map_path=name_map_path,
            schema=schema,
            working_dir=working_dir,
        )
        results[kernel_name] = success

    # Summary
    print(f"\n{'='*60}")
    print("Phase 1 Summary")
    print(f"{'='*60}")
    for name, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  [{status}] {name}")
    print(f"\nCost JSONs saved to: {costs_dir}")


if __name__ == "__main__":
    asyncio.run(main())
```

**Step 2: Verify the script parses correctly**

Run: `uv run python -c "import cost_analyzer; print('OK')"`

Expected: "OK" (no import errors)

**Step 3: Commit**

```bash
git add cost_analyzer.py
git commit -m "feat: rewrite cost_analyzer.py for Phase 1 top-down BFS"
```

---

### Task 6: Test end-to-end with GPT2Attention graph

Run Phase 1 against the existing GPT2Attention dependency graph to validate the full pipeline.

**Files:**
- No new files created manually — agent produces `analysis/costs/*.json`

**Step 1: Run cost_analyzer against GPT2Attention**

Run:
```bash
uv run python cost_analyzer.py \
  --graph graphs/transformers.models.gpt2.modeling_gpt2.GPT2Attention.json \
  --transformers ./transformers \
  --pytorch ./pytorch \
  --output ./analysis
```

**Step 2: Check that leaf kernels were seeded**

Run: `ls analysis/costs/ | head -20`

Expected: Should see `torch.matmul.json`, `F.softmax.json`, etc.

**Step 3: Check that agent produced composite kernel outputs**

Run: `ls analysis/costs/*.json | grep -v "^analysis/costs/torch\.\|^analysis/costs/F\."`

Expected: Should see at least:
- `analysis/costs/transformers.models.gpt2.modeling_gpt2.GPT2Attention.json`
- `analysis/costs/transformers.pytorch_utils.Conv1D.json`

**Step 4: Validate output structure**

Run:
```bash
uv run python -c "
import json
for name in ['transformers.models.gpt2.modeling_gpt2.GPT2Attention', 'transformers.pytorch_utils.Conv1D']:
    with open(f'analysis/costs/{name}.json') as f:
        data = json.load(f)
    assert 'children' in data, f'{name}: missing children'
    assert 'flops' not in data, f'{name}: should not have flops'
    for label, child in data['children'].items():
        assert 'kernel' in child, f'{name}.{label}: missing kernel'
        assert 'bindings' in child, f'{name}.{label}: missing bindings'
    print(f'{name}: OK ({len(data[\"children\"])} children)')
"
```

Expected:
```
transformers.pytorch_utils.Conv1D: OK (1 children)
transformers.models.gpt2.modeling_gpt2.GPT2Attention: OK (N children)
```

**Step 5: Review agent output quality**

Manually inspect the generated JSONs:
- Do bindings make sense? (shapes should trace correctly through the code)
- Are inline ops captured? (scale, residual adds, etc.)
- Are kernel names correct? (matching leaf kernel names or other composite names)

**Step 6: Commit results**

```bash
git add analysis/
git commit -m "test: Phase 1 output for GPT2Attention"
```

---

## Notes

- **Phase 2 is NOT implemented here.** The intermediate cost JSONs are the deliverable. Phase 2 (bottom-up substitution + resolved cost tree) will be implemented after validating Phase 1 output quality.
- The `resolved_cost_schema.json` file is NOT created in this plan — it's a Phase 2 artifact.
- If agent output doesn't match expectations, iterate on `prompts/cost_analyzer.txt` before re-running.
