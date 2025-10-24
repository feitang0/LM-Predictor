import asyncio
import argparse
from pathlib import Path
from typing import Final
import inspect

from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage
import torch
import transformers

SEPARATOR = "-" * 32
# DEFAULT_WORKING_DIR = "transformers"
DISALLOWED_TOOLS: Final[list[str]] = [
    # "Task",
    "Bash",
    # "Glob",
    # "Grep",
    "ExitPlanMode",
    # "Read",
    # "Edit",
    # "Write",
    "NotebookEdit",
    "WebFetch",
    # "TodoWrite",
    "WebSearch",
    "BashOutput",
    "KillShell",
    "Skill",
    "SlashCommand",
]


async def module_analyze(module_name: str, working_dir: str, transformers_dir: str, pytorch_dir: str, module_analysis_dir: str) -> None:
    analysis_file = "model_analysis.json"
    analysis_file_path = Path(f"{working_dir}/{analysis_file}")
    analysis_schema_file = "module_analysis_schema.json"
    scratchpad_file_path = Path(f"{working_dir}/SCRATCHPAD.md")

    if analysis_file_path.exists():
        analysis_file_path.unlink()
    if scratchpad_file_path.exists():
        scratchpad_file_path.unlink()

    prompt = f"""# Task

Expand all module call references in the analysis of {module_name} to produce a fully expanded analysis with ONLY direct operations.

**What You Need to Do**:
1. Read the module analysis JSON for {module_name} from `{module_analysis_dir}`
2. Find all kernels that contain `${{ClassName}}(...)` references in their FLOPs or Memory Access formulas
3. For each `${{ClassName}}(...)` reference:
   - Look up the corresponding analysis file in `{module_analysis_dir}` (e.g., `${{torch.nn.modules.linear.Linear}}` → `torch.nn.modules.linear.Linear.json`)
   - Replace the reference with the actual kernels from that module's analysis
   - If the referenced module analysis is NOT found, STOP and document which module is missing in SCRATCHPAD.md
4. Continue recursively until ALL `${{...}}` references are expanded into direct operations
5. Output the final expanded analysis where every kernel is a direct operation (no `${{...}}` references)

## Available Resources

- Module analysis results: `{module_analysis_dir}` (JSON files containing analysis of all individual modules)

**STOP Conditions**:
1. If you cannot find the analysis file for {module_name} in `{module_analysis_dir}`, STOP and write the reason in SCRATCHPAD.md
2. If you encounter a `${{ClassName}}(...)` reference and cannot find that module's analysis in `{module_analysis_dir}`, STOP and document which module is missing in SCRATCHPAD.md

## Expansion Guidelines

### How Module Reference Expansion Works

When you see a formula like:
```
"flops": "${torch.nn.modules.linear.Linear}(batch_size, seq_len, hidden_size, intermediate_size) + batch_size * seq_len * intermediate_size"
```

You need to:
1. Identify the reference: `${torch.nn.modules.linear.Linear}(batch_size, seq_len, hidden_size, intermediate_size)`
2. Find the file: `torch.nn.modules.linear.Linear.json` in `{module_analysis_dir}`
3. Read that module's kernels and extract their FLOPs/Memory formulas
4. Substitute the parameters from the reference into those formulas
5. Replace the `${...}` reference with the expanded formulas
6. If the expanded formulas also contain `${...}` references, continue recursively

### Parameter Substitution

When expanding a reference like `${{ClassName}}(param1, param2, ...)`, you need to:
- Map the parameters to the module's required parameters
- Substitute them into the module's formulas
- Preserve the parameter names in the final expanded formula

### Unified Parameter Names

- batch_size: batch size
- seq_len: sequence length
- cache_len: KV cache length
- w_bytes: weights precision in bytes
- a_bytes: activations precision in bytes
You can add the variables as needed, but USE these variables to express the formula EVEN IF the source code use other names

**Notation**:
- `num_elements(tensor)` means the total number of elements in the tensor
- Memory access must be expressed in BYTES: multiply num_elements by w_bytes (for weights) or a_bytes (for activations)
- Example: tensor of shape (batch_size, seq_len, hidden_size) has `batch_size * seq_len * hidden_size` elements
  - Memory in bytes: `batch_size * seq_len * hidden_size * a_bytes`

**Examples:**

Example 1 - Expanding a simple module reference into nested structure:

Input kernel from {module_name} analysis:
```json
{{
  "kernel_type": "direct_operation",
  "operation": "Query projection",
  "analysis": "Linear projection for query vectors",
  "flops": "${{torch.nn.modules.linear.Linear}}(batch_size, seq_len, hidden_size, num_heads * head_dim)",
  "memory_access": {{
    "read": "${{torch.nn.modules.linear.Linear}}(batch_size, seq_len, hidden_size, num_heads * head_dim)",
    "write": "${{torch.nn.modules.linear.Linear}}(batch_size, seq_len, hidden_size, num_heads * head_dim)"
  }}
}}
```

Step 1: Detect `${{torch.nn.modules.linear.Linear}}(...)` reference
Step 2: Read `torch.nn.modules.linear.Linear.json` from `{module_analysis_dir}`:
```json
{{
  "kernels": [
    {{
      "kernel_type": "direct_operation",
      "operation": "Matrix multiplication",
      "analysis": "Core matmul operation",
      "flops": "2 * batch_size * seq_len * in_features * out_features",
      "memory_access": {{
        "read": "batch_size * seq_len * in_features * a_bytes + in_features * out_features * w_bytes",
        "write": "batch_size * seq_len * out_features * a_bytes"
      }}
    }}
  ]
}}
```

Step 3: Create nested structure with parameter substitution (in_features→hidden_size, out_features→num_heads*head_dim):
```json
{{
  "kernel_type": "composite_operation",
  "operation": "Query projection",
  "analysis": "Linear projection for query vectors",
  "sub_kernels": [
    {{
      "kernel_type": "direct_operation",
      "operation": "Matrix multiplication",
      "analysis": "Core matmul operation",
      "flops": "2 * batch_size * seq_len * hidden_size * num_heads * head_dim",
      "memory_access": {{
        "read": "batch_size * seq_len * hidden_size * a_bytes + hidden_size * num_heads * head_dim * w_bytes",
        "write": "batch_size * seq_len * num_heads * head_dim * a_bytes"
      }}
    }}
  ]
}}
```

Example 2 - Expanding with additional direct operations:

Input kernel from {module_name} analysis:
```json
{{
  "kernel_type": "direct_operation",
  "operation": "Layer norm with residual addition",
  "analysis": "Applies RMS normalization then adds residual",
  "flops": "${{transformers.models.llama.modeling_llama.LlamaRMSNorm}}(batch_size, seq_len, hidden_size) + batch_size * seq_len * hidden_size",
  "memory_access": {{
    "read": "${{transformers.models.llama.modeling_llama.LlamaRMSNorm}}(batch_size, seq_len, hidden_size) + 2 * batch_size * seq_len * hidden_size * a_bytes",
    "write": "${{transformers.models.llama.modeling_llama.LlamaRMSNorm}}(batch_size, seq_len, hidden_size) + batch_size * seq_len * hidden_size * a_bytes"
  }}
}}
```

After expansion (LlamaRMSNorm has 2 kernels: variance calculation + normalization):
```json
{{
  "kernel_type": "composite_operation",
  "operation": "Layer norm with residual addition",
  "analysis": "Applies RMS normalization then adds residual",
  "sub_kernels": [
    {{
      "kernel_type": "direct_operation",
      "operation": "Variance calculation (from RMSNorm)",
      "analysis": "Compute mean square",
      "flops": "2 * batch_size * seq_len * hidden_size",
      "memory_access": {{
        "read": "batch_size * seq_len * hidden_size * a_bytes",
        "write": "batch_size * seq_len * a_bytes"
      }}
    }},
    {{
      "kernel_type": "direct_operation",
      "operation": "Normalization (from RMSNorm)",
      "analysis": "Scale by inverse sqrt of variance",
      "flops": "3 * batch_size * seq_len * hidden_size",
      "memory_access": {{
        "read": "batch_size * seq_len * hidden_size * a_bytes + batch_size * seq_len * a_bytes + hidden_size * w_bytes",
        "write": "batch_size * seq_len * hidden_size * a_bytes"
      }}
    }},
    {{
      "kernel_type": "direct_operation",
      "operation": "Residual addition",
      "analysis": "Element-wise addition with residual connection",
      "flops": "batch_size * seq_len * hidden_size",
      "memory_access": {{
        "read": "2 * batch_size * seq_len * hidden_size * a_bytes",
        "write": "batch_size * seq_len * hidden_size * a_bytes"
      }}
    }}
  ]
}}
```

## Nested Structure for Expansion

When expanding a kernel that contains `${{ClassName}}(...)` references, create a NESTED structure:

```json
{{
  "kernel_type": "composite_operation",
  "operation": "Original operation description",
  "analysis": "High-level explanation",
  "sub_kernels": [
    {{
      "kernel_type": "direct_operation",
      "operation": "Sub-operation from expanded module",
      "analysis": "Detailed explanation",
      "flops": "explicit formula",
      "memory_access": {{
        "read": "explicit formula",
        "write": "explicit formula"
      }}
    }}
  ]
}}
```

**Key Points**:
- Parent kernel becomes `kernel_type: "composite_operation"` with `sub_kernels` array
- Each sub-kernel from the expanded module becomes an entry in `sub_kernels`
- If a sub-kernel also has `${{...}}` references, expand it recursively (nested sub_kernels)
- Only leaf kernels (those with no sub_kernels) have `flops` and `memory_access` fields
- Direct operations that don't need expansion remain as `kernel_type: "direct_operation"`

## Step-by-Step Expansion Process

Follow this TODO list systematically. Mark each item as you complete it.

### Phase 1: Expansion Planning (SCRATCHPAD.md)

1. [ ] Read the module analysis JSON for {module_name} from `{module_analysis_dir}`
   - If the analysis file is not found, STOP and document which file is missing in SCRATCHPAD.md
2. [ ] Identify all kernels that contain `${{ClassName}}(...)` references in their formulas
3. [ ] For each `${{ClassName}}(...)` reference:
   - List the class name and parameters
   - Check if the corresponding analysis JSON exists in `{module_analysis_dir}`
   - If any module analysis is missing, STOP and document in SCRATCHPAD.md
4. [ ] Plan the expansion order (handle nested references by expanding recursively)
5. [ ] Document your expansion strategy in SCRATCHPAD.md

### Phase 2: Perform Expansion ({analysis_file})

6. [ ] Read the schema file at {working_dir}/{analysis_schema_file}
7. [ ] For each kernel with `${{...}}` references, expand them into nested structure:
   - Read the referenced module's analysis JSON
   - Create a `sub_kernels` array
   - For each kernel from the referenced module, add it to `sub_kernels`
   - Substitute parameters appropriately in all formulas
   - If sub-kernels also have `${{...}}` references, expand them recursively
8. [ ] Continue recursively until all leaf kernels are direct operations
9. [ ] Verify the final output: all leaf kernels have NO `${{...}}` references
10. [ ] Validate the JSON structure matches the schema
11. [ ] Write the fully expanded nested analysis to {analysis_file}

## Output Requirements

### SCRATCHPAD.md (Phase 1)
- Document which module analysis file you're expanding
- List all `${{ClassName}}(...)` references found in the original analysis
- Document which module analysis files you need to read for expansion
- Show your expansion strategy and plan (including nesting depth)
- If any required module analysis is missing, document which one and STOP

### {analysis_file} (Phase 2)
- Fully expanded NESTED analysis following {working_dir}/{analysis_schema_file}
- Use `kernel_type: "composite_operation"` with `sub_kernels` for expanded modules
- Use `kernel_type: "direct_operation"` for leaf operations
- NO `${{...}}` references in leaf kernels - all must have explicit formulas
- All formulas use standardized variable names (batch_size, seq_len, etc.)
- All memory access in bytes (with w_bytes/a_bytes)
- Preserve the hierarchical structure showing which operations came from which modules
- Valid JSON format matching the schema
"""
    print(prompt)
    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            disallowed_tools=DISALLOWED_TOOLS,
            permission_mode="acceptEdits",
            # model="haiku",
            model="sonnet",
            cwd=working_dir,
            add_dirs=[transformers_dir, pytorch_dir, module_analysis_dir],
            agents={},
        ),
    ):
        # print(SEPARATOR)
        if isinstance(message, ResultMessage):
            print(message.result)
        else:
            print(message)
            # pass
        # print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("--transformers", required=True)
    parser.add_argument("--pytorch", required=True)
    parser.add_argument("--module-analysis", required=True)
    args = parser.parse_args()

    asyncio.run(module_analyze(args.model, Path("."), args.transformers, args.pytorch, args.module_analysis))


if __name__ == "__main__":
    main()
