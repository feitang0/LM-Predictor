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


async def module_analyze(module_name: str, working_dir: str) -> None:
    analysis_file = "module_analysis.json"
    analysis_file_path = Path(f"{working_dir}/{analysis_file}")
    analysis_schema_file = "module_analysis_schema.json"
    scratchpad_file_path = Path(f"{working_dir}/SCRATCHPAD.md")

    module_parts = module_name.split(".")
    obj = eval(module_parts[0])
    for part in module_parts[1:]:
        obj = getattr(obj, part)
    module_forward_code_str = inspect.getsource(obj.forward)

    if analysis_file_path.exists():
        analysis_file_path.unlink()
    if scratchpad_file_path.exists():
        scratchpad_file_path.unlink()

    prompt = f"""# Task

Analyze the forward() method of {module_name}, extract all compute/memory intensive kernels, and quantitatively analyze the FLOPs and Memory Access volumes of each kernel.

## Forward Method Source Code

```python
{module_forward_code_str}
```

## Analysis Guidelines

### Inference Stage Configuration
Analyze under STANDARD INFERENCE conditions only:
- use_cache = True
- training = False
- No gradient computation
- Default configuration (e.g., pretraining_tp = 1, no tensor parallelism)
- Most common/default execution path for inference

**Important**: If the code has conditional branches (if/else), analyze ONLY the default inference path. Skip branches for:
- Training-specific code (e.g., gradient_checkpointing, training mode)
- Special configurations (e.g., pretraining_tp > 1, tensor parallelism)
- Non-default options

### Unified name of parameters

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

**FLOPs Counting**:
- Multiply-accumulate (MAC) = 2 FLOPs (1 multiply + 1 add)
- Matrix multiplication (M, K) @ (K, N) = 2 * M * K * N FLOPs
- Element-wise operations = 1 FLOP per element

### Kernel Analysis in the Execution Order

**CRITICAL**: Go through the forward() method line by line from the source code. Analyze EVERY line that performs computation. Use the ACTUAL line numbers from the provided source code.

**How to analyze**:
1. Start from the first line of the forward() method
2. Go through each line sequentially
3. For each line with computation/operations, create an analysis entry
4. Use the actual line number from the source code
5. Copy the exact code from that line

**What to analyze**:
- Variable assignments with computations
- Function/method calls (both module calls and direct operations)
- Tensor operations (element-wise, matmul, reshape, etc.)
- Control flow (if/else) - but only the default inference path

**What to skip**:
- Empty lines, comments
- Pure metadata (e.g., `bsz, seq_len = x.size()` - just unpacking shape info)
- Non-default configuration branches (e.g., `if pretraining_tp > 1`)

**Module Calls vs Direct Operations**:
- **Module calls** (e.g., `self.linear(x)`, `self.layer_norm(x)`): These will be analyzed independently in their own module analysis. Reference them using the format: `${{module_name}}({{parameters}})` where parameters are the standardized variables needed for that module's calculation.
- **Direct operations** (e.g., `x + y`, `torch.matmul(a, b)`): Calculate FLOPs and memory directly using formulas.

**How to use ${{module_name}}(parameters) format**:
- For module calls, use `${{module_name}}(param1, param2, ...)` to represent the FLOPs/Memory that will be calculated by that module
- `module_name` should match the attribute name (e.g., `q_proj`, `layer_norm`, `gate_proj`)
- Include all relevant parameters needed for the module's calculation (e.g., dimensions, batch_size, seq_len)
- If a line has both module calls AND direct operations, sum them together:
  - Example: `${{module}}(params) + direct_operation_flops`
- This allows tracking which modules contribute to each line's total cost

**Format for each analyzed line**:
```
Line X: <exact code from source line X>
```
- Operation: <brief description>
- Analysis: <detailed explanation of FLOPs/memory calculation, tensor shapes>
- FLOPs: <formula using standardized variables>
- Memory Access: Read: <formula in bytes>, Write: <formula in bytes>

**Important**:
- Use ACTUAL line numbers from the source (not "Line 1, Line 2, Line 3...")
- ALWAYS copy the exact code snippet
- Analyze each line separately - NEVER merge similar lines like q/k/v projections
- For complex single-line expressions with multiple operations, analyze the entire line as one kernel

**Examples:**

Example 1 - Module call without additional operations:
```
Line 342: query_states = self.q_proj(hidden_states)
```
- Operation: Query projection through linear layer
- Analysis: The q_proj module will be analyzed independently. Reference the module with its input parameters. The hidden_states has shape (batch_size, seq_len, hidden_size), and q_proj projects to (batch_size, seq_len, num_heads * head_dim).
- FLOPs: ${{q_proj}}(batch_size, seq_len, hidden_size, num_heads, head_dim)
- Memory Access: ${{q_proj}}(batch_size, seq_len, hidden_size, num_heads, head_dim)

Example 2 - Module call WITH additional operations:
```
Line 389: hidden_states = self.layer_norm(hidden_states) + residual
```
- Operation: Layer normalization followed by element-wise addition with residual
- Analysis: The layer_norm module will be analyzed independently. Here we count both the module reference and the element-wise addition. The output tensor has shape (batch_size, seq_len, hidden_size), requiring batch_size * seq_len * hidden_size additions for the residual connection.
- FLOPs: ${{layer_norm}}(batch_size, seq_len, hidden_size) + batch_size * seq_len * hidden_size
- Memory Access:
  - Read: ${{layer_norm}}(batch_size, seq_len, hidden_size) + 2 * batch_size * seq_len * hidden_size * a_bytes
  - Write: ${{layer_norm}}(batch_size, seq_len, hidden_size) + batch_size * seq_len * hidden_size * a_bytes

Example 3 - Direct tensor operation:
```
Line 275: attn_weights = attn_weights / math.sqrt(self.head_dim)
```
- Operation: Element-wise division by scalar (scaling)
- Analysis: The tensor attn_weights has shape (batch_size, num_heads, seq_len, cache_len). Element-wise division requires one division per element.
- FLOPs: batch_size * num_heads * seq_len * cache_len
- Memory Access: Read: batch_size * num_heads * seq_len * cache_len * a_bytes, Write: batch_size * num_heads * seq_len * cache_len * a_bytes

Example 4 - Complex expression with multiple module calls and operations:
```
Line 156: output = self.output_proj(self.dropout(self.activation(self.input_proj(x))) + skip_connection)
```
- Operation: Element-wise addition between dropout output and skip connection, then output projection
- Analysis: Break down what happens in THIS module:
  - self.input_proj(x): module call → reference as ${{input_proj}}(...)
  - self.activation(...): module call → reference as ${{activation}}(...)
  - self.dropout(...): module call → reference as ${{dropout}}(...)
  - Addition (+): direct operation → COUNT IT (batch_size * seq_len * hidden_size)
  - self.output_proj(...): module call → reference as ${{output_proj}}(...)
  Both operands for addition have shape (batch_size, seq_len, hidden_size).
- FLOPs: ${{input_proj}}(batch_size, seq_len, hidden_size, intermediate_size) + ${{activation}}(batch_size, seq_len, intermediate_size) + ${{dropout}}(batch_size, seq_len, intermediate_size) + batch_size * seq_len * intermediate_size + ${{output_proj}}(batch_size, seq_len, intermediate_size, hidden_size)
- Memory Access:
  - Read: ${{input_proj}}(...) + ${{activation}}(...) + ${{dropout}}(...) + 2 * batch_size * seq_len * intermediate_size * a_bytes + ${{output_proj}}(...)
  - Write: ${{input_proj}}(...) + ${{activation}}(...) + ${{dropout}}(...) + batch_size * seq_len * intermediate_size * a_bytes + ${{output_proj}}(...)

Example 5 - Chained operations with element-wise multiply:
```
Line 203: result = self.proj_out(self.norm(features) * self.gate(features))
```
- Operation: Element-wise multiplication between normalized features and gated features, then projection
- Analysis: In THIS module:
  - self.norm(features): module call → reference as ${{norm}}(...)
  - self.gate(features): module call → reference as ${{gate}}(...)
  - Multiplication (*): direct operation → COUNT IT
  - self.proj_out(...): module call → reference as ${{proj_out}}(...)
  Both operands for multiplication have shape (batch_size, seq_len, feature_dim).
- FLOPs: ${{norm}}(batch_size, seq_len, feature_dim) + ${{gate}}(batch_size, seq_len, feature_dim) + batch_size * seq_len * feature_dim + ${{proj_out}}(batch_size, seq_len, feature_dim, output_dim)
- Memory Access:
  - Read: ${{norm}}(...) + ${{gate}}(...) + 2 * batch_size * seq_len * feature_dim * a_bytes + ${{proj_out}}(...)
  - Write: ${{norm}}(...) + ${{gate}}(...) + batch_size * seq_len * feature_dim * a_bytes + ${{proj_out}}(...)

## Step-by-Step Analysis Process

Follow this TODO list systematically. Mark each item as you complete it.

### Phase 1: Scratchpad Analysis (SCRATCHPAD.md)

1. [ ] Read and understand the forward() method source code
2. [ ] Identify the default inference path (skip training/special config branches)
3. [ ] List all variables and their dimensions/shapes used in the method
4. [ ] Go through the code line by line, noting actual line numbers from the source
5. [ ] For each computational line, create an analysis entry with:
   - Line number and exact code snippet
   - Operation description
   - Detailed analysis with tensor shapes
   - FLOPs formula using standardized variables
   - Memory access formula (Read/Write in bytes)
6. [ ] Review the scratchpad to ensure no computational lines were missed

### Phase 2: Structured Output ({analysis_file})

7. [ ] Read the schema file at {working_dir}/{analysis_schema_file}
8. [ ] Reorganize your scratchpad analysis into {analysis_file} following the schema exactly
9. [ ] Verify all formulas use standardized variable names (batch_size, seq_len, etc.)
10. [ ] Double-check memory access is in BYTES (multiply by w_bytes/a_bytes)
11. [ ] Validate the JSON structure matches the schema

## Output Requirements

### SCRATCHPAD.md (Phase 1)
- Informal, working document for your analysis
- Show your thinking process
- Include all line-by-line analysis in execution order
- Use the format specified in "Kernel Analysis in the Execution Order" section
- Can be verbose and include intermediate calculations

### {analysis_file} (Phase 2)
- Formal JSON output following {working_dir}/{analysis_schema_file}
- Clean, structured, and concise
- All formulas use standardized variable names
- All memory access in bytes
- Valid JSON format
"""
    print(prompt)
    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            disallowed_tools=DISALLOWED_TOOLS,
            permission_mode="acceptEdits",
            # model="haiku",
            model="sonnet",
            # cwd=working_dir,
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
    parser.add_argument("-m", "--module", required=True)
    args = parser.parse_args()

    asyncio.run(module_analyze(args.module, Path(".")))


if __name__ == "__main__":
    main()
