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


async def module_analyze(module_name: str, working_dir: str, transformers_dir: str, pytorch_dir: str) -> None:
    analysis_file = "module_analysis.json"
    analysis_file_path = Path(f"{working_dir}/{analysis_file}")
    analysis_schema_file = "module_analysis_schema.json"
    scratchpad_file_path = Path(f"{working_dir}/SCRATCHPAD.md")

    if analysis_file_path.exists():
        analysis_file_path.unlink()
    if scratchpad_file_path.exists():
        scratchpad_file_path.unlink()

    prompt = f"""# Task

Analyze the forward() method of {module_name}, extract all compute/memory intensive kernels, and quantitatively analyze the FLOPs and Memory Access volumes of each kernel.

## Available Resources

- Transformers library source: `{transformers_dir}`
- PyTorch library source: `{pytorch_dir}`

**IMPORTANT**: If you cannot find the source code for {module_name} in the available resources above, STOP immediately. Write the reason why you cannot find it in SCRATCHPAD.md and do NOT proceed with assumptions or hypothetical analysis.

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
- **Module calls** (e.g., `self.linear(x)`, `self.layer_norm(x)`): These will be analyzed independently in their own module analysis. Reference them using the format: `${{fully.qualified.ClassName}}({{parameters}})` where parameters are the standardized variables needed for that module's calculation.
- **Direct operations** (e.g., `x + y`, `torch.matmul(a, b)`): Calculate FLOPs and memory directly using formulas.

**How to use ${{fully.qualified.ClassName}}(parameters) format**:
- For module calls, use `${{fully.qualified.ClassName}}(param1, param2, ...)` to represent the FLOPs/Memory that will be calculated by that module
- `fully.qualified.ClassName` should be the complete Python import path (e.g., `torch.nn.modules.linear.Linear`, `transformers.models.llama.modeling_llama.LlamaRMSNorm`)
- You need to determine the actual class type of each module (e.g., `self.q_proj` might be a `torch.nn.modules.linear.Linear`)
- Include all relevant parameters needed for the module's calculation (e.g., dimensions, batch_size, seq_len)
- If a line has both module calls AND direct operations, sum them together:
  - Example: `${{fully.qualified.Class}}(params) + direct_operation_flops`
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
- Analysis: The q_proj module is a Linear layer and will be analyzed independently. Reference the module with its fully qualified class name and input parameters. The hidden_states has shape (batch_size, seq_len, hidden_size), and q_proj projects to (batch_size, seq_len, num_heads * head_dim).
- FLOPs: ${{torch.nn.modules.linear.Linear}}(batch_size, seq_len, hidden_size, num_heads, head_dim)
- Memory Access: ${{torch.nn.modules.linear.Linear}}(batch_size, seq_len, hidden_size, num_heads, head_dim)

Example 2 - Module call WITH additional operations:
```
Line 389: hidden_states = self.layer_norm(hidden_states) + residual
```
- Operation: Layer normalization followed by element-wise addition with residual
- Analysis: The layer_norm module (e.g., LlamaRMSNorm) will be analyzed independently. Here we count both the module reference and the element-wise addition. The output tensor has shape (batch_size, seq_len, hidden_size), requiring batch_size * seq_len * hidden_size additions for the residual connection.
- FLOPs: ${{transformers.models.llama.modeling_llama.LlamaRMSNorm}}(batch_size, seq_len, hidden_size) + batch_size * seq_len * hidden_size
- Memory Access:
  - Read: ${{transformers.models.llama.modeling_llama.LlamaRMSNorm}}(batch_size, seq_len, hidden_size) + 2 * batch_size * seq_len * hidden_size * a_bytes
  - Write: ${{transformers.models.llama.modeling_llama.LlamaRMSNorm}}(batch_size, seq_len, hidden_size) + batch_size * seq_len * hidden_size * a_bytes

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
  - self.input_proj(x): Linear layer → reference as ${{torch.nn.modules.linear.Linear}}(...)
  - self.activation(...): SiLU activation → reference as ${{torch.nn.modules.activation.SiLU}}(...)
  - self.dropout(...): Dropout layer → reference as ${{torch.nn.modules.dropout.Dropout}}(...)
  - Addition (+): direct operation → COUNT IT (batch_size * seq_len * intermediate_size)
  - self.output_proj(...): Linear layer → reference as ${{torch.nn.modules.linear.Linear}}(...)
  Both operands for addition have shape (batch_size, seq_len, intermediate_size).
- FLOPs: ${{torch.nn.modules.linear.Linear}}(batch_size, seq_len, hidden_size, intermediate_size) + ${{torch.nn.modules.activation.SiLU}}(batch_size, seq_len, intermediate_size) + ${{torch.nn.modules.dropout.Dropout}}(batch_size, seq_len, intermediate_size) + batch_size * seq_len * intermediate_size + ${{torch.nn.modules.linear.Linear}}(batch_size, seq_len, intermediate_size, hidden_size)
- Memory Access:
  - Read: ${{torch.nn.modules.linear.Linear}}(...) + ${{torch.nn.modules.activation.SiLU}}(...) + ${{torch.nn.modules.dropout.Dropout}}(...) + 2 * batch_size * seq_len * intermediate_size * a_bytes + ${{torch.nn.modules.linear.Linear}}(...)
  - Write: ${{torch.nn.modules.linear.Linear}}(...) + ${{torch.nn.modules.activation.SiLU}}(...) + ${{torch.nn.modules.dropout.Dropout}}(...) + batch_size * seq_len * intermediate_size * a_bytes + ${{torch.nn.modules.linear.Linear}}(...)

Example 5 - Chained operations with element-wise multiply:
```
Line 203: result = self.proj_out(self.norm(features) * self.gate(features))
```
- Operation: Element-wise multiplication between normalized features and gated features, then projection
- Analysis: In THIS module:
  - self.norm(features): RMSNorm layer → reference as ${{transformers.models.llama.modeling_llama.LlamaRMSNorm}}(...)
  - self.gate(features): Linear layer → reference as ${{torch.nn.modules.linear.Linear}}(...)
  - Multiplication (*): direct operation → COUNT IT
  - self.proj_out(...): Linear layer → reference as ${{torch.nn.modules.linear.Linear}}(...)
  Both operands for multiplication have shape (batch_size, seq_len, feature_dim).
- FLOPs: ${{transformers.models.llama.modeling_llama.LlamaRMSNorm}}(batch_size, seq_len, feature_dim) + ${{torch.nn.modules.linear.Linear}}(batch_size, seq_len, feature_dim, feature_dim) + batch_size * seq_len * feature_dim + ${{torch.nn.modules.linear.Linear}}(batch_size, seq_len, feature_dim, output_dim)
- Memory Access:
  - Read: ${{transformers.models.llama.modeling_llama.LlamaRMSNorm}}(...) + ${{torch.nn.modules.linear.Linear}}(...) + 2 * batch_size * seq_len * feature_dim * a_bytes + ${{torch.nn.modules.linear.Linear}}(...)
  - Write: ${{transformers.models.llama.modeling_llama.LlamaRMSNorm}}(...) + ${{torch.nn.modules.linear.Linear}}(...) + batch_size * seq_len * feature_dim * a_bytes + ${{torch.nn.modules.linear.Linear}}(...)

## Step-by-Step Analysis Process

Follow this TODO list systematically. Mark each item as you complete it.

### Phase 1: Scratchpad Analysis (SCRATCHPAD.md)

1. [ ] Find and read the forward() method source code for {module_name}
   - If you cannot find the source code in the available resources, STOP here
   - Write in SCRATCHPAD.md why you cannot find it (e.g., "Module not found in provided directories")
   - Do NOT proceed with assumptions or hypothetical analysis
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
            cwd=working_dir,
            add_dirs=[transformers_dir, pytorch_dir],
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
    parser.add_argument("--transformers", required=True)
    parser.add_argument("--pytorch", required=True)
    args = parser.parse_args()

    asyncio.run(module_analyze(args.module, Path("."), args.transformers, args.pytorch))


if __name__ == "__main__":
    main()
