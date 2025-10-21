import asyncio
import argparse
from pathlib import Path
from typing import Final

from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage

SEPARATOR = "-" * 32
DEFAULT_WORKING_DIR = "transformers"
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
    analysis_file = "module_analysis.txt"
    analysis_file_path = Path(f"{DEFAULT_WORKING_DIR}/{analysis_file}")

    if analysis_file_path.exists():
        analysis_file_path.unlink()

    async for message in query(
        prompt=f"""Analyze ONLY the forward() method of {module_name}.

**Step 1: Define Variables**
First, examine the forward() signature and common tensor shapes to define symbolic variables. Use readable names:
- batch_size
- seq_len
- hidden_size
- Any other relevant dimensions (num_heads, intermediate_size, head_dim, etc.)

**Step 2: Line-by-line Analysis in Execution Order**

Analyze each line in the exact order it executes in the forward() method. Do NOT create separate sections for different operation types. Output a single sequential list.

For each line, follow this format:

**Direct operations** (tensor ops, element-wise ops, etc.):
```
Line X: <actual code snippet from source>
```
- Operation: <description>
- FLOPs: <formula using {{variable}}>
- Memory Access: Read: <formula>, Write: <formula>

**Sub-module calls**:
```
Line X: <actual code snippet from source>
```
- Sub-module: ${{SubModuleClassName}}(input_params) -> output_shapes
- Memory Access: None (assignment only updates reference)

**CRITICAL: Always include the actual code snippet in a code block for EVERY analyzed line/operation**

Where:
- input_params: ONLY computational parameters with shapes (exclude control flags like use_cache)
- output_shapes: Shapes of all returned tensors (will be read when actually used in subsequent operations)

**Critical rules:**
1. MUST output in strict execution order - NO grouping, NO separate sections
2. Use {{variable}} notation for all formulas
3. For sub-modules: treat as black boxes, only specify input/output shapes
4. Memory Access for sub-modules: Assignment is just pointer update (no memory read). The actual memory read happens at the next operation that uses the tensor.
5. DO NOT read sub-module source code

Write your analysis to module_analysis.txt.""",
        options=ClaudeAgentOptions(
            disallowed_tools=DISALLOWED_TOOLS,
            permission_mode="acceptEdits",
            model="haiku",
            cwd=working_dir,
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

    asyncio.run(module_analyze(args.module, DEFAULT_WORKING_DIR))


if __name__ == "__main__":
    main()
