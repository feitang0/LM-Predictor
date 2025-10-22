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
    analysis_file = "module_analysis.txt"
    analysis_file_path = Path(f"{working_dir}/{analysis_file}")

    module_parts = module_name.split(".")
    obj = eval(module_parts[0])
    for part in module_parts[1:]:
        obj = getattr(obj, part)
    module_forward_code_str = inspect.getsource(obj.forward)

    if analysis_file_path.exists():
        analysis_file_path.unlink()

    prompt = f"""# Task

Analyze the forward() method below, extract all compute or memory intensive kernels, and quantitatively analyze the FLOPs and Memory Access volumes of each kernel.

## Forward Method Source Code

```python
{module_forward_code_str}
```

## TODO: Extract compute or memory intensive kernels

### Format for Each Kernel

**Direct operations** (tensor ops, element-wise ops, etc.):
```
Line X-Y: <actual code snippet from source>
```
- **Operation**: <description>
- **FLOPs**: <formula>
- **Memory Access**: Read: <formula>, Write: <formula>

**Module calls** (calls to `self.xxx(...)`):
```
Line X-Y: <actual code snippet from source>
```
Only count FLOPs and Access in this module:
Examples:
a = m(b, c)
Only reference copy, thus FLOPs:0, Memory Access: 0
a = m(b, c) + d
FLOPs: <Only count the + part, do not count the FLOPs in internel module m>
Access: Read: <need read the m returned tensor and tensor d>, Write: <need write tensor a>

## Output

Write your analysis to `module_analysis.txt`.
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
