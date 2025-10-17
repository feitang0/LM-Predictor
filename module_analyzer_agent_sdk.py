import asyncio
import argparse
from typing import Final

from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage

SEPARATOR = "-" * 32
DEFAULT_WORKING_DIR = "transformers"
DISALLOWED_TOOLS: Final[list[str]] = [
    "Bash",
    "ExitPlanMode",
    "Edit",
    "Write",
    "NotebookEdit",
    "WebFetch",
    "WebSearch",
    "BashOutput",
    "KillShell",
    "Skill",
    "SlashCommand",
]


async def module_analyze(module_name: str, working_dir: str) -> None:
    async for message in query(
        prompt=f"Find all compute or memory access kernels of the {module_name}",
        options=ClaudeAgentOptions(
            disallowed_tools=DISALLOWED_TOOLS,
            cwd=working_dir,
            agents={},
        ),
    ):
        print(SEPARATOR)
        if isinstance(message, ResultMessage):
            print(message.result)
        else:
            print(message)
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--module", required=True)
    args = parser.parse_args()

    asyncio.run(module_analyze(args.module, DEFAULT_WORKING_DIR))


if __name__ == "__main__":
    main()
