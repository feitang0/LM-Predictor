import asyncio
import argparse
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


async def model_arch_analyze(model_name: str, working_dir: str) -> None:
    async for message in query(
        prompt=f"Analyze the architecture of {model_name}, write the detail analysis in {model_name}_arch.json formt",
        options=ClaudeAgentOptions(
            disallowed_tools=DISALLOWED_TOOLS,
            cwd=working_dir,
            permission_mode="acceptEdits",
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
    parser.add_argument("-m", "--model", required=True)
    args = parser.parse_args()

    asyncio.run(model_arch_analyze(args.model, DEFAULT_WORKING_DIR))


if __name__ == "__main__":
    main()
