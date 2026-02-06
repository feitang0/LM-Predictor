import asyncio
import sys
from pathlib import Path
from typing import Final

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from claude_agent_sdk import query, ClaudeAgentOptions
from utils import print_pretty_message

DISALLOWED_TOOLS: Final[list[str]] = [
    # "Task",
    "Bash",
    # "Glob",
    # "Grep",
    # "ExitPlanMode",
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


async def main():
    options = ClaudeAgentOptions(
        disallowed_tools=DISALLOWED_TOOLS,
    )

    async for message in query(
        prompt="Which tools you can use?",
        options=options
    ):
        print_pretty_message(message)

asyncio.run(main())
