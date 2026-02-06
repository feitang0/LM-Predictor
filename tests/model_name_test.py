import asyncio
import argparse
from pathlib import Path
from typing import Final
import inspect

from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage
import torch
import transformers

SEPARATOR = "=" * 64

schema = {
    "type": "object",
    "properties": {
        "temperature": {"type": "string"},
        "city": {"type": "string"}
    },
    "required": ["temperature", "city"]
}

async def model_name_test() -> None:
    prompt = "What's the weather today in Beijing? When querying weather, do not use WebSearch, use Bash tool"
    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            # allowed_tools = ['Bash'],
            permission_mode="bypassPermissions",
            # model="sonnet",
            # agents={},
            output_format={
                "type": "json_schema",
                "schema": schema
                }
        ),
    ):
        print(SEPARATOR)
        print(message)
        if hasattr(message, 'structured_output'):
            print("Structured Output:", message.structured_output)
        # print()


def main() -> None:
    asyncio.run(model_name_test())


if __name__ == "__main__":
    main()
