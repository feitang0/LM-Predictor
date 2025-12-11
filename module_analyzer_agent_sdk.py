import asyncio
import argparse
from pathlib import Path
from typing import Final
import inspect

from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage
import torch
import transformers

from utils import print_pretty_message
# DEFAULT_WORKING_DIR = "transformers"
DISALLOWED_TOOLS: Final[list[str]] = [
    # "Task",
    "Bash",
    # "Glob",
    # "Grep",
    # "Edit",  # Not needed - agent only writes new files, doesn't edit existing ones
    # "ExitPlanMode",
    # "Read",
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


def load_prompt_template(filename: str) -> str:
    """Load a prompt template from the prompts directory."""
    prompt_path = Path(__file__).parent / "prompts" / filename
    return prompt_path.read_text()


async def module_analyze(module_name: str, working_dir: str, transformers_dir: str, pytorch_dir: str, output_dir: str) -> None:
    # analysis_file = "module_analysis.json"
    # analysis_file_path = Path(f"{working_dir}/{analysis_file}")
    analysis_schema_file = "module_analysis_schema.json"
    # scratchpad_file_path = Path(f"{working_dir}/SCRATCHPAD.md")

    # if analysis_file_path.exists():
    #     analysis_file_path.unlink()
    # if scratchpad_file_path.exists():
    #     scratchpad_file_path.unlink()

    # Load prompt template and substitute variables
    prompt_template = load_prompt_template("module_analyzer.txt")
    prompt = prompt_template.format(
        module_name=module_name,
        transformers_dir=transformers_dir,
        pytorch_dir=pytorch_dir,
        working_dir=working_dir,
        output_dir=output_dir,
        # analysis_file=analysis_file,
        analysis_schema_file=analysis_schema_file,
    )
    print(prompt)
    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            disallowed_tools=DISALLOWED_TOOLS,
            permission_mode="acceptEdits",
            # model="haiku",
            # model="sonnet",
            cwd=working_dir,
            add_dirs=[transformers_dir, pytorch_dir, output_dir],
            # agents={},
        ),
    ):
        print_pretty_message(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--module", required=True)
    parser.add_argument("--transformers", required=True)
    parser.add_argument("--pytorch", required=True)
    args = parser.parse_args()

    asyncio.run(module_analyze(args.module, Path("."), args.transformers, args.pytorch, "modules"))


if __name__ == "__main__":
    main()
