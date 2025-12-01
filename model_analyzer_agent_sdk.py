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


def load_prompt_template(filename: str) -> str:
    """Load a prompt template from the prompts directory."""
    prompt_path = Path(__file__).parent / "prompts" / filename
    return prompt_path.read_text()


async def module_analyze(module_name: str, working_dir: str, transformers_dir: str, pytorch_dir: str, module_analysis_dir: str, model_analysis_dir: str) -> None:
    # analysis_file = "model_analysis.json"
    # analysis_file_path = Path(f"{working_dir}/{analysis_file}")
    analysis_schema_file = "module_analysis_schema.json"
    # scratchpad_file_path = Path(f"{working_dir}/SCRATCHPAD.md")

    # if analysis_file_path.exists():
    #     analysis_file_path.unlink()
    # if scratchpad_file_path.exists():
    #     scratchpad_file_path.unlink()

    # Load prompt template and substitute variables
    prompt_template = load_prompt_template("model_analyzer.txt")
    prompt = prompt_template.format(
        module_name=module_name,
        module_analysis_dir=module_analysis_dir,
        working_dir=working_dir,
        # analysis_file=analysis_file,
        analysis_schema_file=analysis_schema_file,
        model_output_dir=model_analysis_dir,
    )
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

    asyncio.run(module_analyze(args.model, Path("."), args.transformers, args.pytorch, args.module_analysis, "models"))


if __name__ == "__main__":
    main()
