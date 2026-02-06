#!/usr/bin/env python3
"""
Cost Analyzer - Phase 1 of LM-Predictor

Self-driven top-down BFS cost analysis. Starting from a root module, runs a
Claude agent for each kernel to discover children and write bindings. The
agent's output drives the BFS — each new child kernel reference that doesn't
already have a cost JSON is added to the queue.

Hand-written leaf kernel JSONs (in kernels/) are seeded into the costs
directory before BFS starts. Any kernel with an existing cost JSON is skipped.

Usage:
    uv run python cost_analyzer.py \
        --module transformers.models.gpt2.modeling_gpt2.GPT2Attention \
        --transformers ./transformers \
        --pytorch ./pytorch \
        --output ./analysis
"""

import argparse
import asyncio
import json
import shutil
from collections import deque
from pathlib import Path
from typing import Final

import jsonschema
from claude_agent_sdk import query, ClaudeAgentOptions
from utils.message_printer import print_pretty_message


DISALLOWED_TOOLS: Final[list[str]] = [
    "Bash",
    "NotebookEdit",
    "WebFetch",
    "WebSearch",
    "BashOutput",
    "KillShell",
    "Skill",
    "SlashCommand",
    "AgentOutputTool",
]


def load_prompt_template(filename: str) -> str:
    """Load a prompt template from the prompts directory."""
    prompt_path = Path(__file__).parent / "prompts" / filename
    return prompt_path.read_text()


def load_name_map(name_map_path: Path) -> dict[str, str]:
    """Load the kernel name mapping (fully-qualified -> short name)."""
    if name_map_path.is_file():
        with open(name_map_path) as f:
            return json.load(f)
    return {}


def seed_leaf_kernels(
    kernels_dir: Path, costs_dir: Path, name_map: dict[str, str]
) -> int:
    """Copy hand-written leaf kernel JSONs into the output costs directory.

    Also creates copies under fully-qualified names from the name map.
    Returns the number of files copied.
    """
    copied = 0
    costs_dir.mkdir(parents=True, exist_ok=True)

    if not kernels_dir.is_dir():
        return copied

    # Copy all kernel JSONs to costs dir
    for json_file in kernels_dir.glob("*.json"):
        dest = costs_dir / json_file.name
        shutil.copy2(json_file, dest)
        copied += 1

    # Build reverse map: short_name -> [fq_name, ...]
    reverse_map: dict[str, list[str]] = {}
    for fq_name, short_name in name_map.items():
        reverse_map.setdefault(short_name, []).append(fq_name)

    # For each leaf kernel, create copies under fully-qualified names
    for json_file in kernels_dir.glob("*.json"):
        short_name = json_file.stem
        for fq_name in reverse_map.get(short_name, []):
            fq_dest = costs_dir / f"{fq_name}.json"
            if not fq_dest.exists():
                shutil.copy2(json_file, fq_dest)
                copied += 1

    return copied


def has_cost_json(kernel_name: str, costs_dir: Path, name_map: dict[str, str]) -> bool:
    """Check if a kernel already has a cost JSON (under its own name or mapped name)."""
    if (costs_dir / f"{kernel_name}.json").exists():
        return True
    # Check mapped name
    resolved = name_map.get(kernel_name, kernel_name)
    if resolved != kernel_name and (costs_dir / f"{resolved}.json").exists():
        return True
    return False


def extract_child_kernels(cost_json_path: Path) -> list[str]:
    """Read a cost JSON and return the list of child kernel names."""
    with open(cost_json_path) as f:
        data = json.load(f)
    return [
        child["kernel"]
        for child in data.get("children", {}).values()
    ]


async def analyze_kernel(
    kernel_name: str,
    transformers_dir: str,
    pytorch_dir: str,
    costs_dir: Path,
    kernels_dir: Path,
    name_map_path: Path,
    schema: dict,
    working_dir: Path,
) -> bool:
    """Run the Claude agent to analyze one kernel.

    Returns True if the agent produced a valid cost JSON, False otherwise.
    """
    print(f"\n{'='*60}")
    print(f"Analyzing kernel: {kernel_name}")
    print(f"{'='*60}")

    template = load_prompt_template("cost_analyzer.txt")
    prompt = template.format(
        kernel_name=kernel_name,
        transformers_dir=transformers_dir,
        pytorch_dir=pytorch_dir,
        costs_dir=str(costs_dir),
        kernels_dir=str(kernels_dir),
        name_map_path=str(name_map_path),
    )

    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            disallowed_tools=DISALLOWED_TOOLS,
            permission_mode="acceptEdits",
            cwd=str(working_dir),
            add_dirs=[transformers_dir, pytorch_dir],
        ),
    ):
        print_pretty_message(message)

    # Validate agent output
    output_file = costs_dir / f"{kernel_name}.json"
    if not output_file.exists():
        print(f"  WARNING: Agent did not produce output for {kernel_name}")
        return False

    try:
        with open(output_file) as f:
            output = json.load(f)
        jsonschema.validate(output, schema)
        print(f"  OK: {kernel_name} validated successfully")
        return True
    except json.JSONDecodeError as e:
        print(f"  ERROR: Invalid JSON in {output_file}: {e}")
        return False
    except jsonschema.ValidationError as e:
        print(f"  ERROR: Schema validation failed for {kernel_name}: {e.message}")
        return False


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1: Top-down BFS cost analysis"
    )
    parser.add_argument(
        "-m", "--module",
        required=True,
        help="Root module to analyze (e.g., transformers.models.gpt2.modeling_gpt2.GPT2Attention)",
    )
    parser.add_argument(
        "--transformers",
        default="./transformers",
        help="Transformers source directory",
    )
    parser.add_argument(
        "--pytorch",
        default="./pytorch",
        help="PyTorch source directory",
    )
    parser.add_argument(
        "--output",
        default="./analysis",
        help="Output directory for cost analysis",
    )
    args = parser.parse_args()

    working_dir = Path(__file__).parent
    output_dir = Path(args.output)
    costs_dir = output_dir / "costs"
    kernels_dir = working_dir / "kernels"
    name_map_path = working_dir / "kernel_name_map.json"
    schema_path = working_dir / "kernel_cost_schema.json"

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    costs_dir.mkdir(exist_ok=True)

    # Load name map and schema
    name_map = load_name_map(name_map_path)
    with open(schema_path) as f:
        schema = json.load(f)

    # Seed leaf kernels
    seeded_count = seed_leaf_kernels(kernels_dir, costs_dir, name_map)

    print(f"Root module: {args.module}")
    print(f"Transformers dir: {args.transformers}")
    print(f"PyTorch dir: {args.pytorch}")
    print(f"Output dir: {output_dir}")
    print(f"Leaf kernels seeded: {seeded_count} files")

    # Self-driven BFS
    queue: deque[str] = deque([args.module])
    analyzed: set[str] = set()
    results: dict[str, bool] = {}

    while queue:
        kernel_name = queue.popleft()

        if kernel_name in analyzed:
            continue
        analyzed.add(kernel_name)

        # Skip if cost JSON already exists (leaf or previously analyzed)
        if has_cost_json(kernel_name, costs_dir, name_map):
            print(f"\nSkipping {kernel_name}: cost JSON already exists")
            continue

        # Run agent
        success = await analyze_kernel(
            kernel_name=kernel_name,
            transformers_dir=args.transformers,
            pytorch_dir=args.pytorch,
            costs_dir=costs_dir,
            kernels_dir=kernels_dir,
            name_map_path=name_map_path,
            schema=schema,
            working_dir=working_dir,
        )
        results[kernel_name] = success

        # If agent succeeded, discover new children and add to queue
        if success:
            cost_file = costs_dir / f"{kernel_name}.json"
            for child_kernel in extract_child_kernels(cost_file):
                if child_kernel not in analyzed:
                    queue.append(child_kernel)

    # Summary
    succeeded = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)

    print(f"\n{'='*60}")
    print(f"Phase 1 Analysis Complete")
    print(f"{'='*60}")
    print(f"  Kernels analyzed: {len(results)}")
    print(f"  Succeeded: {succeeded}")
    print(f"  Failed: {failed}")
    print(f"  Kernels skipped (already exist): {len(analyzed) - len(results)}")
    print(f"  Costs directory: {costs_dir}")

    if failed > 0:
        print(f"\nFailed kernels:")
        for name, success in results.items():
            if not success:
                print(f"  - {name}")

    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
