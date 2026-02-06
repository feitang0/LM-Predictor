#!/usr/bin/env python3
"""
Cost Analyzer - Stage 2 Phase 1 of LM-Predictor

Top-down BFS cost analysis. For each composite kernel in the Stage 1
dependency graph, runs a Claude agent to produce an intermediate cost JSON
with structured children and bindings. Leaf kernels (hand-written in
kernels/) are seeded into the output costs directory; the agent only
analyzes composite kernels that don't yet have a cost JSON.

Usage:
    uv run python cost_analyzer.py \
        --graph graphs/transformers.models.gpt2.modeling_gpt2.GPT2Attention.json \
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


def load_leaf_kernel_names(kernels_dir: Path, name_map_path: Path) -> set[str]:
    """Load leaf kernel names from the kernels/ directory and name map.

    Returns a set containing:
    - Short names from kernel JSON filenames (e.g. "F.softmax", "torch.matmul")
    - Fully-qualified names that map to those short names (e.g. "torch.nn.functional.softmax")
    """
    names: set[str] = set()

    # Collect short names from kernel JSON filenames
    if kernels_dir.is_dir():
        for json_file in kernels_dir.glob("*.json"):
            short_name = json_file.stem  # e.g. "F.softmax"
            names.add(short_name)

    # Also add fully-qualified names from the name map
    if name_map_path.is_file():
        with open(name_map_path) as f:
            name_map: dict[str, str] = json.load(f)
        for fq_name, short_name in name_map.items():
            # Only add the fq_name if the short_name actually exists as a leaf kernel
            if short_name in names:
                names.add(fq_name)

    return names


def seed_leaf_kernels(kernels_dir: Path, costs_dir: Path, name_map_path: Path) -> int:
    """Copy hand-written leaf kernel JSONs into the output costs directory.

    Also creates copies under fully-qualified names from the name map.
    For example, if name_map says "torch.nn.functional.softmax" -> "F.softmax",
    then kernels/F.softmax.json is copied to both:
      - costs/F.softmax.json
      - costs/torch.nn.functional.softmax.json

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
    if name_map_path.is_file():
        with open(name_map_path) as f:
            name_map: dict[str, str] = json.load(f)

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


def collect_composite_kernels_bfs(graph: dict, leaf_names: set[str]) -> list[str]:
    """BFS through graph, return composite kernel names in top-down order (deduped)."""
    order: list[str] = []
    seen: set[str] = set()
    queue: deque[dict] = deque([graph])

    while queue:
        node = queue.popleft()
        name = node.get("kernel_name", "")
        ktype = node.get("kernel_type", "")

        if ktype == "composite" and name not in seen and name not in leaf_names:
            seen.add(name)
            order.append(name)

        for child in node.get("sub_kernels", []):
            queue.append(child)

    return order


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
    """Run the Claude agent to analyze one composite kernel.

    Returns True if the agent produced a valid cost JSON, False otherwise.
    """
    print(f"\n{'='*60}")
    print(f"Analyzing kernel: {kernel_name}")
    print(f"{'='*60}")

    # Build prompt from template
    template = load_prompt_template("cost_analyzer.txt")
    prompt = template.format(
        kernel_name=kernel_name,
        transformers_dir=transformers_dir,
        pytorch_dir=pytorch_dir,
        costs_dir=str(costs_dir),
        kernels_dir=str(kernels_dir),
        name_map_path=str(name_map_path),
    )

    # Run Claude agent
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

    # Check and validate agent output
    output_file = costs_dir / f"{kernel_name}.json"
    if not output_file.exists():
        print(f"  WARNING: Agent did not produce output for {kernel_name}")
        return False

    try:
        with open(output_file) as f:
            output = json.load(f)
        jsonschema.validate(output, schema)
        print(f"  OK: {kernel_name} cost JSON validated successfully")
        return True
    except json.JSONDecodeError as e:
        print(f"  ERROR: Invalid JSON in {output_file}: {e}")
        return False
    except jsonschema.ValidationError as e:
        print(f"  ERROR: Schema validation failed for {kernel_name}: {e.message}")
        return False


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Top-down BFS cost analysis (Stage 2 Phase 1)"
    )
    parser.add_argument(
        "-g", "--graph",
        required=True,
        help="Path to kernel dependency graph JSON (Stage 1 output)",
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

    # Load the Stage 1 dependency graph
    graph_path = Path(args.graph)
    with open(graph_path) as f:
        graph = json.load(f)

    # Load schema for validation
    with open(schema_path) as f:
        schema = json.load(f)

    # Step 1: Seed leaf kernels into costs directory
    leaf_names = load_leaf_kernel_names(kernels_dir, name_map_path)
    seeded_count = seed_leaf_kernels(kernels_dir, costs_dir, name_map_path)

    print(f"Graph: {args.graph}")
    print(f"Transformers dir: {args.transformers}")
    print(f"PyTorch dir: {args.pytorch}")
    print(f"Output dir: {output_dir}")
    print(f"Leaf kernels: {len(leaf_names)} names ({seeded_count} files seeded)")

    # Step 2: BFS to collect composite kernels in top-down order
    composite_kernels = collect_composite_kernels_bfs(graph, leaf_names)
    print(f"Composite kernels to analyze: {len(composite_kernels)}")
    for i, name in enumerate(composite_kernels, 1):
        print(f"  {i}. {name}")

    # Step 3: Analyze each composite kernel (skip if cost JSON already exists)
    results: dict[str, bool] = {}
    for kernel_name in composite_kernels:
        cost_file = costs_dir / f"{kernel_name}.json"
        if cost_file.exists():
            print(f"\nSkipping {kernel_name}: cost JSON already exists")
            results[kernel_name] = True
            continue

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

    # Step 4: Print summary
    succeeded = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)

    print(f"\n{'='*60}")
    print(f"Phase 1 Analysis Complete")
    print(f"{'='*60}")
    print(f"  Total composite kernels: {len(composite_kernels)}")
    print(f"  Succeeded: {succeeded}")
    print(f"  Failed: {failed}")
    print(f"  Costs directory: {costs_dir}")

    if failed > 0:
        print(f"\nFailed kernels:")
        for name, success in results.items():
            if not success:
                print(f"  - {name}")

    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
