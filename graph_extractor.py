#!/usr/bin/env python3
"""
Graph Extractor - Stage 1 of LM-Predictor

Builds a kernel dependency graph from a model root to atomic kernels.
Uses Python + Agent hybrid: Python orchestrates, Agent analyzes ONE kernel at a time.

Usage:
    uv run python graph_extractor.py \
        --module "transformers.models.llama.modeling_llama.LlamaForCausalLM" \
        --transformers ./transformers \
        --pytorch ./pytorch \
        --output ./graphs
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Final

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


def load_atomic_kernels(working_dir: Path) -> set[str]:
    """Load atomic kernel names from atomic_kernels.json."""
    atomic_file = working_dir / "atomic_kernels.json"
    with open(atomic_file) as f:
        data = json.load(f)

    # Flatten all kernel names from all categories
    kernels = set()
    for category_name, category in data.items():
        if isinstance(category, dict) and category_name not in ("$schema", "$comment"):
            kernels.update(category.keys())
    return kernels


def is_atomic_kernel(kernel_type: str, atomic_kernels: set[str]) -> bool:
    """Check if a kernel type is a known atomic kernel."""
    # Check exact match
    if kernel_type in atomic_kernels:
        return True
    # Check common torch.nn modules that are atomic
    atomic_nn_modules = {
        "torch.nn.Linear",
        "torch.nn.Embedding",
        "torch.nn.LayerNorm",
        "torch.nn.RMSNorm",
        "torch.nn.Dropout",
        "torch.nn.SiLU",
        "torch.nn.GELU",
        "torch.nn.ReLU",
        "torch.nn.Conv1d",
        "torch.nn.Conv2d",
        "torch.nn.Conv3d",
    }
    if kernel_type in atomic_nn_modules:
        return True
    return False


async def analyze_single_kernel(
    kernel_name: str,
    transformers_dir: str,
    pytorch_dir: str,
    output_dir: Path,
    working_dir: Path,
) -> dict:
    """Use Claude Agent to analyze ONE kernel's forward() method."""
    print(f"\n{'='*60}")
    print(f"Analyzing kernel: {kernel_name}")
    print(f"{'='*60}")

    # Build prompt
    template = load_prompt_template("kernel_analyzer.txt")
    kernels_dir = output_dir / "kernels"

    prompt = template.format(
        kernel_name=kernel_name,
        transformers_dir=transformers_dir,
        pytorch_dir=pytorch_dir,
        output_dir=str(kernels_dir),
    )

    # Run Claude Agent
    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            disallowed_tools=DISALLOWED_TOOLS,
            permission_mode="acceptEdits",
            cwd=str(working_dir),
            add_dirs=[transformers_dir, pytorch_dir],
        ),
    ):
        # Print agent messages with pretty formatting
        print_pretty_message(message)

    # Read the output file
    output_file = kernels_dir / f"{kernel_name}.json"
    if output_file.exists():
        with open(output_file) as f:
            return json.load(f)

    raise RuntimeError(f"Agent did not produce output file: {output_file}")


async def build_full_graph(
    root_kernel: str,
    transformers_dir: str,
    pytorch_dir: str,
    output_dir: Path,
    working_dir: Path,
) -> dict:
    """Build the complete dependency graph using BFS."""
    atomic_kernels = load_atomic_kernels(working_dir)
    analyzed_kernels: dict[str, dict] = {}

    # BFS queue
    queue = [root_kernel]

    while queue:
        kernel_name = queue.pop(0)

        # Skip if already analyzed
        if kernel_name in analyzed_kernels:
            continue

        # Skip if it's an atomic kernel
        if is_atomic_kernel(kernel_name, atomic_kernels):
            print(f"Skipping atomic kernel: {kernel_name}")
            continue

        # Agent analyzes this ONE kernel
        try:
            result = await analyze_single_kernel(
                kernel_name,
                transformers_dir,
                pytorch_dir,
                output_dir,
                working_dir,
            )
            analyzed_kernels[kernel_name] = result

            # Add composite children to queue
            for child in result.get("sub_kernels", []):
                if child.get("kernel_type") == "composite":
                    child_name = child.get("kernel_name")
                    if child_name and child_name not in analyzed_kernels:
                        queue.append(child_name)

        except Exception as e:
            print(f"Error analyzing {kernel_name}: {e}")
            continue

    return build_nested_tree(root_kernel, analyzed_kernels, atomic_kernels)


def build_nested_tree(
    root: str,
    flat_graph: dict[str, dict],
    atomic_kernels: set[str],
) -> dict:
    """Convert flat analysis results into nested tree structure."""
    if root not in flat_graph:
        # It's an atomic kernel
        return {
            "kernel_name": root,
            "kernel_type": "atomic",
        }

    result = flat_graph[root].copy()
    result["kernel_type"] = "composite"

    # Recursively build sub_kernels
    nested_children = []
    for child in result.get("sub_kernels", []):
        child_name = child.get("kernel_name", "")

        if child.get("kernel_type") == "composite" and child_name in flat_graph:
            # Recursively expand composite kernel
            nested_child = build_nested_tree(child_name, flat_graph, atomic_kernels)
            # Preserve call_site and multiplier from parent's reference
            if "call_site" in child:
                nested_child["call_site"] = child["call_site"]
            if "multiplier" in child:
                nested_child["multiplier"] = child["multiplier"]
            nested_children.append(nested_child)
        else:
            # Keep atomic kernel as-is
            nested_children.append(child)

    result["sub_kernels"] = nested_children
    return result


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract kernel dependency graph from a model"
    )
    parser.add_argument(
        "-m", "--module",
        required=True,
        help="Root module to analyze (e.g., transformers.models.llama.modeling_llama.LlamaForCausalLM)"
    )
    parser.add_argument(
        "--transformers",
        default="./transformers",
        help="Transformers source directory"
    )
    parser.add_argument(
        "--pytorch",
        default="./pytorch",
        help="PyTorch source directory"
    )
    parser.add_argument(
        "--output",
        default="./graphs",
        help="Output directory for graphs"
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Analyze only the specified kernel without recursing into dependencies"
    )
    args = parser.parse_args()

    working_dir = Path(__file__).parent
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "kernels").mkdir(exist_ok=True)

    print(f"Transformers dir: {args.transformers}")
    print(f"PyTorch dir: {args.pytorch}")
    print(f"Output dir: {output_dir}")

    if args.single:
        # Single kernel analysis mode
        print(f"Analyzing single kernel: {args.module}")
        result = await analyze_single_kernel(
            kernel_name=args.module,
            transformers_dir=args.transformers,
            pytorch_dir=args.pytorch,
            output_dir=output_dir,
            working_dir=working_dir,
        )

        output_file = output_dir / "kernels" / f"{args.module}.json"
        print(f"\n{'='*60}")
        print(f"Single kernel analysis saved to: {output_file}")
        print(f"Memory file: {output_dir / 'kernels' / f'{args.module}.md'}")
        print(f"{'='*60}")
    else:
        # Full graph building mode
        print(f"Building kernel dependency graph for: {args.module}")
        graph = await build_full_graph(
            root_kernel=args.module,
            transformers_dir=args.transformers,
            pytorch_dir=args.pytorch,
            output_dir=output_dir,
            working_dir=working_dir,
        )

        # Save final graph
        output_file = output_dir / f"{args.module}.json"
        with open(output_file, "w") as f:
            json.dump(graph, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Graph saved to: {output_file}")
        print(f"Individual kernel analyses saved to: {output_dir / 'kernels'}")
        print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
