# LM-Predictor
LM-Predictor is a Deep Learning Model FLOPs/Memory cost analysis system. It first builds a kernel dependency graph from top-to-bottom, and then calculates FLOPs/Memory using a bottom-to-top method.

## Critical Notes
- Never directly write code. Instead, propose several available methods and think carefully before making any modifications.

## Architecture Overview

**Stage 1: Kernel Dependency Graph Extraction**
- Build a structural dependency graph from model root to atomic kernels
- Output: Nested JSON tree showing what calls what

**Stage 2: Bottom-Up FLOPs/Memory Analysis**
- Atomic kernels have hardcoded formulas (human-maintained)
- Agent resolves parameters from child to parent context
- Aggregate costs bottom-up from atomic kernels to root

## Key Files
- `docs/DESIGN.md`: Detailed design documentation
- `atomic_kernels.json`: Catalog of atomic kernels with hardcoded FLOPs/Memory formulas
- `kernel_dependency_schema.json`: Schema for dependency graph output
- `model_analyzer.py`: Model architecture inspection tool
- `graph_extractor.py`: Stage 1 kernel dependency graph extractor
- `prompts/`: Prompt templates for graph extraction and cost analysis
- `utils/`: Utility modules for message printing and common functions
- `graphs/`: Kernel dependency graphs output directory
- `analysis/`: FLOPs/memory analysis output directory

## Common Workflows

1. **Show Model Architecture**: Use `model_analyzer.py` with a HuggingFace model ID to inspect model structure
2. **Extract Kernel Dependency Graph**: Build dependency graph from model to atomic kernels (Stage 1)
3. **Compute FLOPs/Memory**: Bottom-up analysis using dependency graph and atomic kernel formulas (Stage 2)

## External Dependencies

- **transformers/** submodule: HuggingFace Transformers source code for model introspection
- **pytorch/** submodule: PyTorch source code for understanding operations
- Use `uv` for Python environment management
