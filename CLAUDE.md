# LM-Predictor

FLOP and memory analysis system for PyTorch models using nested formula templates and Claude Code agent.

## Environment Setup

- Use `uv` for all dependency management (NOT pip/conda)
- Python 3.11+ required
- `uv sync` - Install all dependencies
- `uv add <package>` - Add new dependency
- `uv run <command>` - Run commands in managed environment

## Common Commands

- `uv run python model_analyzer.py --model_id <model>` - Extract model architecture
- `uv run python model_analyzer.py --model_id <model> --analyze --batch_size <B> --seq_len <S>` - Full FLOP/memory analysis
- `uv run python module_analyzer.py --module <ModuleName>` - Analyze unknown module (cache-first, agent fallback)
- `uv run python -m pytest` - Run tests (when implemented)

## Code Style

- Type hints required for all functions and class attributes
- Use dataclasses for structured data
- Follow PEP 8 naming conventions
- Prefer composition over inheritance
- Document complex algorithms with step-by-step comments

## Key Files

- `model_analyzer.py` - Main entry point for model FLOP/memory analysis
- `module_analyzer.py` - Cache-first module analysis with Claude Code agent fallback
- `module_db.json` - Formula template cache (starts empty, populated by agent)
- `generated_modules/` - Auto-generated Python modules from templates
- `DESIGN.md` - System architecture and design decisions
- `PROGRESS.md` - Implementation status and current milestones
- `SCRATCHPAD.md` - Active issues, debugging notes, and temporary work

## Architecture Overview

1. **Formula Templates**: JSON templates with `${param}` and `{Module}()` syntax (unified for FLOP/memory)
2. **Nested Evaluation**: Recursive resolution of module dependencies
3. **Library Namespacing**: `torch__nn__Linear` format prevents naming conflicts
4. **Agent Integration**: Claude Code analyzes unknown modules automatically
5. **Generated Modules**: Type-safe Python classes auto-generated from templates

## Important Notes

- NEVER store executable code in JSON - only formula templates
- Use registry system for module resolution: `compute_flops("torch.nn.Linear", ...)`, `compute_memory("torch.nn.Linear", ...)`
- Formula parameters use `${param}` syntax, module calls use `{Module}()` syntax (same for FLOP and memory templates)
- All separators (dots and underscores) become double underscores: `torch.nn.Linear` → `torch__nn__Linear`