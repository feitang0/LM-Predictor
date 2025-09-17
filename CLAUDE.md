# LM-Predictor

FLOP and memory analysis system for PyTorch models using nested formula templates and Claude Code agent.

## Environment Setup

- Use `uv` for all dependency management (NOT pip/conda)
- Python 3.11+ required
- `cp .env.example .env` - Copy environment template and add your Hugging Face token
- `uv sync` - Install all dependencies
- `uv add <package>` - Add new dependency
- `uv run <command>` - Run commands in managed environment

## Common Commands

- `uv run python module_analyzer.py <ModuleName>` - Analyze module (cache-first, agent fallback)
- `uv run python module_analyzer_agent.py <ModuleName>` - Direct agent analysis of unknown module
- `uv run python model_analyzer.py --model_id <model> --analyze --batch_size <B> --seq_len <S>` - Legacy FLOP analysis

## Code Style

- Type hints required for all functions and class attributes
- Use dataclasses for structured data
- Follow PEP 8 naming conventions
- Prefer composition over inheritance
- Document complex algorithms with step-by-step comments

## Key Files

- `module_analyzer.py` - Cache-first module analysis with Claude Code agent fallback
- `module_analyzer_agent.py` - Claude Code agent for analyzing PyTorch module forward functions
- `module_db.json` - Formula template cache (starts empty, populated by agent)
- `module_db_schema_v2.json` - Standard JSON schema for module database structure
- `module_db_examples.json` - Example module analyses for agent templates
- `module_db_schema.json` - Legacy schema (deprecated, use v2)
- `model_analyzer.py` - Legacy model FLOP/memory analysis (hardware-specific)
- `DESIGN.md` - System architecture and design decisions
- `.env.example` - Environment variable template (copy to .env and fill in tokens)

## Architecture Overview

1. **Cache-First Analysis**: Check `module_db.json` before using Claude Code agent
2. **Formula Templates**: JSON templates with `${param}` and `{Module}()` syntax (unified for FLOP/memory)
3. **Claude Code Agent**: Automatically analyzes unknown PyTorch modules using source code
4. **Library Namespacing**: `torch__nn__Linear` format prevents naming conflicts
5. **Typed Parameters**: Parameters include name, type, and description for validation

## Important Notes

- NEVER store executable code in JSON - only formula templates
- Use registry system for module resolution: `compute_flops("torch.nn.Linear", ...)`, `compute_memory("torch.nn.Linear", ...)`
- Formula parameters use `${param}` syntax, module calls use `{Module}()` syntax (same for FLOP and memory templates)
- All separators (dots and underscores) become double underscores: `torch.nn.Linear` → `torch__nn__Linear`
- Always think harder first, do not rush to write code
- Use argparse to process all input arguments
- Always to ultrathink