# LM-Predictor Development Guide

## Core Principles

### 1. Design Documentation
- **DESIGN.md** is the single source of truth for system architecture and design principles
- Always keep DESIGN.md updated with the latest design decisions
- Reference DESIGN.md before making any architectural changes

### 2. Package Management with uv
UV is an ultra-fast Python package manager that replaces pip, poetry, virtualenv, and more.

**Project Setup:**
- `uv init` - Initialize new project with pyproject.toml
- `uv sync` - Sync dependencies after cloning repository

**Dependency Management:**
- `uv add <package>` - Add new dependency
- `uv remove <package>` - Remove dependency
- `uv pip list` - List installed packages

**Running Code:**
- `uv run <command>` - Run commands in managed environment
- `uv run python script.py` - Execute Python scripts
- Replace `python` with `uv run` for automatic environment management

**Environment Setup:**
- `uv python install <version>` - Install specific Python version
- `uv python list` - List available Python versions

### 3. Development Approach
- **Always ultrathink first** - Plan thoroughly before coding
- Break down complex tasks into clear, manageable steps
- Avoid rushing to implementation without proper planning