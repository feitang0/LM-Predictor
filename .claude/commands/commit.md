---
allowed-tools: Bash(git *), Read, Glob, Grep
description: Analyze file changes, categorize them, and create multiple commits based on categories
argument-hint: [optional-prefix-message]
---

## Context

- Current git status: !`git status`
- Current git diff (staged and unstaged changes): !`git diff HEAD`
- Current branch: !`git branch --show-current`
- Recent commits: !`git log --oneline -5`
- Modified files: !`git diff --name-only`
- Untracked files: !`git ls-files --others --exclude-standard`

## Your task

Analyze all the file changes above and categorize them into logical groups. Then create multiple focused commits based on these categories:

**Categories to consider:**
- **docs**: Documentation files (*.md, *.rst, *.txt, README*, docs/)
- **config**: Configuration files (*.json, *.yaml, *.toml, *.ini, pyproject.toml, setup.py, requirements.txt, .gitignore, .env*)
- **core**: Core functionality changes (main business logic)
- **features**: New features or significant additions
- **fixes**: Bug fixes and corrections
- **refactor**: Code refactoring and cleanup
- **tests**: Test files and testing-related changes
- **temp**: Temporary files that should be cleaned up (.DS_Store, *.tmp, __pycache__)

**Instructions:**
1. First, analyze each changed file and categorize it based on:
   - File path and extension patterns
   - Content of the changes (if it's a modification)
   - Whether it's adding new functionality, fixing issues, or refactoring

2. Group files by category and create separate commits for each category

3. Use descriptive commit messages that follow this pattern:
   - `<action> <category>: <brief description>`
   - Examples: "Update docs: improve README clarity", "Fix core: resolve authentication bug", "Add config: enable new feature flags"

4. Create commits in this order (most logical dependency):
   1. temp (cleanup first)
   2. config (configuration changes)
   3. core (core functionality)
   4. fixes (bug fixes)
   5. refactor (refactoring)
   6. features (new features)
   7. tests (tests)
   8. docs (documentation last)

5. For each commit, add the Claude Code signature:
   ```
   🤖 Generated with [Claude Code](https://claude.ai/code)

   Co-Authored-By: Claude <noreply@anthropic.com>
   ```

6. If a custom prefix message is provided as an argument ($ARGUMENTS), prepend it to each commit message

**Important:**
- Only commit files that are actually changed or new
- Skip categories that have no files
- Show a summary of what will be committed before proceeding
- Ask for confirmation before creating the commits