# LM-Predictor
LM-Predictor is a Model FLOP/Memory Analysis System that uses AI agents to automatically analyze neural network modules and compute computational costs (FLOPs) and memory access patterns. 

## Critical Notes
- Never write code unless explicitly requested by the user, prefer to give guidelines first.

## Key Files
- `docs/DESIGN.md`: The design of this repo.
- `module_analyzer_agent_sdk.py`: Module analysis script
- `model_analyzer_agent_sdk.py`: Model analysis
- `model_analyzer.py`: Main entry
- `module_analysis_schema.json`: Analysis output schema
- `prompts/`: Prompt templates for analysis
- `modules/`: Module analysis dir
- `models/`: Model analysis dir

## Common Workflows

1. **Show Model Architecture**: Use `model_analyzer.py` with a HuggingFace model ID (e.g., `openai-community/gpt2`) to show its model architecture
2. **Analyze Individual Modules**: Use `module_analyzer_agent_sdk.py` to analyze each module from the model
3. **Aggregate Module Analysis**: Use `model_analyzer_agent_sdk.py` to aggregate all individual modules into a model analysis
4. **Calculate FLOPs/Memory**: Use `model_analyzer.py` to calculate the FLOPs/memory access according to the analyzed model

## External Dependencies

- **transformers/** submodule: HuggingFace Transformers source code for deep model introspection
- **pytorch/** submodule: PyTorch source code for understanding low-level operations
- Use `uv` for Python environment management
