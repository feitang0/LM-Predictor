<system_context>
LM-Predictor analyzes FLOP and memory characteristics of PyTorch models by building a reusable database of module computation patterns. The system recursively decomposes models layer-by-layer, using cached formulas for known modules and Claude Code agent analysis for unknown ones.
</system_context>

<file_map>
## FILE MAP
- `/model_analyzer.py` - Main entry point for model FLOP/memory analysis
- `/module_analyzer.py` - Cache-first module analysis with Claude Code agent fallback
- `/module_generator_agent.py` - Claude Code agent for generating module files
- `/module_db.json` - Module analysis database with formula templates
- `/models/` - Standard model architecture JSON representations
- `/generated_modules/` - Auto-generated Python modules (flat structure)
  - `registry.py` - ModuleRegistry with auto-discovery and convenience functions
  - `base.py` - BaseModule abstract class
  - `torch_*.py` - Generated torch module calculators
  - `transformers_*.py` - Generated transformers module calculators
</file_map>

<paved_path>
## ARCHITECTURE (PAVED PATH)

### Pipeline Flow
```
Model → Analyze Architecture → Recursive Layer-by-Layer Computation → Results
  ↓              ↓                           ↓                           ↓
[Input]   Extract Modules           For Each Module:                   Report
          (layer-by-layer)              ↓
                                   Known Module?
                                   ├── Yes: Call generated_modules/ functions
                                   └── No: Agent Analysis
                                           ├── Output analyzed JSON
                                           ├── Store to database
                                           ├── Generate callable module
                                           └── Continue computation
```

### Standard Model Representation
All models use standardized hierarchical JSON structure in `models/` directory:

**Core Structure:**
```json
{
  "model_id": "model-name",
  "layers": [
    {
      "name": "layer_name",
      "class": "full.class.path.ClassName",
      "repeat": 32,                    // Optional: only if > 1
      "sub_layers": [...]              // Optional: only for composite layers
    }
  ]
}
```

**Basic vs Composite Layers:**
- **Basic Layers**: WITHOUT `sub_layers` field - perform actual computation
- **Composite Layers**: WITH `sub_layers` field - organizational containers

### Module Naming Conventions
- **Database Keys**: `library_ClassName` format (e.g., `torch_Linear`, `transformers_LlamaMLP`)
- **Full Class Names**: Complete Python paths (e.g., `torch.nn.modules.linear.Linear`)
- **Generated Classes**: `LibraryClassName` format (e.g., `TorchLinear`, `TransformersLlamaMLP`)
- **File Names**: snake_case from class names (e.g., `torch_linear.py`, `transformers_llama_mlp.py`)

### Formula Syntax
- **Parameters**: `${param_name}` - substituted with actual values
- **Module Calls**: `{ModuleName}` - dependent module names for FLOP/memory calculation

### Global Convenience Functions (PAVED PATH)
```python
from generated_modules.registry import get_required_parameters, compute_flops, compute_memory

# Get required parameters using PyTorch class name
required_params = get_required_parameters('torch.nn.modules.linear.Linear')

# Compute FLOPs and memory using the same module name
params = {'N': 10, 'in_features': 512, 'out_features': 256, 'dtype_bytes': 4}
flops = compute_flops('torch.nn.modules.linear.Linear', **params)
memory = compute_memory('torch.nn.modules.linear.Linear', **params)
```

This is the PREFERRED method - uses intuitive PyTorch class names directly with clean, consistent API.
</paved_path>

<patterns>
## PATTERNS

### Module Generation Pattern
```python
# GOOD: Generated module structure
class TorchLinear(BaseModule):
    def get_required_parameters(self) -> Dict[str, str]:
        return {"N": "int", "in_features": "int", "out_features": "int", "dtype_bytes": "int"}

    def compute_flops(self, **params: Dict[str, Any]) -> int:
        self.validate_parameters(**params)
        # Clear formula with breakdown
        matrix_mult_flops = 2 * params['N'] * params['in_features'] * params['out_features']
        bias_add_flops = params['N'] * params['out_features']
        return matrix_mult_flops + bias_add_flops
```

### Database Entry Pattern
```json
// GOOD: Complete module entry
{
  "torch_Linear": {
    "full_class_name": "torch.nn.modules.linear.Linear",
    "code_location": {"file": "pytorch/torch/nn/modules/linear.py", "line_start": 103},
    "flop_analysis": {
      "thinking_process": "Linear layer: y = xW^T + b. Matrix mult: 2*N*in*out FLOPs, bias add: N*out FLOPs",
      "parameters": [{"name": "N", "type": "int", "description": "batch size"}],
      "calculation_formula": "2 * ${N} * ${in_features} * ${out_features} + ${N} * ${out_features}",
      "module_depends": []
    },
    "memory_analysis": {
      "reads_calculation_formula": "${N} * ${in_features} * ${dtype_bytes} + ${out_features} * ${in_features} * ${dtype_bytes} + ${out_features} * ${dtype_bytes}",
      "writes_calculation_formula": "${N} * ${out_features} * ${dtype_bytes}",
      "intermediates_calculation_formula": "0"
    }
  }
}
```

### Registry Usage Pattern
```python
# GOOD: Use registry for module discovery
from generated_modules.registry import ModuleRegistry

registry = ModuleRegistry()
module_class = registry.get_module('torch.nn.modules.linear.Linear')
if module_class:
    instance = module_class()
    flops = instance.compute_flops(**params)

# BETTER: Use convenience functions
from generated_modules.registry import compute_flops
flops = compute_flops('torch.nn.modules.linear.Linear', **params)
```
</patterns>

<critical_notes>
## CRITICAL NOTES

- **Always plan thoroughly before coding** - Complex system with multiple interdependencies
- **Use `uv add <package>` to add dependencies** - Consistent package management
- **Use `uv run python script.py` to run Python scripts** - Proper virtual environment
- **NEVER use `any` types** - Type safety is mandatory throughout system
- **Basic layers only for calculations** - Layers WITHOUT `sub_layers` perform computation
- **Composite layers are organizational** - Layers WITH `sub_layers` are containers only
- **Database keys use underscore format** - `torch_Linear` not `torch.nn.Linear`
- **Generated classes use CamelCase** - `TorchLinear` not `torch_linear`
- **Formula templates not executable code** - JSON templates with `${param}` syntax
- **Agent analysis for unknown modules only** - Check database first, minimize agent calls
- **Validate parameters before computation** - Use `validate_parameters()` in all compute methods
- **Registry provides clean API** - Use convenience functions over direct class access
</critical_notes>

<workflow>
## WORKFLOW

### Analyze Model FLOPs/Memory
1. **Load model architecture** using `ModelAnalyzer(model_id="meta-llama/Llama-2-7b-hf")`
2. **Extract modules** to build layer inventory with full class names
3. **Check database** for each module using `module_analyzer.py`
4. **Generate missing modules** via Claude Code agent if not in database
5. **Compute recursively** through model hierarchy with parameter substitution

### Add New Module Type
1. **Check if exists** in `module_db.json` using full class name
2. **Use agent analysis** via `module_generator_agent.py` if not found
3. **Review generated entry** in `module_db.json` for accuracy
4. **Generate Python class** to `generated_modules/` directory
5. **Update registry mapping** in `registry.py` module_mapping dict
6. **Test calculations** with known parameter values

### Analyze Unknown Module
```bash
# Use module generator agent for analysis
uv run python module_generator_agent.py --full-class-name "torch.nn.modules.activation.GELU"
```

### Run Model Analysis
```bash
# Run complete model FLOP/memory analysis
uv run python model_analyzer.py --model-id "meta-llama/Llama-2-7b-hf"
# OR with pre-built JSON
uv run python model_analyzer.py --model-json models/llama-2-7b-hf.json
```

### Test Generated Module
```python
# Test individual module calculations
from generated_modules.registry import compute_flops, get_required_parameters

# Check parameters needed
params_needed = get_required_parameters('torch.nn.modules.linear.Linear')
print(params_needed)  # {'N': 'int', 'in_features': 'int', ...}

# Calculate with real values
flops = compute_flops('torch.nn.modules.linear.Linear', N=32, in_features=4096, out_features=11008, dtype_bytes=4)
```
</workflow>