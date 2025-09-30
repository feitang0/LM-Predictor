# Module Registry Naming Consistency Issue - 2025-09-30

## Problem Statement

**Error**: `Failed to analyze torch.nn.modules.activation.SiLU: Module not found: torch.nn.modules.activation.SiLU`

**Root Cause**: Agent uses semantic understanding for naming (e.g., `LlamaRMSNorm` → `llama_rms_norm`), but registry uses mechanical rules (→ `llamarmnsnorm`), causing mismatches.

### Detailed Analysis

**What happens now:**
```
module_analyzer_agent.py
  ↓ writes formulas to
module_db.json
  ↓ read by
module_generator_agent.py
  ↓ agent decides semantic naming
  - LlamaRMSNorm → file: transformers_llama_rms_norm.py, class: TransformersLlamaRMSNorm
  - SiLU → file: torch_silu.py, class: TorchSilu (not TorchSiLU!)
  ↓ writes result to
generation_result.json (naming decisions NOT persisted to module_db.json) ❌
  ↓
registry.py tries to load
  ↓ uses mechanical derivation from module_key
  - module_key: "transformers_LlamaRMSNorm"
  - file_name: module_key.lower() → "transformers_llamarmnsnorm" ❌ WRONG
  - class_name: "".join(part.capitalize() for part in parts) → "TransformersLlamarmnsnorm" ❌ WRONG
  ↓
MISMATCH: Expected vs Actual
  - Expected file: transformers_llama_rms_norm.py
  - Actual file: transformers_llamarmnsnorm (doesn't exist)
  - Expected class: TransformersLlamaRMSNorm
  - Actual class: TransformersLlamarmnsnorm (doesn't exist)
```

**Why this happens:**
- **Agent**: Uses semantic decomposition (understands acronyms, word boundaries)
  - `SiLU` → recognizes acronym → `si_lu`
  - `LlamaRMSNorm` → recognizes model + acronym + type → `llama_rms_norm`
- **Mechanical algorithm**: Just applies `.lower()` and `.capitalize()`
  - `SiLU` → `silu` (loses acronym structure)
  - `LlamaRMSNorm` → `llamarmnsnorm` (loses all boundaries)

### Current Workaround in Code

Registry has manual mapping (line 40-51):
```python
module_mapping = {
    "torch_SiLU": ("torch_silu", "TorchSilu"),  # But says TorchSiLU ❌ WRONG
    "transformers_LlamaRMSNorm": ("transformers_llama_rms_norm", "TransformersLlamaRMSNorm"),
    # ... 8 more manual entries
}
```

**Problem with manual mapping**:
- Doesn't scale (must update for every new module)
- Has errors (TorchSiLU should be TorchSilu)
- Agent's decisions not preserved

## Solution Discussion

### Option 1: Store Naming in module_db.json (RECOMMENDED)

Add agent-decided names to database after generation:

```json
{
  "transformers_LlamaRMSNorm": {
    "full_class_name": "transformers.models.llama.modeling_llama.LlamaRMSNorm",
    "generated_file_name": "transformers_llama_rms_norm.py",  // NEW
    "generated_class_name": "TransformersLlamaRMSNorm",        // NEW
    "flop_analysis": { ... }
  }
}
```

**Pros:**
- ✅ Single source of truth (database)
- ✅ Preserves agent's semantic naming decisions
- ✅ 100% consistency guarantee (no derivation = no mismatch)
- ✅ Scales automatically (all future modules)
- ✅ Documents actual naming for developers

**Cons:**
- Requires schema change
- Need to update existing modules

**Files to modify:**
1. **module_db_schema.json** - Add optional `generated_file_name` and `generated_class_name` fields
2. **module_generator_agent.py** - After generating file, write naming back to module_db.json
3. **registry.py** - Read naming from module_data, fallback to derivation for legacy entries
4. **Existing modules** - Regenerate with `--force --generate` to populate new fields

---

### Option 2: Dynamic File Discovery

Scan filesystem at runtime to find actual generated files:

```python
def _load_module_class(self, module_key: str, full_class_name: str):
    library = module_key.split("_")[0]
    for file_path in glob.glob(f"generated_modules/{library}_*.py"):
        # Import and inspect for matching full_class_name
```

**Pros:**
- ✅ Always accurate (uses actual files)
- ✅ No schema changes

**Cons:**
- ❌ Slower (imports multiple modules)
- ❌ Complex implementation
- ❌ Fragile matching logic

---

### Option 3: Standardize Agent Naming Rules

Force agents to use deterministic naming (no semantics):

```
Rule: Simply lowercase the class name
- SiLU → torch_silu.py (but actually torch_silu exists!)
- LlamaRMSNorm → transformers_llamarmnsnorm.py
```

**Pros:**
- ✅ Automatic derivation works
- ✅ No schema changes

**Cons:**
- ❌ Loses semantic meaning (unreadable)
- ❌ Harder to find files
- ❌ Goes against Python conventions

---

## Agreed Solution: Option 1

**Rationale:**
- Database already stores all module metadata
- Agents already write to module_db.json
- Registry just reads - no complex derivation
- Impossible to have mismatch if using stored names
- Best documentation for developers

## Implementation Plan

### 1. Update module_db_schema.json
Add two new optional fields to module entry:
```json
{
  "generated_file_name": {
    "type": "string",
    "description": "Actual generated filename (e.g., 'transformers_llama_rms_norm.py')"
  },
  "generated_class_name": {
    "type": "string",
    "description": "Actual generated class name (e.g., 'TransformersLlamaRMSNorm')"
  }
}
```

### 2. Update module_generator_agent.py
After successful generation (after writing generation_result.json):

```python
# Read generation result
with open("generation_result.json", 'r') as f:
    gen_result = json.load(f)

if gen_result["status"] == "success":
    # Update module_db.json with naming info
    db = load_db()
    for module_key, module_data in db["modules"].items():
        if module_data["full_class_name"] == full_class_name:
            module_data["generated_file_name"] = os.path.basename(gen_result["module_file"])
            module_data["generated_class_name"] = gen_result["class_name"]
            break
    save_db(db)
```

### 3. Update registry.py
In `_load_module_class()`:

```python
def _load_module_class(self, module_key: str, full_class_name: str) -> Optional[Type[BaseModule]]:
    try:
        # Get module data from database
        with open(self._module_db_path, 'r') as f:
            db = json.load(f)

        module_data = db.get("modules", {}).get(module_key, {})

        # Try to read stored naming (new approach)
        file_name = module_data.get("generated_file_name")
        class_name = module_data.get("generated_class_name")

        # Fallback to derivation for legacy entries
        if not file_name or not class_name:
            file_name = module_key.lower()
            parts = module_key.split("_")
            class_name = "".join(part.capitalize() for part in parts)
        else:
            # Remove .py extension from stored filename
            file_name = file_name.replace(".py", "")

        module_path = f"generated_modules.{file_name}"
        module = importlib.import_module(module_path)
        return getattr(module, class_name, None)

    except (ImportError, AttributeError) as e:
        print(f"Warning: Could not load module {module_key}: {e}")
        return None
```

Remove manual mapping dict entirely (lines 40-51).

### 4. Regenerate Existing Modules
Run for all 5 existing modules:
```bash
uv run python module_analyzer.py torch.nn.modules.activation.SiLU --force --generate
uv run python module_analyzer.py torch.nn.modules.linear.Linear --force --generate
uv run python module_analyzer.py torch.nn.modules.sparse.Embedding --force --generate
uv run python module_analyzer.py transformers.models.llama.modeling_llama.LlamaRMSNorm --force --generate
uv run python module_analyzer.py transformers.models.llama.modeling_llama.LlamaRotaryEmbedding --force --generate
```

This populates the new fields in module_db.json.

### 5. Testing
```bash
# Test the fix
uv run python model_analyzer.py --analyze \
  --model_json models/meta-llama-Llama-2-7b-hf_populated.json \
  --batch_size 1 --seq_len 2048 --w_bit 16 --a_bit 16
```

Should see no "Module not found" errors.

## Key Insights from Discussion

1. **Only modifying module_generator_agent.py is NOT sufficient** - Need to update schema, registry, and regenerate existing modules

2. **Agent makes intelligent naming decisions** that mechanical algorithms cannot replicate:
   - Recognizes acronyms (RMS, SDPA, SiLU)
   - Understands word boundaries (LlamaRMSNorm → llama_rms_norm)
   - Applies Python conventions (snake_case for files, PascalCase for classes)

3. **The real issue**: These naming decisions are NOT persisted anywhere, causing registry to guess incorrectly

4. **Why Option 1 is best**:
   - Preserves agent intelligence
   - Single source of truth (database)
   - Scales automatically
   - Prevents future mismatches

## Status

**Current**: Plan documented, ready for implementation
**Next**: User approval, then implement all 5 steps above
**Expected Outcome**: 100% consistent naming, no more "Module not found" errors