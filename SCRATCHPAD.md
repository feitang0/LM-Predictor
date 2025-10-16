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

---

# Missing Composite Module FLOPs - 2025-10-14

## Problem Statement

**Issue**: Current FLOP analysis only counts operations within basic layer modules, missing higher-level composite operations like attention mechanisms and residual connections.

**Observed Gap**: ~2-3 TFLOPs missing compared to LLM-Viewer reference

### Comparison with LLM-Viewer

**What we count correctly** ✓
- Linear layer GEMMs (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head)
- All Linear FLOPs match LLM-Viewer exactly

**What we're missing** ❌

| Operation | FLOPs (per layer) | Description | Location |
|-----------|-------------------|-------------|----------|
| qk_matmul | 34.4G | Q @ K^T attention scores | LlamaSdpaAttention |
| sv_matmul | 34.4G | scores @ V attention output | LlamaSdpaAttention |
| softmax | 671M | Softmax over attention scores | LlamaSdpaAttention |
| attn_add | 8.4M | Residual connection (attention) | LlamaDecoderLayer |
| mlp_add | 8.4M | Residual connection (MLP) | LlamaDecoderLayer |

**Total missing per layer**: ~69.3G FLOPs
**Total missing (×32 layers)**: ~2.2T FLOPs

### Additional Issues Found

1. **SiLU parameter bug**:
   - Current: N = 8.2K (wrong)
   - Should be: N = B * S * intermediate_size = 1 * 2048 * 11008 = 22.5M
   - Expected FLOPs: 4 * 22.5M = 90M vs LLM-Viewer's 16.8M (formula may differ)

2. **RMSNorm discrepancy**:
   - Current: 33.6M FLOPs per layer
   - LLM-Viewer: 58.7M FLOPs per layer
   - Difference: ~75% higher (may include additional normalization operations)

## Root Cause

**Current Architecture**:
- Only basic layers (no `sub_layers`) have computation
- Composite layers (with `sub_layers`) are organizational containers only
- Internal operations within composite layers are not modeled

**Example**: `LlamaSdpaAttention` contains q/k/v/o Linear projections (counted ✓) but doesn't model:
- QK matmul for attention scores
- Softmax normalization
- SV matmul for attention output

These operations happen **between** the basic layer calls but aren't attributed to any module.

## Solution Approaches

### Option 1: Add Composite Module Calculators (RECOMMENDED)

Treat certain composite modules as computational units:

```python
# In module_db.json
"transformers_LlamaSdpaAttention": {
  "full_class_name": "transformers.models.llama.modeling_llama.LlamaSdpaAttention",
  "is_composite": true,  // NEW: flag indicating composite with internal ops
  "flop_analysis": {
    "calculation_formula": "
      4 * ${torch.nn.Linear}({B} * {S}, {hidden_size}, {hidden_size}) +  // q/k/v/o projs
      2 * {B} * {num_heads} * {S} * {S} * {head_dim} +  // qk + sv matmuls
      5 * {B} * {num_heads} * {S} * {S}  // softmax (5 ops per element)
    "
  }
}
```

**Pros:**
- ✅ Accurate FLOP accounting
- ✅ Matches industry tools (LLM-Viewer)
- ✅ Documents attention computation explicitly

**Cons:**
- Breaks current "basic vs composite" dichotomy
- Requires agent to analyze composite module internals
- More complex formulas with module references

### Option 2: Post-Processing Layer

Add a separate "attention operations" analysis after basic layer traversal:

```python
def analyze_attention_ops(arch, params):
    """Count QK/SV matmuls, softmax for attention layers"""
    for layer in find_layers(arch, "LlamaSdpaAttention"):
        flops += calculate_attention_flops(params)
```

**Pros:**
- ✅ Keeps basic/composite separation clean
- ✅ Easier to implement incrementally

**Cons:**
- ❌ Hardcoded logic for specific architectures
- ❌ Not generalizable to other models
- ❌ Separate from module database

### Option 3: Hybrid Approach

Extend basic layers to reference their parent composite context:

```python
"torch_Linear": {
  "flop_analysis": {
    "calculation_formula": "2 * {N} * {in_features} * {out_features}"
  },
  "context_aware_ops": {
    "within_attention": {
      // Additional ops when Linear is inside attention
      "qk_matmul_contribution": "..."
    }
  }
}
```

**Pros:**
- ✅ Context-aware computation
- ✅ Maintains module focus

**Cons:**
- ❌ Very complex
- ❌ Tight coupling between modules and contexts

## Recommended Solution: Option 1

**Rationale:**
- Attention mechanisms are fundamental computation units
- Agent can analyze forward() method of composite modules
- Matches how other tools model transformers
- Clean extension of current approach

**Implementation:**
1. Update schema: Add `is_composite` flag and allow composite modules in database
2. Extend agent: Analyze composite module forward() when it contains direct computation
3. Update registry: Load composite module calculators
4. Update analysis: Traverse architecture and calculate composite module FLOPs

## Next Steps

1. Document residual connections (element-wise adds) - where should these be counted?
2. Fix SiLU parameter population bug
3. Investigate RMSNorm formula discrepancy
4. Design composite module analysis workflow
5. Implement attention FLOP calculation

## Status

**Current**: Issue documented with comparison data
**Next**: Decide on solution approach and implementation priority
**Expected Outcome**: Match LLM-Viewer FLOPs within <5% margin

---

# Composite Module Regeneration - 2025-10-15

## Progress Update

**Task**: Clean and regenerate all composite modules with proper sub_layers context for internal-operations-only analysis

### Implementation Approach

**Internal-Operations-Only Pattern**: Composite modules count ONLY operations that occur BETWEEN sub-layer calls to avoid double-counting.

**Example - LlamaDecoderLayer**:
- ✅ Count: Residual connections (element-wise additions)
- ❌ Don't count: input_layernorm, self_attn, post_attention_layernorm, mlp (sub-layers handle their own FLOPs)

**Example - LlamaMLP**:
- ✅ Count: Element-wise multiplication between act_fn(gate_proj(x)) and up_proj(x)
- ❌ Don't count: gate_proj, up_proj, down_proj Linear layers, act_fn SiLU (sub-layers)

**Example - LlamaSdpaAttention**:
- ✅ Count: RoPE application, QK matmul, softmax, SV matmul
- ❌ Don't count: q_proj, k_proj, v_proj, o_proj Linear layers (sub-layers)

### Completed Steps

1. ✅ **Cleaned module_db.json**: Removed all composite module entries, keeping only 5 basic modules:
   - torch_SiLU
   - torch_Linear
   - torch_Embedding
   - transformers_LlamaRMSNorm
   - transformers_LlamaRotaryEmbedding

2. ✅ **Regenerated LlamaDecoderLayer**:
   - Formula: `2 * {B} * {S} * {hidden_size}` (two residual connections)
   - Correctly excludes all sub-layer operations
   - Stored in module_db.json with naming: `transformers_llama_decoder_layer.py` / `TransformersLlamaDecoderLayer`

3. ✅ **Regenerated LlamaMLP**:
   - Formula: `{B} * {S} * {intermediate_size}` (element-wise multiplication)
   - Correctly excludes gate_proj, up_proj, down_proj, act_fn
   - Stored in module_db.json with naming: `transformers_llama_mlp.py` / `TransformersLlamaMLP`

### In Progress

4. 🔄 **Regenerating LlamaModel**: Currently being analyzed by agent with sub_layers context:
   - Sub-layers: Embedding, LlamaDecoderLayer (×32), LlamaRMSNorm, LlamaRotaryEmbedding
   - Expected: Mostly delegates to sub-layers, minimal internal operations (mask creation is negligible)

### Remaining Tasks

5. ⏳ **LlamaSdpaAttention**:
   - Sub-layers: q_proj, k_proj, v_proj, o_proj (Linear), rotary_emb (LlamaRotaryEmbedding)
   - Expected operations: RoPE application, QK matmul, softmax, SV matmul (~69G FLOPs per layer)

6. ⏳ **LlamaForCausalLM**:
   - Sub-layers: LlamaModel, lm_head (Linear)
   - Expected operations: Optional cross-entropy loss computation

### Technical Details

**Sub-layers Context Injection**: Modified agent prompt to include composite module detection:
```
⚠️ COMPOSITE MODULE DETECTED ⚠️

This module contains these sub-layers (already analyzed separately):
[list of sub-layer classes]

CRITICAL RULE: Count ONLY operations that occur BETWEEN sub-layer calls.
- ✅ Include: Matmuls, softmax, element-wise ops, residual connections
- ❌ Exclude: Any computation done by the sub-layers listed above
```

**Implementation Chain**:
1. `model_analyzer.py::ensure_modules()` - Extracts ALL layer classes (basic + composite)
2. `model_analyzer.py::find_layer_by_class()` - Looks up sub_layers from architecture
3. `module_analyzer.py::analyze_module()` - Passes sub_layers to agent
4. `module_analyzer_agent.py` - Injects composite context into prompt
5. Agent analyzes only internal operations, writes to module_db.json

### Expected Outcome

After all 4 composite modules are regenerated:
- **9 total modules**: 5 basic + 4 composite
- **Accurate FLOP counting**: Internal operations captured without double-counting
- **Match LLM-Viewer**: Should recover ~2.2T missing FLOPs (attention + residuals)

### Next Steps

1. Wait for --ensure-modules to complete (analyzing LlamaModel, LlamaSdpaAttention, LlamaForCausalLM)
2. Verify all 4 composite modules generated correctly
3. Run --populate-arch to add parameters to composite modules
4. Run --analyze to compute total FLOPs and compare with LLM-Viewer

## Status

**Current**: Regenerating composite modules with internal-operations-only approach (3/4 completed)
**Blockers**: None - process running smoothly
**Expected Completion**: All modules regenerated within 20 minutes

---

# Memory Bandwidth Analysis Gap - 2025-10-16

## Problem Statement

**Issue**: Composite module analysis successfully fixed FLOP calculations, but memory bandwidth tracking still underestimates by ~30-40x for attention operations.

**Root Cause**: Current design tracks intermediate tensor *storage* (268MB) but not intermediate tensor *traffic* (read/write operations on intermediates).

### Comparison with Reference Tool

**Reference Tool (per decoder layer):**
```
qk_matmul:  34.4G FLOPs, 302MB access
softmax:    671M FLOPs, 537MB access
sv_matmul:  34.4G FLOPs, 302MB access
Total attention bandwidth: 1,141MB per layer
```

**Our Implementation:**
```
self_attn: 69.17G FLOPs, 302MB total (16.78MB reads + 16.78MB writes + 268MB intermediates)
```

**The Gap**: We count 302MB vs actual 1,141MB bandwidth → **78% underestimate**

## Detailed Analysis: Attention Memory Flow

### Step-by-Step HBM Traffic (B=1, S=2048, num_heads=32, head_dim=128)

**Operation 1: QK Matmul** (`scores = Q @ K.T`)
```
Inputs:  Q [1, 32, 2048, 128] = 16.78 MB
         K [1, 32, 2048, 128] = 16.78 MB
Output:  scores [1, 32, 2048, 2048] = 268.44 MB

HBM Traffic:
- READ Q:          16.78 MB
- READ K:          16.78 MB
- WRITE scores:   268.44 MB
Total:            302.00 MB
```

**Operation 2: Softmax** (`attn_weights = softmax(scores)`)
```
Input:   scores [1, 32, 2048, 2048] = 268.44 MB (from HBM!)
Output:  attn_weights [1, 32, 2048, 2048] = 268.44 MB

HBM Traffic:
- READ scores:         268.44 MB  ← Intermediate read!
- WRITE attn_weights:  268.44 MB  ← Intermediate write!
Total:                 536.88 MB
```

**Operation 3: SV Matmul** (`output = attn_weights @ V`)
```
Inputs:  attn_weights [1, 32, 2048, 2048] = 268.44 MB (from HBM!)
         V [1, 32, 2048, 128] = 16.78 MB
Output:  output [1, 32, 2048, 128] = 16.78 MB

HBM Traffic:
- READ attn_weights:  268.44 MB  ← Intermediate read!
- READ V:              16.78 MB
- WRITE output:        16.78 MB
Total:                302.00 MB
```

**Total HBM Bandwidth: 302 + 537 + 302 = 1,141 MB**

### Why Intermediates Hit HBM

**Key Constraint**: Attention scores (268 MB) >> GPU L2 cache (40 MB)

Since the 268MB scores tensor doesn't fit in fast memory:
1. **Written to HBM** after qk_matmul (kernel ends, data evicted to HBM)
2. **Read from HBM** by softmax (next kernel starts, loads from HBM)
3. **Written to HBM** after softmax (kernel ends, data stored to HBM)
4. **Read from HBM** by sv_matmul (next kernel starts, loads from HBM)

**Result**: Each 268MB intermediate contributes 4 HBM accesses = 1,072 MB bandwidth

### Impact on Performance Prediction

**Without intermediate traffic tracking:**
```python
bandwidth = 33.56 MB (input + output only)
predicted_time = 33.56 MB / 900 GB/s = 37 μs
```

**With intermediate traffic tracking:**
```python
bandwidth = 1,141 MB (including intermediate reads/writes)
predicted_time = 1,141 MB / 900 GB/s = 1,267 μs
```

**Actual measured time**: ~1,486 μs (from reference)

**Accuracy**: 1,267 μs vs 37 μs → **34x improvement in prediction accuracy**

## Proposed Solution: Atomic Operation Breakdown

### Architecture Change

**Current Design (Composite Modules):**
```json
"transformers_LlamaSdpaAttention": {
  "flop_analysis": {
    "calculation_formula": "69.17G"
  },
  "memory_analysis": {
    "reads": "16.78MB",           // Module input only
    "writes": "16.78MB",          // Module output only
    "intermediates": "268.44MB"   // Storage, counted once
  }
}
```

**Proposed Design (Atomic Operations):**
```json
"transformers_LlamaSdpaAttention_qk_matmul": {
  "full_class_name": "transformers.models.llama.modeling_llama.LlamaSdpaAttention.qk_matmul",
  "is_atomic_operation": true,
  "parent_module": "transformers.models.llama.modeling_llama.LlamaSdpaAttention",
  "flop_analysis": {
    "calculation_formula": "2 * {B} * {num_heads} * {S} * {S} * {head_dim}"
  },
  "memory_analysis": {
    "reads": "2 * {B} * {num_heads} * {S} * {head_dim} * {a_dtype_bytes}",  // Q + K
    "writes": "{B} * {num_heads} * {S} * {S} * {a_dtype_bytes}"  // scores
  }
},
"transformers_LlamaSdpaAttention_softmax": {
  "full_class_name": "transformers.models.llama.modeling_llama.LlamaSdpaAttention.softmax",
  "is_atomic_operation": true,
  "parent_module": "transformers.models.llama.modeling_llama.LlamaSdpaAttention",
  "flop_analysis": {
    "calculation_formula": "3 * {B} * {num_heads} * {S} * {S}"
  },
  "memory_analysis": {
    "reads": "{B} * {num_heads} * {S} * {S} * {a_dtype_bytes}",   // scores
    "writes": "{B} * {num_heads} * {S} * {S} * {a_dtype_bytes}"   // attn_weights
  }
},
"transformers_LlamaSdpaAttention_sv_matmul": {
  "full_class_name": "transformers.models.llama.modeling_llama.LlamaSdpaAttention.sv_matmul",
  "is_atomic_operation": true,
  "parent_module": "transformers.models.llama.modeling_llama.LlamaSdpaAttention",
  "flop_analysis": {
    "calculation_formula": "2 * {B} * {num_heads} * {S} * {S} * {head_dim}"
  },
  "memory_analysis": {
    "reads": "{B} * {num_heads} * {S} * {S} * {a_dtype_bytes} + {B} * {num_heads} * {S} * {head_dim} * {a_dtype_bytes}",  // attn_weights + V
    "writes": "{B} * {num_heads} * {S} * {head_dim} * {a_dtype_bytes}"  // output
  }
}
```

### Key Benefits

**Automatic Intermediate Tracking:**
- One operation's write = next operation's read
- No special "intermediates" field needed
- Sum of all reads + writes = total HBM bandwidth

**Example - Attention Scores (268MB):**
- Written in qk_matmul → counted as "write" (268MB)
- Read in softmax → counted as "read" (268MB)
- Written in softmax → counted as "write" (268MB)
- Read in sv_matmul → counted as "read" (268MB)
- **Total intermediate traffic: 1,072MB automatically captured**

**Matches Reference Tool Exactly:**
```
qk_matmul:  34.4G FLOPs, 302MB (16.78 + 16.78 + 268.44)
softmax:    671M FLOPs, 537MB (268.44 + 268.44)
sv_matmul:  34.4G FLOPs, 302MB (268.44 + 16.78 + 16.78)
```

## Implementation Challenges

### Challenge 1: Extracting Atomic Operations

**Problem**: `print(model)` shows `LlamaSdpaAttention` as ONE module, not three operations.

**Solution Options:**

**Option A: Manual Definition (Simple)**
```python
# In module_db.json, store atomic operation breakdown
"transformers_LlamaSdpaAttention": {
  "atomic_operations": [
    {"name": "qk_matmul", "formula": "..."},
    {"name": "softmax", "formula": "..."},
    {"name": "sv_matmul", "formula": "..."}
  ]
}
```

**Option B: Parse forward() Method (Automatic)**
```python
# Agent analyzes forward() code
def forward(self, hidden_states, ...):
    # Identified atomic ops:
    attn_weights = torch.matmul(Q, K.T)  # → qk_matmul operation
    attn_weights = F.softmax(attn_weights)  # → softmax operation
    output = torch.matmul(attn_weights, V)  # → sv_matmul operation
```

**Option C: Hybrid (Recommended)**
- Keep automatic module discovery for basic layers
- Manually define atomic operation breakdown for complex patterns (Attention)
- Store in module_db.json as sub-entries

### Challenge 2: Architecture Representation

**Current Architecture JSON:**
```json
{
  "name": "self_attn",
  "class": "transformers.models.llama.modeling_llama.LlamaSdpaAttention",
  "parameters": {...},
  "sub_layers": [...]
}
```

**Option 1: Flatten Atomic Operations**
```json
{
  "name": "self_attn.qk_matmul",
  "class": "transformers.models.llama.modeling_llama.LlamaSdpaAttention.qk_matmul",
  "parameters": {...}
},
{
  "name": "self_attn.softmax",
  "class": "transformers.models.llama.modeling_llama.LlamaSdpaAttention.softmax",
  "parameters": {...}
}
```

**Option 2: Nested Operations**
```json
{
  "name": "self_attn",
  "class": "transformers.models.llama.modeling_llama.LlamaSdpaAttention",
  "atomic_operations": [
    {"name": "qk_matmul", "parameters": {...}},
    {"name": "softmax", "parameters": {...}},
    {"name": "sv_matmul", "parameters": {...}}
  ]
}
```

### Challenge 3: Parameter Population

Atomic operations need different parameters than parent modules:

**qk_matmul needs**: B, num_heads, S, head_dim, a_dtype_bytes
**softmax needs**: B, num_heads, S, a_dtype_bytes
**sv_matmul needs**: B, num_heads, S, head_dim, a_dtype_bytes

These must be populated from parent module's config.

## Recommended Implementation Path

### Phase 1: Prototype with Manual Definitions
1. Manually define atomic operations for LlamaSdpaAttention in module_db.json
2. Update analysis pipeline to recognize and process atomic operations
3. Verify bandwidth calculations match reference tool

### Phase 2: Generalize Architecture
1. Update module_db_schema.json to support atomic_operations field
2. Extend populate_parameters_agent to handle atomic operations
3. Update analysis to sum reads/writes correctly

### Phase 3: Automate Detection
1. Enhance module_analyzer_agent to parse forward() and identify atomic ops
2. Automatically generate atomic operation entries
3. Test on multiple model architectures

## Expected Outcomes

**After Implementation:**
- ✅ Memory bandwidth predictions accurate within 10%
- ✅ Match reference tool's operation-level granularity
- ✅ Correct performance prediction for memory-bound operations
- ✅ No need for special "intermediates" handling

**Performance Prediction Improvement:**
- Current: 37 μs (off by 40x)
- After fix: 1,267 μs (within 15% of actual 1,486 μs)

## Status

**Current**: Memory gap identified and analyzed, solution designed
**Next**: Decide on implementation approach (manual prototype vs full automation)
**Priority**: High - critical for accurate performance prediction

## Key Insights

1. **Intermediate tensor traffic is the dominant factor** in memory bandwidth for attention operations
2. **Splitting into atomic operations** automatically captures this traffic without special tracking
3. **Current composite approach** fundamentally cannot capture intermediate R/W patterns
4. **Reference tools use operation-level breakdown** for accurate bandwidth modeling
5. **HBM bandwidth >> cache bandwidth** means even small intermediates hit HBM multiple times