# Work Progress - Model Architecture to Basic Layer FLOPs/Memory Calculation

## Current Objective
Convert hierarchical model architecture into layer-by-layer FLOPs/memory calculations at the basic computational layer level (not composite layers like LlamaDecoderLayer, but individual Linear, Embedding, etc.).

## Key Accomplishments

### 1. Standard Model Representation Structure ✅
- **File**: `models/llama-2-7b-hf.json`
- **Structure**: Hierarchical JSON with `layers`, `sub_layers`, `repeat` fields
- **Key Insight**: Layers WITHOUT `sub_layers` = Basic/computational layers
- **Key Insight**: Layers WITH `sub_layers` = Composite/organizational layers

### 2. Updated DESIGN.md ✅
- Added complete "Standard Model Representation" section
- Documented basic vs composite layer distinction
- Added layer identification rules: `"sub_layers" not in layer_dict`
- Updated file structure to include `models/` directory

### 3. Model Architecture Extraction ✅
- Verified actual Llama-2-7b-hf structure using `--inspect_only`
- Found accurate class names and hierarchy
- Corrected initial assumptions about model structure

### 4. Basic Layer Parameter Population (IN PROGRESS)
- **Completed**: First basic layer `embed_tokens`
- **Location**: `models/llama-2-7b-hf.json` line 8-17
- **Parameters Added**:
  ```json
  {
    "name": "embed_tokens",
    "class": "torch.nn.modules.sparse.Embedding",
    "parameters": {
      "num_indices": "{batch_size} * {seq_len}",
      "embedding_dim": 4096,
      "dtype_bytes": "{dtype_bytes}",
      "index_dtype_bytes": "{index_dtype_bytes}"
    }
  }
  ```

## Current State

### Files Modified
1. **DESIGN.md** - Added standard model representation documentation
2. **models/llama-2-7b-hf.json** - Created with correct structure, first basic layer populated
3. **models/** - New directory created

### Key Data Structures
- **Static Parameters**: Values from model architecture (e.g., `embedding_dim: 4096`)
- **Dynamic Parameters**: Template placeholders (e.g., `"{batch_size} * {seq_len}"`)

### Parameter Population Strategy
- Use `generated_modules/{module}.py` → `get_required_parameters()` to find what each layer needs
- Populate ALL required parameters (no blanks)
- Static values from model config/architecture
- Dynamic values as string templates with curly braces

## Next Steps (Remaining Work)

### Immediate Next Task
Continue populating basic layers with required parameters. **Next basic layer**: `q_proj` (first Linear layer in attention)

### Remaining Basic Layers to Populate (in order of appearance):
1. **q_proj, k_proj, v_proj, o_proj** (4× Linear layers in attention)
   - Class: `torch.nn.modules.linear.Linear`
   - Need: `in_features`, `out_features`, bias info
   - Location: Inside `self_attn` → `sub_layers`

2. **rotary_emb** (inside attention)
   - Class: `transformers.models.llama.modeling_llama.LlamaRotaryEmbedding`
   - Check `generated_modules/transformers_llama_rotary_embedding.py`

3. **gate_proj, up_proj, down_proj** (3× Linear layers in MLP)
   - Class: `torch.nn.modules.linear.Linear`
   - Different dimensions than attention layers

4. **act_fn** (SiLU activation)
   - Class: `torch.nn.modules.activation.SiLU`
   - Check `generated_modules/torch_silu.py`

5. **input_layernorm, post_attention_layernorm** (2× per decoder layer)
   - Class: `transformers.models.llama.modeling_llama.LlamaRMSNorm`
   - Check `generated_modules/transformers_llama_rms_norm.py`

6. **norm** (final model norm)
   - Class: `transformers.models.llama.modeling_llama.LlamaRMSNorm`

7. **rotary_emb** (model-level)
   - Class: `transformers.models.llama.modeling_llama.LlamaRotaryEmbedding`

8. **lm_head** (final linear layer)
   - Class: `torch.nn.modules.linear.Linear`
   - Maps hidden_size → vocab_size

### Process for Each Basic Layer
1. Identify layer in JSON (look for layers without `sub_layers`)
2. Find corresponding module in `generated_modules/`
3. Check `get_required_parameters()` method
4. Get static values from model architecture inspection
5. Use `{placeholder}` format for dynamic values
6. Add `"parameters": {...}` field to JSON

### Tools and Commands
```bash
# Inspect model architecture
uv run python model_analyzer.py --inspect_only

# Get specific layer parameters
uv run python -c "
# Load model and inspect specific layers
from transformers import AutoModelForCausalLM, AutoConfig
# ... check layer.in_features, layer.out_features, etc.
"

# Check what parameters a module needs
# Look at generated_modules/{module}.py → get_required_parameters()
```

### Model Architecture Summary (for reference)
- **Total Basic Layers**: ~260
  - Linear: 129 (4 per attention × 32 + 3 per MLP × 32 + 1 lm_head)
  - Embedding: 1
  - Normalization: 65 (2 per decoder × 32 + 1 final)
  - Activation: 32 (1 SiLU per MLP)
  - Rotary embedding: 33 (1 per attention + 1 model-level)

### Key Architecture Values (Llama-2-7b-hf)
- `hidden_size`: 4096
- `intermediate_size`: 11008
- `num_attention_heads`: 32
- `vocab_size`: 32000
- `num_hidden_layers`: 32

## Expected End Result
Complete `models/llama-2-7b-hf.json` with ALL basic layers populated with required parameters. This enables:
1. Layer-by-layer FLOP calculation using `generated_modules/`
2. Individual memory calculation for each computational layer
3. Precise analysis at the most granular level (Linear, Embedding, etc.)
4. Template for other model architectures

## Current Status: 1/260+ basic layers completed (embed_tokens)
**Next**: Continue with q_proj, k_proj, v_proj, o_proj Linear layers in attention mechanism.