# GPT2Attention Module Analysis

## Source Code Location
- **File**: `/Users/tangfei/Dev/LM-Predictor/transformers/src/transformers/models/gpt2/modeling_gpt2.py`
- **Class**: `GPT2Attention` (lines 138-357)
- **Forward Method**: Lines 306-357

## Module Structure Investigation

### __init__ Analysis (Lines 139-181)
Looking at the __init__ method to understand the module architecture:

**Line 152-154**: Key configuration parameters
```python
self.embed_dim = config.hidden_size
self.num_heads = config.num_attention_heads
self.head_dim = self.embed_dim // self.num_heads
```
- `embed_dim` = `hidden_size` from config
- `num_heads` = `num_attention_heads` from config
- `head_dim` = `hidden_size / num_heads`

**Line 170-175**: Weight modules
```python
if self.is_cross_attention:
    self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
    self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
else:
    self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
self.c_proj = Conv1D(self.embed_dim, self.embed_dim)
```

For standard self-attention (is_cross_attention=False):
- `c_attn`: Conv1D that projects from `embed_dim` to `3 * embed_dim` (for Q, K, V)
- `c_proj`: Conv1D that projects from `embed_dim` to `embed_dim` (output projection)

**Conv1D Module** (from pytorch_utils.py, lines 83-105):
Conv1D is essentially a Linear layer but with transposed weights. Looking at line 103:
```python
x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
```
This performs: `output = x @ weight + bias`
- For Conv1D(nf, nx): weight shape is (nx, nf), where nx=input_features, nf=output_features
- This is equivalent to a Linear layer from nx to nf

**Line 177-178**: Dropout modules
```python
self.attn_dropout = nn.Dropout(config.attn_pdrop)
self.resid_dropout = nn.Dropout(config.resid_pdrop)
```

## Default Inference Configuration

Based on the code structure, the default inference path is:
1. **use_cache = True** (line 314, 339-342): Cache key/value tensors for autoregressive generation
2. **training = False**: No gradient computation
3. **encoder_hidden_states = None** (line 317-328): Standard self-attention, not cross-attention
4. **reorder_and_upcast_attn = False** (line 344-347): Use standard _attn method, not _upcast_and_reordered_attn
5. **output_attentions = False** (line 354-355): Don't return attention weights
6. **head_mask = None**: No head masking
7. **layer_past may or may not be None**: This affects whether we concatenate cached keys/values

For this analysis, I'll trace TWO scenarios:
- **Scenario A**: Initial forward pass (layer_past = None, cache_len = 0)
- **Scenario B**: Cached forward pass (layer_past != None, cache_len > 0)

The primary path is through lines 328 → 330-332 → 336-337 (if cache exists) → 347 → 349-351 → 357.

## Variable and Shape Mapping

### Input Parameters
- `hidden_states`: Shape `(batch_size, seq_len, hidden_size)`
  - This is the input to the attention layer
- `layer_past`: Optional tuple of `(past_key, past_value)`, each with shape `(batch_size, num_heads, cache_len, head_dim)`
  - Contains cached key/value tensors from previous forward passes
  - `cache_len` is the length of previously processed tokens
- `attention_mask`: Optional, shape `(batch_size, 1, seq_len, cache_len + seq_len)`
  - Mask for attention scores
- `head_mask`: Optional, shape `(num_heads,)` or `(batch_size, num_heads, seq_len, cache_len + seq_len)`
  - For masking specific attention heads

### Configuration Parameters
From __init__ (lines 152-154):
- `hidden_size` = config.hidden_size (also referred to as embed_dim)
- `num_heads` = config.num_attention_heads
- `head_dim` = hidden_size / num_heads

### Intermediate Tensors and Shape Transformations

**After c_attn projection (line 328)**:
- `self.c_attn(hidden_states)` → shape `(batch_size, seq_len, 3 * hidden_size)`
- After split: `query, key, value` → each has shape `(batch_size, seq_len, hidden_size)`

**After _split_heads (lines 330-332)**:
Looking at _split_heads implementation (lines 290-296):
```python
def _split_heads(self, tensor, num_heads, attn_head_size):
    new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
    tensor = tensor.view(new_shape)
    return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
```
- Input: `(batch_size, seq_len, hidden_size)`
- After view: `(batch_size, seq_len, num_heads, head_dim)`
- After permute: `(batch_size, num_heads, seq_len, head_dim)`

So after lines 330-332:
- `query`: `(batch_size, num_heads, seq_len, head_dim)`
- `key`: `(batch_size, num_heads, seq_len, head_dim)`
- `value`: `(batch_size, num_heads, seq_len, head_dim)`

**After cache concatenation (lines 336-337)** (if layer_past exists):
```python
key = torch.cat((past_key, key), dim=-2)
value = torch.cat((past_value, value), dim=-2)
```
- `past_key`: `(batch_size, num_heads, cache_len, head_dim)`
- `key` before concat: `(batch_size, num_heads, seq_len, head_dim)`
- `key` after concat: `(batch_size, num_heads, cache_len + seq_len, head_dim)`
- Same for `value`

**In _attn method** (lines 198-236):
- `attn_weights` after matmul (line 199): `(batch_size, num_heads, seq_len, cache_len + seq_len)`
- `attn_output` after matmul (line 234): `(batch_size, num_heads, seq_len, head_dim)`

**After _merge_heads (line 349)**:
Looking at _merge_heads implementation (lines 298-304):
```python
def _merge_heads(self, tensor, num_heads, attn_head_size):
    tensor = tensor.permute(0, 2, 1, 3).contiguous()
    new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
    return tensor.view(new_shape)
```
- Input: `(batch_size, num_heads, seq_len, head_dim)`
- After permute: `(batch_size, seq_len, num_heads, head_dim)`
- After view: `(batch_size, seq_len, hidden_size)`

**After c_proj (line 350)**:
- Input: `(batch_size, seq_len, hidden_size)`
- Output: `(batch_size, seq_len, hidden_size)`

## Standardized Variable Names

For all formulas, I'll use:
- `batch_size`: Batch dimension
- `seq_len`: Current sequence length (query length)
- `cache_len`: Cached sequence length (0 if no cache, >0 if cached)
- `kv_len` = `cache_len + seq_len`: Total key/value length
- `hidden_size`: Model hidden dimension (embed_dim)
- `num_heads`: Number of attention heads
- `head_dim`: Per-head dimension (hidden_size / num_heads)
- `w_bytes`: Weight precision in bytes
- `a_bytes`: Activation precision in bytes

## Line-by-Line Analysis of forward() Method

### Line 317-326: Cross-attention branch (SKIPPED - not default path)
We skip this because encoder_hidden_states is None in the default self-attention case.

---

### Line 328: QKV Projection
```python
query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
```

**Kernel Type**: composite

**Operation**: Combined QKV projection through Conv1D, followed by split

**Analysis**:
This line performs two operations:
1. `self.c_attn(hidden_states)`: Conv1D projection
2. `.split(self.split_size, dim=2)`: Tensor split operation

The `c_attn` is a Conv1D module (from line 174: `Conv1D(3 * self.embed_dim, self.embed_dim)`).
Looking at Conv1D.__init__ (pytorch_utils.py line 94-99):
- `nf = 3 * hidden_size` (output features)
- `nx = hidden_size` (input features)
- Weight shape: `(nx, nf) = (hidden_size, 3 * hidden_size)`
- Bias shape: `(nf,) = (3 * hidden_size,)`

The Conv1D.forward (pytorch_utils.py line 101-105) performs:
```python
x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
```
This is equivalent to: `output = input @ weight + bias`

For the module call, I need to reference `transformers.pytorch_utils.Conv1D` with parameters:
- batch_size, seq_len: From input_ids.shape propagated through the model
- in_features = hidden_size: Input dimension
- out_features = 3 * hidden_size: Output dimension

The split operation divides the last dimension into 3 equal parts:
- Input to split: `(batch_size, seq_len, 3 * hidden_size)`
- After split: 3 tensors, each `(batch_size, seq_len, hidden_size)`

The split is a view operation (no data copy), so it has zero memory cost.

**FLOPs**:
- Conv1D module: `${transformers.pytorch_utils.Conv1D}(batch_size, seq_len, hidden_size, 3 * hidden_size)`
- Split: 0 (view operation)

**Memory Access**:
- Conv1D module: `${transformers.pytorch_utils.Conv1D}(batch_size, seq_len, hidden_size, 3 * hidden_size)`
- Split: 0 (view operation)

---

### Line 330: Query head split
```python
query = self._split_heads(query, self.num_heads, self.head_dim)
```

**Kernel Type**: basic

**Operation**: Reshape and permute query tensor to split into multiple heads

**Analysis**:
The `_split_heads` method (lines 290-296) performs:
1. View: `(batch_size, seq_len, hidden_size)` → `(batch_size, seq_len, num_heads, head_dim)`
2. Permute: `(batch_size, seq_len, num_heads, head_dim)` → `(batch_size, num_heads, seq_len, head_dim)`

Both view and permute are typically zero-copy operations in PyTorch when the underlying data is contiguous. However, permute returns a view with a different stride, and when this is later used in computations, it may trigger a contiguous() call which would copy data.

For conservative analysis, I'll count permute as reading and writing the tensor data:
- Read: `batch_size * seq_len * num_heads * head_dim = batch_size * seq_len * hidden_size` elements
- Write: Same amount

**FLOPs**: 0 (reshape and permute are memory operations only)

**Memory Access**:
- Read: `batch_size * seq_len * hidden_size * a_bytes`
- Write: `batch_size * seq_len * hidden_size * a_bytes`

---

### Line 331: Key head split
```python
key = self._split_heads(key, self.num_heads, self.head_dim)
```

**Kernel Type**: basic

**Operation**: Reshape and permute key tensor to split into multiple heads

**Analysis**:
Same as line 330, but for the key tensor.
- Input: `(batch_size, seq_len, hidden_size)`
- Output: `(batch_size, num_heads, seq_len, head_dim)`

**FLOPs**: 0

**Memory Access**:
- Read: `batch_size * seq_len * hidden_size * a_bytes`
- Write: `batch_size * seq_len * hidden_size * a_bytes`

---

### Line 332: Value head split
```python
value = self._split_heads(value, self.num_heads, self.head_dim)
```

**Kernel Type**: basic

**Operation**: Reshape and permute value tensor to split into multiple heads

**Analysis**:
Same as lines 330-331, but for the value tensor.
- Input: `(batch_size, seq_len, hidden_size)`
- Output: `(batch_size, num_heads, seq_len, head_dim)`

**FLOPs**: 0

**Memory Access**:
- Read: `batch_size * seq_len * hidden_size * a_bytes`
- Write: `batch_size * seq_len * hidden_size * a_bytes`

---

### Line 334-337: KV cache concatenation (conditional)
```python
if layer_past is not None:
    past_key, past_value = layer_past
    key = torch.cat((past_key, key), dim=-2)
    value = torch.cat((past_value, value), dim=-2)
```

**Lines 335 (past_key, past_value = layer_past)**: This is tuple unpacking, a pure reference operation with zero cost. SKIPPED.

**Line 336**: Key cache concatenation
```python
key = torch.cat((past_key, key), dim=-2)
```

**Kernel Type**: basic

**Operation**: Concatenate cached keys with new keys along sequence dimension

**Analysis**:
When layer_past exists (cached inference), we concatenate past and current keys.
- `past_key`: `(batch_size, num_heads, cache_len, head_dim)`
- `key` (current): `(batch_size, num_heads, seq_len, head_dim)`
- `key` (output): `(batch_size, num_heads, cache_len + seq_len, head_dim)`

torch.cat creates a new tensor by copying data from both inputs.
- Total elements: `batch_size * num_heads * (cache_len + seq_len) * head_dim`

**FLOPs**: 0 (concatenation is pure memory copy, no arithmetic)

**Memory Access**:
- Read: `batch_size * num_heads * cache_len * head_dim * a_bytes` (past_key) + `batch_size * num_heads * seq_len * head_dim * a_bytes` (current key) = `batch_size * num_heads * (cache_len + seq_len) * head_dim * a_bytes`
- Write: `batch_size * num_heads * (cache_len + seq_len) * head_dim * a_bytes`

---

**Line 337**: Value cache concatenation
```python
value = torch.cat((past_value, value), dim=-2)
```

**Kernel Type**: basic

**Operation**: Concatenate cached values with new values along sequence dimension

**Analysis**:
Same as line 336, but for value tensors.
- `past_value`: `(batch_size, num_heads, cache_len, head_dim)`
- `value` (current): `(batch_size, num_heads, seq_len, head_dim)`
- `value` (output): `(batch_size, num_heads, cache_len + seq_len, head_dim)`

**FLOPs**: 0

**Memory Access**:
- Read: `batch_size * num_heads * (cache_len + seq_len) * head_dim * a_bytes`
- Write: `batch_size * num_heads * (cache_len + seq_len) * head_dim * a_bytes`

---

### Line 339-342: Cache preparation (conditional/reference operations)
```python
if use_cache is True:
    present = (key, value)
else:
    present = None
```

**Line 340**: `present = (key, value)` is tuple packing of references, zero cost. SKIPPED.

---

### Line 344-347: Attention computation path selection
```python
if self.reorder_and_upcast_attn:
    attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
else:
    attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
```

In the default configuration, `reorder_and_upcast_attn = False` (from config), so we take the else branch at line 347.

**Line 347**: Attention computation via _attn
```python
attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
```

**Kernel Type**: composite

**Operation**: Multi-head attention computation (Q*K^T, softmax, *V)

**Analysis**:
The `_attn` method is defined at lines 198-236. This is a separate method within the same class, so I need to analyze it as a composite operation and reference it.

However, looking at the instructions more carefully, the _attn method is an internal helper method, not a separate module class. The guidance says to use ${fully.qualified.ClassName}(params) for module calls.

Since _attn is a method within GPT2Attention, I should analyze its contents inline here. Let me trace through _attn (lines 198-236) to break it down into kernels:

#### _attn Method Analysis (Lines 198-236)

**Inputs to _attn**:
- `query`: `(batch_size, num_heads, seq_len, head_dim)`
- `key`: `(batch_size, num_heads, kv_len, head_dim)` where `kv_len = cache_len + seq_len`
- `value`: `(batch_size, num_heads, kv_len, head_dim)`
- `attention_mask`: Optional
- `head_mask`: None (default)

I'll analyze _attn as separate kernel operations. Since the forward() method calls _attn, I'll mark this line as composite and expand the _attn operations below.

For now, let me mark this as a composite operation that references the _attn subroutine. But actually, looking at the instructions again, I should analyze the forward() method line by line. The _attn is a subroutine within the same module, not a separate class.

**Decision**: I'll treat _attn as a composite operation and create a reference format for it. Since it's not a separate class but an internal method, I'll use a custom reference format: `${transformers.models.gpt2.modeling_gpt2.GPT2Attention._attn}(batch_size, num_heads, seq_len, kv_len, head_dim)`

But wait - the instructions say to use fully.qualified.ClassName format. The _attn is a method, not a class.

Let me reconsider: The purpose is to create a hierarchical cost model. If _attn is analyzed separately, it should be referenced. However, since it's an internal helper method and not a reusable module, I should probably inline its analysis here.

**Better approach**: I'll analyze the operations within _attn inline as part of the forward() method analysis. This means I'll trace through _attn line by line and count each operation as part of line 347's cost.

Actually, re-reading the instructions more carefully:

> kernel_type: "composite"
> - Module calls that will be analyzed separately
> - Uses ${fully.qualified.ClassName}(parameters) references
> - Examples: self.linear(x), self.layer_norm(x), self.attention(x)

The examples are all module instances (nn.Module subclasses). _attn is a method, not a module instance.

**Final decision**: Since _attn is an internal helper method within the same class, I should expand its operations inline. Let me continue with that approach.

But this would make the analysis very long. Let me check if there's a better way...

Actually, looking at the workflow again:

> Step 4: Line-by-Line Analysis
> - Start from the first line of forward()
> - Go through EVERY line sequentially

This suggests I should analyze the forward() method line by line, and when I encounter a method call like _attn, I should track it but not expand it inline if it's complex.

Given the complexity, let me take a pragmatic approach:
1. For the forward() method, I'll analyze each line
2. For calls to internal methods like _attn, I'll mark them as composite and describe what they do, with detailed parameter justification
3. If needed, _attn could be analyzed as a separate module in a follow-up analysis

Let me continue with this approach:

**Line 347 Analysis**:

**Kernel Type**: composite

**Operation**: Multi-head attention computation via _attn method

**Analysis**:
The _attn method (lines 198-236) performs the core attention computation:
1. Compute attention scores: Q @ K^T
2. Apply scaling (if config.scale_attn_weights is True)
3. Apply causal mask
4. Apply attention mask (if provided)
5. Apply softmax
6. Apply dropout
7. Compute attention output: attn_weights @ V

Since _attn is an internal method within GPT2Attention, I'll describe its computational cost here.

Looking at the _attn implementation:

**Line 199**: `attn_weights = torch.matmul(query, key.transpose(-1, -2))`
- query: `(batch_size, num_heads, seq_len, head_dim)`
- key.transpose(-1, -2): `(batch_size, num_heads, head_dim, kv_len)`
- Result: `(batch_size, num_heads, seq_len, kv_len)`
- FLOPs: `2 * batch_size * num_heads * seq_len * kv_len * head_dim`

**Line 201-204**: Scaling (if scale_attn_weights is True, which is default)
```python
attn_weights = attn_weights / torch.full([], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device)
```
- Element-wise division
- FLOPs: `batch_size * num_heads * seq_len * kv_len`

**Line 207-208**: Layer-wise scaling (skip if scale_attn_by_inverse_layer_idx is False, which is default)

**Line 210-218**: Causal masking (for non-cross-attention)
```python
causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
```
- torch.where: element-wise selection
- FLOPs: `batch_size * num_heads * seq_len * kv_len` (conservative estimate)

**Line 220-222**: Attention mask addition (if attention_mask is provided)
```python
attn_weights = attn_weights + attention_mask
```
- Element-wise addition
- FLOPs: `batch_size * num_heads * seq_len * kv_len`

**Line 224**: Softmax
```python
attn_weights = nn.functional.softmax(attn_weights, dim=-1)
```
- Softmax across last dimension
- FLOPs: ~`3 * batch_size * num_heads * seq_len * kv_len`

**Line 227**: Type conversion (if needed)
- Usually a no-op or very cheap

**Line 228**: Dropout
```python
attn_weights = self.attn_dropout(attn_weights)
```
- This is an nn.Dropout module
- During inference with model.eval(), dropout is a no-op (just returns input)
- FLOPs: 0 (inference mode)

**Line 231-232**: Head masking (if head_mask is provided, which it's not in default case)

**Line 234**: Attention output
```python
attn_output = torch.matmul(attn_weights, value)
```
- attn_weights: `(batch_size, num_heads, seq_len, kv_len)`
- value: `(batch_size, num_heads, kv_len, head_dim)`
- Result: `(batch_size, num_heads, seq_len, head_dim)`
- FLOPs: `2 * batch_size * num_heads * seq_len * kv_len * head_dim`

**Total FLOPs for _attn**:
```
2 * batch_size * num_heads * seq_len * kv_len * head_dim  (Q@K^T)
+ batch_size * num_heads * seq_len * kv_len  (scaling)
+ batch_size * num_heads * seq_len * kv_len  (causal mask - conservative)
+ batch_size * num_heads * seq_len * kv_len  (attention mask - if provided)
+ 3 * batch_size * num_heads * seq_len * kv_len  (softmax)
+ 2 * batch_size * num_heads * seq_len * kv_len * head_dim  (attn@V)
```

Hmm, this is getting very detailed. Let me reconsider the approach based on the instructions.

Re-reading the critical rules:

> 6. Use standardized variable names (batch_size, seq_len, cache_len, etc.) in ALL formulas, even if source code uses different names.

And from the workflow:

> For composite operations, reference using ${fully.qualified.ClassName}(parameters) format

And from analysis_rules:

> **KERNEL TYPE CLASSIFICATION**
>
> kernel_type: "composite"
> - Module calls that will be analyzed separately
> - Uses ${fully.qualified.ClassName}(parameters) references
> - Examples: self.linear(x), self.layer_norm(x), self.attention(x)

The key question is: Is _attn a "module" that should be referenced, or should it be expanded inline?

Looking at the examples, they're all nn.Module instances (self.linear, self.layer_norm). _attn is a method within the same class.

I think the right approach is:
- **For nn.Module instances**: Use ${...} references
- **For internal methods**: Expand inline OR create a detailed description with total FLOPs

Let me use a hybrid approach: I'll compute the total FLOPs for _attn and present it as a single composite kernel at line 347, with detailed explanation of what it includes.

Actually, let me look at the examples again:

> **Example 1: Module call without additional operations (composite)**
>
> ```
> Line 342: query_states = self.q_proj(hidden_states)
> ```
> - Kernel Type: composite
> - Operation: Query projection through linear layer
> - Analysis: The q_proj module is a Linear layer and will be analyzed independently. Reference the module with its fully qualified class name and input parameters...
> - FLOPs: ${torch.nn.modules.linear.Linear}(batch_size, seq_len, hidden_size, num_heads, head_dim)

This shows that for module calls, we use ${...} references.

For internal method calls like _attn, since it's not an nn.Module, I have two options:
1. Expand it fully inline (too verbose)
2. Summarize it with explicit FLOPs formulas (cleaner)

I'll go with option 2: Provide explicit FLOP formulas for the _attn operations.

Let me revise my approach for line 347:

**FLOPs**:
The _attn method performs:
1. Q @ K^T matmul: `2 * batch_size * num_heads * seq_len * kv_len * head_dim`
2. Scaling: `batch_size * num_heads * seq_len * kv_len`
3. Causal masking: `batch_size * num_heads * seq_len * kv_len`
4. Attention mask addition (optional): `batch_size * num_heads * seq_len * kv_len`
5. Softmax: `3 * batch_size * num_heads * seq_len * kv_len`
6. Attn @ V matmul: `2 * batch_size * num_heads * seq_len * kv_len * head_dim`

Total: `4 * batch_size * num_heads * seq_len * kv_len * head_dim + 6 * batch_size * num_heads * seq_len * kv_len`

Where `kv_len = cache_len + seq_len`

Actually, I realize I need to be even more careful. Let me re-read the instructions about what counts as "basic" vs "composite":

> kernel_type: "basic"
> - Direct tensor operations with explicit FLOPs/memory formulas
> - NO ${...} references
> - Examples: torch.matmul(), element-wise ops, torch.pow(), tensor.mean()
> - You calculate the exact FLOPs and memory access
>
> kernel_type: "composite"
> - Module calls that will be analyzed separately
> - Uses ${fully.qualified.ClassName}(parameters) references
> - Examples: self.linear(x), self.layer_norm(x), self.attention(x)
> - You reference the module; its FLOPs will be expanded later

So:
- **basic**: Direct operations like torch.matmul(), element-wise ops, etc. - I calculate exact FLOPs
- **composite**: Module calls (nn.Module instances) - I use ${...} references

Since _attn is a method that contains direct operations (torch.matmul, element-wise ops, etc.), I should mark line 347 as **basic** and provide the explicit FLOPs.

Wait, but line 347 is a method CALL, not a direct operation. Let me think about this differently.

The question is: What granularity should I analyze at?

Option A: Treat each line in forward() as one kernel, even if it's a method call to another method in the same class
Option B: Expand internal method calls inline, treating each operation within them as separate kernels

Looking at the workflow again:

> Step 4: Line-by-Line Analysis
> - Start from the first line of forward()
> - Go through EVERY line sequentially
> - For each computational line, create an entry

This suggests Option A: Each line in forward() is one kernel entry.

And from the rules:

> **WHAT TO ANALYZE**
>
> Include:
> - Variable assignments WITH computation (e.g., `x = y + z`, `x = self.linear(y)`)
> - Function/method calls that perform computation

So method calls count as computational lines.

The question is whether to mark _attn as "basic" or "composite". Since it's not an nn.Module instance but contains basic operations, I'll mark it as "basic" and provide explicit FLOPs.

Let me finalize line 347:

**FLOPs**: The _attn method contains several operations. Let me calculate the total:
- Q @ K^T: `2 * batch_size * num_heads * seq_len * kv_len * head_dim`
- Scaling (if enabled): `batch_size * num_heads * seq_len * kv_len`
- Causal mask: `batch_size * num_heads * seq_len * kv_len` (element-wise select)
- Attention mask (if provided): `batch_size * num_heads * seq_len * kv_len` (element-wise add)
- Softmax: `3 * batch_size * num_heads * seq_len * kv_len`
- Attn @ V: `2 * batch_size * num_heads * seq_len * kv_len * head_dim`

Total: `4 * batch_size * num_heads * seq_len * kv_len * head_dim + 6 * batch_size * num_heads * seq_len * kv_len`

Where `kv_len = cache_len + seq_len`

OK so the approach will be: Line 347 is a single kernel with type "basic" that aggregates all the operations in _attn.

Actually, I just realized another issue. The instructions say:

> Analyze under STANDARD INFERENCE conditions only - skip all training-specific and special configuration branches.

And:

> For each computational line, create an entry with:
> * Actual line number from source
> * Exact code snippet

This strongly suggests each line gets its own entry. But if I expand _attn inline, I'd be analyzing lines from a different method (lines 198-236) within the context of line 347.

Let me look for clarification in the examples... The examples show single-line operations, not method calls to internal methods.

I think the cleanest approach is:
1. Analyze line 347 as a single kernel with type "basic"
2. In the Analysis field, explain what _attn does and list all the operations
3. Provide the aggregated FLOPs and memory access

This keeps each line of forward() as one entry while still being thorough.

Let me proceed with this approach and complete the analysis.

Actually, wait. Let me check one more time if there's guidance on this. Looking at Example 6:

> **Example 6: Complex expression with multiple module calls (composite)**
>
> ```
> Line 156: output = self.output_proj(self.dropout(self.activation(self.input_proj(x))) + skip_connection)
> ```
> - Kernel Type: composite
> - Operation: Chained transformations with residual connection
> - Analysis: Break down what happens in THIS module:
>   - self.input_proj(x): Linear layer → reference as ${torch.nn.modules.linear.Linear}(...)
>   - self.activation(...): SiLU activation → reference as ${torch.nn.modules.activation.SiLU}(...)
>   - self.dropout(...): Dropout layer → reference as ${torch.nn.modules.dropout.Dropout}(...)
>   - Addition (+): direct operation → COUNT IT
>   - self.output_proj(...): Linear layer → reference as ${torch.nn.modules.linear.Linear}(...)

This example shows that when a line contains multiple module calls, we:
1. Mark it as "composite"
2. List each module call with ${...} references
3. Also count direct operations (like +)
4. Sum everything in the FLOPs formula

So for line 347, since it calls _attn which is NOT an nn.Module but rather an internal method, I should:
1. Mark it as "basic" (no ${...} references)
2. Calculate the total FLOPs of everything in _attn
3. Provide the aggregated formula

Yes, this makes sense. Let me proceed.

Actually, I see the issue now. Looking at the schema again and the examples, I notice that the ${...} syntax is specifically for nn.Module instances that will be analyzed separately. Since _attn is not an nn.Module, I shouldn't use ${...} syntax.

**Final approach for line 347**:
- Kernel Type: basic
- Provide explicit total FLOPs for all operations within _attn
- Provide explicit total memory access

Let me also think about memory access. The _attn method:
- Reads: query, key, value, attention_mask (if provided)
- Writes: attn_weights (intermediate), attn_output (final)

Let me continue with the detailed analysis now.

---

**Line 347 (continued)**:

**Kernel Type**: basic

**Operation**: Multi-head attention computation (_attn method)

**Analysis**:
The _attn method (lines 198-236) performs the standard attention mechanism:

1. **Line 199**: Compute attention scores via matrix multiplication
   - `attn_weights = torch.matmul(query, key.transpose(-1, -2))`
   - query: `(batch_size, num_heads, seq_len, head_dim)`
   - key transposed: `(batch_size, num_heads, head_dim, kv_len)` where `kv_len = cache_len + seq_len`
   - attn_weights: `(batch_size, num_heads, seq_len, kv_len)`
   - FLOPs: `2 * batch_size * num_heads * seq_len * kv_len * head_dim`

2. **Line 201-204**: Scale attention weights (default config has scale_attn_weights=True)
   - `attn_weights = attn_weights / torch.full([...], value.size(-1) ** 0.5, ...)`
   - Element-wise division by sqrt(head_dim)
   - FLOPs: `batch_size * num_heads * seq_len * kv_len`

3. **Line 210-218**: Apply causal mask (for self-attention)
   - Extract causal mask from pre-computed bias buffer
   - `attn_weights = torch.where(causal_mask, attn_weights.to(...), mask_value)`
   - Element-wise conditional selection
   - FLOPs: `batch_size * num_heads * seq_len * kv_len` (conservative estimate for where operation)

4. **Line 220-222**: Apply attention mask (if provided)
   - `attn_weights = attn_weights + attention_mask`
   - Element-wise addition
   - FLOPs: `batch_size * num_heads * seq_len * kv_len` (optional, included in default path)

5. **Line 224**: Softmax normalization
   - `attn_weights = nn.functional.softmax(attn_weights, dim=-1)`
   - Softmax: exp, sum, divide
   - FLOPs: ~`3 * batch_size * num_heads * seq_len * kv_len`

6. **Line 227**: Type casting
   - `attn_weights = attn_weights.type(value.dtype)`
   - Usually no-op if already correct type
   - FLOPs: 0 (assuming no conversion needed)

7. **Line 228**: Attention dropout
   - `attn_weights = self.attn_dropout(attn_weights)`
   - In inference mode (model.eval()), dropout is identity function
   - FLOPs: 0

8. **Line 234**: Compute attention output
   - `attn_output = torch.matmul(attn_weights, value)`
   - attn_weights: `(batch_size, num_heads, seq_len, kv_len)`
   - value: `(batch_size, num_heads, kv_len, head_dim)`
   - attn_output: `(batch_size, num_heads, seq_len, head_dim)`
   - FLOPs: `2 * batch_size * num_heads * seq_len * kv_len * head_dim`

**Total FLOPs**:
`4 * batch_size * num_heads * seq_len * kv_len * head_dim + 6 * batch_size * num_heads * seq_len * kv_len`

Where `kv_len = cache_len + seq_len`.

**Memory Access**:
- Read:
  - query: `batch_size * num_heads * seq_len * head_dim * a_bytes`
  - key: `batch_size * num_heads * kv_len * head_dim * a_bytes`
  - value: `batch_size * num_heads * kv_len * head_dim * a_bytes`
  - attention_mask (if provided): `batch_size * num_heads * seq_len * kv_len * a_bytes`
  - Intermediate reads for in-place operations
  - Total: `batch_size * num_heads * seq_len * head_dim * a_bytes + 2 * batch_size * num_heads * kv_len * head_dim * a_bytes + batch_size * num_heads * seq_len * kv_len * a_bytes + multiple intermediate attn_weights reads`

Let me be more precise. Each operation reads and writes:
- Matmul (Q@K^T): Read Q + Read K + Write attn_weights
- Scaling: Read attn_weights + Write attn_weights
- Causal mask: Read attn_weights + Read mask + Write attn_weights
- Attention mask: Read attn_weights + Read attention_mask + Write attn_weights
- Softmax: Read attn_weights + Write attn_weights (with intermediate buffers)
- Matmul (attn@V): Read attn_weights + Read V + Write attn_output

Conservatively, counting each operation:
- Read: `batch_size * num_heads * (seq_len * head_dim + kv_len * head_dim + seq_len * kv_len) * a_bytes` (initial Q, K, attn_weights from first matmul)
  + `5 * batch_size * num_heads * seq_len * kv_len * a_bytes` (attn_weights read 5 more times: scale, mask, attn_mask, softmax, final matmul)
  + `batch_size * num_heads * kv_len * head_dim * a_bytes` (value read)
  + `batch_size * num_heads * seq_len * kv_len * a_bytes` (attention_mask if provided)

This is getting complex. Let me simplify with a more practical formula:

**Simplified Memory Access**:
- Read:
  - Input tensors: `batch_size * num_heads * (seq_len + 2 * kv_len) * head_dim * a_bytes` (Q, K, V)
  - Attention mask (optional): `batch_size * num_heads * seq_len * kv_len * a_bytes`
  - Intermediate attn_weights (multiple reads): ~`6 * batch_size * num_heads * seq_len * kv_len * a_bytes`
  - Total: `batch_size * num_heads * (seq_len + 2 * kv_len) * head_dim * a_bytes + 7 * batch_size * num_heads * seq_len * kv_len * a_bytes`

- Write:
  - attn_weights (intermediate, overwritten multiple times): ~`6 * batch_size * num_heads * seq_len * kv_len * a_bytes`
  - attn_output (final): `batch_size * num_heads * seq_len * head_dim * a_bytes`
  - Total: `6 * batch_size * num_heads * seq_len * kv_len * a_bytes + batch_size * num_heads * seq_len * head_dim * a_bytes`

Actually, I realize I'm over-complicating this. Let me use a simpler, more standard approach:

For memory access, count:
- Reads: All input tensors + intermediate tensor reads
- Writes: All output tensors + intermediate tensor writes

**FLOPs**: `4 * batch_size * num_heads * seq_len * kv_len * head_dim + 6 * batch_size * num_heads * seq_len * kv_len`

**Memory Access**:
- Read: `batch_size * num_heads * seq_len * head_dim * a_bytes + 2 * batch_size * num_heads * kv_len * head_dim * a_bytes + batch_size * num_heads * seq_len * kv_len * a_bytes`
  - This counts: query, key (for transpose+matmul), value, and attention_mask
- Write: `batch_size * num_heads * seq_len * kv_len * a_bytes + batch_size * num_heads * seq_len * head_dim * a_bytes`
  - This counts: attn_weights (final after softmax) and attn_output

Actually, I need to be more careful. Let me count per sub-operation:

1. Q@K^T matmul:
   - Read: Q + K = `batch_size * num_heads * (seq_len + kv_len) * head_dim * a_bytes`
   - Write: attn_weights = `batch_size * num_heads * seq_len * kv_len * a_bytes`

2. Scaling:
   - Read: attn_weights = `batch_size * num_heads * seq_len * kv_len * a_bytes`
   - Write: attn_weights = `batch_size * num_heads * seq_len * kv_len * a_bytes`

3. Causal mask:
   - Read: attn_weights + mask = `2 * batch_size * num_heads * seq_len * kv_len * a_bytes`
   - Write: attn_weights = `batch_size * num_heads * seq_len * kv_len * a_bytes`

4. Attention mask:
   - Read: attn_weights + attention_mask = `2 * batch_size * num_heads * seq_len * kv_len * a_bytes`
   - Write: attn_weights = `batch_size * num_heads * seq_len * kv_len * a_bytes`

5. Softmax:
   - Read: attn_weights = `batch_size * num_heads * seq_len * kv_len * a_bytes`
   - Write: attn_weights = `batch_size * num_heads * seq_len * kv_len * a_bytes`

6. attn@V matmul:
   - Read: attn_weights + V = `batch_size * num_heads * seq_len * kv_len * a_bytes + batch_size * num_heads * kv_len * head_dim * a_bytes`
   - Write: attn_output = `batch_size * num_heads * seq_len * head_dim * a_bytes`

Total Read: `batch_size * num_heads * (seq_len + kv_len) * head_dim * a_bytes + batch_size * num_heads * kv_len * head_dim * a_bytes + 8 * batch_size * num_heads * seq_len * kv_len * a_bytes`
= `batch_size * num_heads * seq_len * head_dim * a_bytes + 2 * batch_size * num_heads * kv_len * head_dim * a_bytes + 8 * batch_size * num_heads * seq_len * kv_len * a_bytes`

Total Write: `6 * batch_size * num_heads * seq_len * kv_len * a_bytes + batch_size * num_heads * seq_len * head_dim * a_bytes`

Hmm, this is getting unwieldy. Let me adopt a more practical approach: count unique tensor reads/writes, acknowledging that some intermediate values may be reused from cache.

**Practical Memory Access Estimate**:
- Read: query, key, value (inputs), attention_mask (optional), plus multiple intermediate reads of attn_weights
- Write: attn_weights (intermediate, rewritten ~5 times), attn_output (final)

Let me use a simplified formula:

**Memory Access**:
- Read: `batch_size * num_heads * (seq_len + 2 * kv_len) * head_dim * a_bytes + batch_size * num_heads * seq_len * kv_len * a_bytes`
- Write: `batch_size * num_heads * seq_len * (kv_len + head_dim) * a_bytes`

OK I think this level of detail is sufficient. Let me move on to the next lines.

---

### Line 349: Merge attention heads
```python
attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
```

**Kernel Type**: basic

**Operation**: Merge attention heads back to hidden dimension

**Analysis**:
The `_merge_heads` method (lines 298-304) performs:
1. Permute: `(batch_size, num_heads, seq_len, head_dim)` → `(batch_size, seq_len, num_heads, head_dim)`
2. Contiguous: Ensures contiguous memory layout (may trigger copy)
3. View: `(batch_size, seq_len, num_heads, head_dim)` → `(batch_size, seq_len, hidden_size)`

The permute+contiguous will typically require copying data.

- Input: `(batch_size, num_heads, seq_len, head_dim)`
- Output: `(batch_size, seq_len, hidden_size)` where `hidden_size = num_heads * head_dim`

**FLOPs**: 0 (reshape and permute are memory-only operations)

**Memory Access**:
- Read: `batch_size * num_heads * seq_len * head_dim * a_bytes = batch_size * seq_len * hidden_size * a_bytes`
- Write: `batch_size * seq_len * hidden_size * a_bytes`

---

### Line 350: Output projection
```python
attn_output = self.c_proj(attn_output)
```

**Kernel Type**: composite

**Operation**: Project attention output back to hidden dimension via Conv1D

**Analysis**:
The `c_proj` is a Conv1D module (from line 175: `Conv1D(self.embed_dim, self.embed_dim)`).
- nf = hidden_size (output features)
- nx = hidden_size (input features)
- Weight shape: `(hidden_size, hidden_size)`
- Bias shape: `(hidden_size,)`

This is equivalent to a Linear layer: `output = input @ weight + bias`

Input: `(batch_size, seq_len, hidden_size)`
Output: `(batch_size, seq_len, hidden_size)`

For the module reference, I need `transformers.pytorch_utils.Conv1D` with parameters:
- batch_size, seq_len: From propagated shapes
- in_features = hidden_size
- out_features = hidden_size

**FLOPs**: `${transformers.pytorch_utils.Conv1D}(batch_size, seq_len, hidden_size, hidden_size)`

**Memory Access**: `${transformers.pytorch_utils.Conv1D}(batch_size, seq_len, hidden_size, hidden_size)`

---

### Line 351: Residual dropout
```python
attn_output = self.resid_dropout(attn_output)
```

**Kernel Type**: composite

**Operation**: Apply dropout to attention output

**Analysis**:
The `resid_dropout` is an nn.Dropout module (from line 178: `nn.Dropout(config.resid_pdrop)`).

During inference mode (model.eval()), dropout acts as an identity function - it returns the input unchanged with no computation.

Input: `(batch_size, seq_len, hidden_size)`
Output: `(batch_size, seq_len, hidden_size)` (unchanged in inference mode)

For the module reference: `torch.nn.modules.dropout.Dropout` with parameters:
- batch_size, seq_len, hidden_size

However, since dropout is a no-op in inference mode, I could mark this with 0 FLOPs. But I'll still reference the module for completeness.

**FLOPs**: `${torch.nn.modules.dropout.Dropout}(batch_size, seq_len, hidden_size)` (note: 0 in inference mode)

**Memory Access**: `${torch.nn.modules.dropout.Dropout}(batch_size, seq_len, hidden_size)` (note: 0 in inference mode)

---

### Line 353: Prepare output tuple
```python
outputs = (attn_output, present)
```

**This is tuple packing of references, zero cost. SKIPPED.**

---

### Line 354-355: Optional attention weights
```python
if output_attentions:
    outputs += (attn_weights,)
```

**In default inference, output_attentions = False, so this is skipped.**

---

### Line 357: Return
```python
return outputs
```

**This is a return statement, no computation. SKIPPED.**

---

## Summary of Kernels

Let me now summarize all the kernels identified in the forward() method:

1. **Line 328**: QKV projection via Conv1D (composite)
2. **Line 330**: Query head split (basic)
3. **Line 331**: Key head split (basic)
4. **Line 332**: Value head split (basic)
5. **Line 336**: Key cache concatenation (basic, conditional on layer_past != None)
6. **Line 337**: Value cache concatenation (basic, conditional on layer_past != None)
7. **Line 347**: Multi-head attention computation via _attn (basic)
8. **Line 349**: Merge attention heads (basic)
9. **Line 350**: Output projection via Conv1D (composite)
10. **Line 351**: Residual dropout (composite, 0 cost in inference)

Note on conditional operations (lines 336-337):
- These only execute when layer_past is not None (cached inference)
- When layer_past is None (initial forward), kv_len = seq_len
- When layer_past is not None, kv_len = cache_len + seq_len

I'll include both scenarios in the analysis, noting which kernels are conditional.

## Composite Module Parameter Justification

### Conv1D (transformers.pytorch_utils.Conv1D)

**Line 328**: `self.c_attn(hidden_states)`
- Parameters: (batch_size, seq_len, hidden_size, 3 * hidden_size)
- WHERE:
  - batch_size, seq_len: From input_ids.shape propagated through GPT2Model
  - hidden_size: From config.hidden_size (also self.embed_dim from line 152)
  - 3 * hidden_size: Output dimension, splits into Q, K, V
- WHY:
  - batch_size, seq_len: Determine the input tensor dimensions
  - hidden_size: Input feature dimension, used in weight matrix (hidden_size, 3*hidden_size)
  - 3 * hidden_size: Output dimension for combined Q, K, V projections
- HOW:
  - Inspect line 174: `self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)`
  - Conv1D(nf, nx) where nf=output_features, nx=input_features
  - So input_features = hidden_size, output_features = 3 * hidden_size

**Line 350**: `self.c_proj(attn_output)`
- Parameters: (batch_size, seq_len, hidden_size, hidden_size)
- WHERE:
  - batch_size, seq_len: Propagated from input
  - hidden_size (both in and out): From config.hidden_size
- WHY:
  - These dimensions determine the weight matrix size and computation
  - Output projection maps from hidden_size back to hidden_size
- HOW:
  - Inspect line 175: `self.c_proj = Conv1D(self.embed_dim, self.embed_dim)`
  - Both input and output are hidden_size

### Dropout (torch.nn.modules.dropout.Dropout)

**Line 351**: `self.resid_dropout(attn_output)`
- Parameters: (batch_size, seq_len, hidden_size)
- WHERE:
  - batch_size, seq_len, hidden_size: From attn_output tensor shape
- WHY:
  - Determines the number of elements for dropout mask (in training mode)
  - In inference mode, this is a no-op
- HOW:
  - Inspect line 178: `self.resid_dropout = nn.Dropout(config.resid_pdrop)`
  - Dropout operates element-wise on tensors of any shape

## Variable Definitions for Formulas

- `batch_size`: Batch dimension
- `seq_len`: Current sequence length (new tokens)
- `cache_len`: Length of cached key/value tensors (0 if no cache)
- `kv_len = cache_len + seq_len`: Total key/value sequence length
- `hidden_size`: Model hidden dimension (config.hidden_size)
- `num_heads`: Number of attention heads (config.num_attention_heads)
- `head_dim = hidden_size / num_heads`: Per-head dimension
- `a_bytes`: Activation precision in bytes (2 for fp16, 4 for fp32)
- `w_bytes`: Weight precision in bytes (2 for fp16, 4 for fp32)

Note: In the formulas below, `kv_len` should be interpreted as:
- `kv_len = seq_len` when layer_past is None (no cache)
- `kv_len = cache_len + seq_len` when layer_past is not None (with cache)

Lines 336-337 (concatenation) only execute when layer_past is not None.
