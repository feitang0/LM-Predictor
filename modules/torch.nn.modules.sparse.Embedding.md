# Analysis of torch.nn.modules.sparse.Embedding

## Source Code Location

**File**: `pytorch/torch/nn/modules/sparse.py`
**Class**: `Embedding` (lines 15-268)
**Forward Method**: Lines 191-200

## Forward Method Source Code

```python
def forward(self, input: Tensor) -> Tensor:
    return F.embedding(
        input,
        self.weight,
        self.padding_idx,
        self.max_norm,
        self.norm_type,
        self.scale_grad_by_freq,
        self.sparse,
    )
```

The forward method delegates to `F.embedding()` which is located in `pytorch/torch/nn/functional.py` at lines 2432-2546.

## F.embedding Implementation Analysis

Looking at `pytorch/torch/nn/functional.py` lines 2432-2546:

```python
def embedding(
    input: Tensor,
    weight: Tensor,
    padding_idx: Optional[int] = None,
    max_norm: Optional[float] = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> Tensor:
    # ... documentation ...
    if has_torch_function_variadic(input, weight):
        return handle_torch_function(...)
    if padding_idx is not None:
        if padding_idx > 0:
            assert padding_idx < weight.size(0), ...
        elif padding_idx < 0:
            assert padding_idx >= -weight.size(0), ...
            padding_idx = weight.size(0) + padding_idx
    else:
        padding_idx = -1
    if max_norm is not None:
        input = input.contiguous()
        _no_grad_embedding_renorm_(weight, input, max_norm, norm_type)
    return torch.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)
```

## Inference Configuration

For standard inference analysis, we assume:
- **training = False** (no gradient computation)
- **max_norm = None** (no renormalization; this is a training-time feature)
- **padding_idx** may or may not be set (doesn't affect computation, just gradients)
- **scale_grad_by_freq = False** (default, training-only feature)
- **sparse = False** (default, affects gradient computation only)

Therefore, the inference path is:
1. Check if padding_idx needs adjustment (lines 2522-2533) - **pure control flow, no computation**
2. Skip max_norm branch (line 2534-2545) - **not executed in default inference**
3. Execute `torch.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)` (line 2546)

## Understanding torch.embedding Operation

The `torch.embedding()` function is a native PyTorch operation (implemented in C++/CUDA). Based on the documentation and behavior:

**Input shapes**:
- `input`: Shape `(*)` where `*` represents arbitrary dimensions. Common cases:
  - 1D: `(seq_len,)` - single sequence
  - 2D: `(batch_size, seq_len)` - batched sequences
  - General: `(d1, d2, ..., dn)` - n-dimensional tensor of indices
- `weight`: Shape `(num_embeddings, embedding_dim)` - the embedding lookup table

**Output shape**:
- `(*, embedding_dim)` where `*` matches the input shape
- Example: input `(batch_size, seq_len)` → output `(batch_size, seq_len, embedding_dim)`

**Operation**: Embedding lookup is essentially a **gather/index operation**:
- For each index `i` in the input tensor, retrieve row `weight[i]`
- This is a memory lookup operation, NOT a matrix multiplication
- No arithmetic operations (FLOPs = 0 for the lookup itself)

## Variable Definitions

Let's define standardized variables for embedding operations:

- **num_indices**: Total number of indices in the input tensor
  - For 1D input: `num_indices = seq_len`
  - For 2D input: `num_indices = batch_size * seq_len`
  - General: `num_indices = product of all input dimensions`
- **num_embeddings**: Size of vocabulary (number of rows in weight matrix)
- **embedding_dim**: Dimension of each embedding vector (number of columns in weight matrix)
- **w_bytes**: Bytes per weight element (typically 2 for fp16, 4 for fp32)
- **a_bytes**: Bytes per activation element (typically 2 for fp16, 4 for fp32)

## Line-by-Line Analysis of Embedding.forward()

### Line 191: `def forward(self, input: Tensor) -> Tensor:`
- This is the function signature, no computation.

### Line 192-200: Return statement calling F.embedding
```python
return F.embedding(
    input,
    self.weight,
    self.padding_idx,
    self.max_norm,
    self.norm_type,
    self.scale_grad_by_freq,
    self.sparse,
)
```

This is a single composite operation that delegates to the functional API. The actual implementation in `F.embedding` (line 2546) calls `torch.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)`.

## Detailed Analysis of torch.embedding Kernel

Since `torch.embedding` is a built-in operation, I need to analyze what it does:

**Operation**: Embedding lookup via indexing
- For each index in `input`, fetch the corresponding row from `weight`
- This is purely a **memory operation** - gathering data from the weight matrix

**Computation breakdown**:

1. **FLOPs**:
   - Embedding lookup is a pure gather/index operation
   - **NO arithmetic operations** are performed
   - FLOPs = **0**

2. **Memory Access**:
   - **Read from weight matrix**: For each index, we read one embedding vector
     - Number of reads: `num_indices` vectors of size `embedding_dim`
     - Total read from weights: `num_indices * embedding_dim * w_bytes`
   - **Read from input**: We need to read all indices
     - Input is integer tensor (LongTensor), typically 8 bytes per index
     - Total read from input: `num_indices * 8` bytes
   - **Write to output**: Create output tensor with gathered embeddings
     - Total write: `num_indices * embedding_dim * a_bytes`

**Total memory access**:
- **Read**: `num_indices * embedding_dim * w_bytes + num_indices * 8`
- **Write**: `num_indices * embedding_dim * a_bytes`

Note: The input index read (`num_indices * 8`) is typically negligible compared to the embedding data transfer, but included for completeness.

## Kernel Summary

The Embedding module has exactly **ONE** computational kernel:

### Kernel 1: Embedding Lookup (lines 192-200)

**Type**: Composite (delegates to torch.embedding)

**Operation**: Embedding table lookup

**Shapes**:
- Input indices: `(batch_size, seq_len)` or more generally `(*)`
- Weight: `(num_embeddings, embedding_dim)`
- Output: `(batch_size, seq_len, embedding_dim)` or `(*, embedding_dim)`

**Analysis**:
This kernel performs an embedding lookup operation by indexing into the weight matrix. For standard inference with default settings (no max_norm, no special padding handling during forward pass), this reduces to a single call to `torch.embedding()` at line 2546 of functional.py.

The operation is a pure memory gather operation:
- Each of the `num_indices = batch_size * seq_len` indices in the input selects one row from the weight matrix
- Each row has `embedding_dim` elements
- No arithmetic computation is performed (it's indexing, not matrix multiplication)

**Parameter Justification**:
- **num_indices**: Derived from input tensor shape. For typical language model input of shape `(batch_size, seq_len)`, we have `num_indices = batch_size * seq_len`. For general n-dimensional input, `num_indices` is the product of all dimensions.
- **embedding_dim**: Comes from `self.embedding_dim` set during module initialization (see line 152 in __init__)
- **w_bytes**: Weight precision in bytes, typically 2 for fp16 or 4 for fp32
- **a_bytes**: Activation precision in bytes, typically 2 for fp16 or 4 for fp32

**FLOPs**: 0 (pure memory lookup, no arithmetic)

**Memory Access**:
- **Read**:
  - Weight matrix: `num_indices * embedding_dim * w_bytes`
  - Input indices: `num_indices * 8` (8 bytes per LongTensor element)
  - **Total**: `num_indices * embedding_dim * w_bytes + num_indices * 8`
  - Simplified: `num_indices * (embedding_dim * w_bytes + 8)`

- **Write**:
  - Output embeddings: `num_indices * embedding_dim * a_bytes`

## Special Cases and Branches NOT Analyzed

The following code paths are NOT executed in standard inference and are therefore excluded:

1. **max_norm renormalization** (lines 2534-2545 in functional.py):
   - Only executed when `max_norm is not None`
   - This is primarily used during training to constrain embedding norms
   - Skipped in standard inference

2. **has_torch_function_variadic check** (lines 2510-2521):
   - Custom tensor type handling
   - Not relevant for standard PyTorch tensors
   - Skipped

3. **Gradient-related parameters** (`scale_grad_by_freq`, `sparse`):
   - Only affect backward pass
   - No impact on forward pass computation
   - Skipped in inference analysis

## Module Reference Format

Since `torch.embedding` is a built-in native operation (not a Python module), I reference it directly in the composite kernel. The operation could be further decomposed into lower-level memory operations, but `torch.embedding` is typically considered an atomic operation in neural network profiling.

If we were to use module reference syntax, it would be:
`${torch.embedding}(num_indices, embedding_dim)`

However, since this is a native operation implemented in C++/CUDA (not a Python nn.Module), I provide the direct FLOPs and memory formulas in the analysis.

## Conclusion

The `torch.nn.modules.sparse.Embedding` module is extremely simple from a computational perspective:
- **Single operation**: Embedding lookup
- **Zero FLOPs**: Pure memory gather operation
- **Memory-bound**: Performance depends entirely on memory bandwidth
- **Memory access**: Dominated by reading embeddings from the weight matrix

The module is typically memory-bandwidth limited rather than compute-limited, making it one of the most straightforward operations to analyze in neural networks.
