"""Universal LM Performance Benchmarking Tool

Supports multi-device (GPU, TPU, CPU, MPS) benchmarking with intelligent
OOM handling and adaptive sweep strategies.
"""

import argparse
import gc
import json
import time
from typing import Callable, List, Dict, Any, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_device_and_sync() -> Tuple[torch.device, Callable, str]:
    """Detect available device and return (device, sync_fn, device_type)

    Priority: TPU > CUDA > MPS > CPU

    Returns:
        device: torch.device object
        sync_fn: synchronization function (e.g., torch.cuda.synchronize)
        device_type: human-readable string ("TPU", "GPU", "MPS", "CPU")
    """
    # Try TPU first
    try:
        import torch_xla
        device = torch_xla.device()
        sync_fn = torch_xla.sync
        device_type = "TPU"
        print(f"Using TPU: {device}")
        return device, sync_fn, device_type
    except ImportError:
        pass

    # Try CUDA
    if torch.cuda.is_available():
        device = torch.device("cuda")
        sync_fn = torch.cuda.synchronize
        device_type = "GPU"
        print(f"Using GPU: {device}")
        return device, sync_fn, device_type

    # Try MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        # MPS synchronization requires PyTorch 2.0+
        if hasattr(torch.mps, "synchronize"):
            sync_fn = torch.mps.synchronize
        else:
            print("Error: MPS device detected but torch.mps.synchronize() not available.")
            print("Please upgrade to PyTorch 2.0+ for accurate benchmarking on MPS.")
            import sys
            sys.exit(1)
        device_type = "MPS"
        print(f"Using MPS: {device}")
        return device, sync_fn, device_type

    # Fallback to CPU
    device = torch.device("cpu")
    sync_fn = lambda: None  # No synchronization needed for CPU
    device_type = "CPU"
    print(f"Using CPU: {device}")
    return device, sync_fn, device_type


def clear_memory(device: torch.device) -> None:
    """Clear GPU/TPU memory after OOM"""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    # TPU: torch_xla handles memory differently, gc.collect() should suffice


def is_oom_error(e: Exception) -> bool:
    """Check if exception is an out-of-memory error across devices"""
    msg = str(e).lower()
    return any(keyword in msg for keyword in [
        "out of memory", "oom", "alloc", "memory",
        "invalid buffer size"  # MPS-specific OOM error
    ])


def generate_sweep_params(max_val: int, min_val: int = 1) -> List[int]:
    """Generate sweep parameters: powers of 2 plus 1.5x intermediate values

    Example for max_val=64, min_val=1:
    [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]

    This gives ~2 points per octave (doubling), providing more uniform
    data distribution on a log scale for better analysis.

    Args:
        max_val: Maximum value to generate
        min_val: Minimum value to start from

    Returns:
        Sorted list of sweep parameters
    """
    result = []
    val = min_val
    while val <= max_val:
        result.append(val)
        mid = int(val * 1.5)
        if mid < val * 2 and mid <= max_val:
            result.append(mid)
        val *= 2
    return sorted(set(result))  # Remove duplicates and sort


def load_model(
    model_id: str,
    device: torch.device,
    device_type: str,
    cache_dir: Optional[str]
) -> Tuple[torch.nn.Module, Any]:
    """Load model with appropriate device handling

    Args:
        model_id: HuggingFace model ID
        device: Target device
        device_type: Device type string
        cache_dir: Optional cache directory for model files

    Returns:
        (model, tokenizer) tuple
    """
    print(f"Loading model: {model_id}")
    kwargs = {
        "low_cpu_mem_usage": True,  # Keep RAM footprint low during load
        "device_map": None,  # Disable accelerate device mapping
    }
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).to(device)
        model.eval()  # Set to evaluation mode (disable dropout, etc.)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if is_oom_error(e):
            print(f"\n✗ Error: Out of memory while loading model '{model_id}' to {device_type}")
            print(f"Suggestions:")
            print(f"  1. Free up device memory (kill other processes)")
            print(f"  2. Use a smaller model")
            print(f"  3. Try CPU mode: --device cpu (slower but no memory limit)")
            import sys
            sys.exit(1)
        else:
            raise

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, cache_dir=cache_dir if cache_dir else None
    )
    print(f"Model loaded on {device_type}")

    # Print memory usage after loading
    if device.type == "mps":
        mem_gb = torch.mps.current_allocated_memory() / (1024 ** 3)
        print(f"  MPS memory after loading: {mem_gb:.2f} GB")
    elif device.type == "cuda":
        mem_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
        print(f"  CUDA memory after loading: {mem_gb:.2f} GB")

    return model, tokenizer


def run_single_config(
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
    sync_fn: Callable,
    batch_size: int,
    warmup_rounds: int,
    test_rounds: int,
    seq_len: Optional[int] = None,
    cache_len: Optional[int] = None
) -> List[float]:
    """Run a single benchmark configuration for prefill or decode

    Handles all logic for both modes:
    - Prefill mode (seq_len): Benchmark prefill with seq_len tokens
    - Decode mode (cache_len): Prefill to generate cache, then benchmark decode

    Args:
        model: The language model
        tokenizer: The tokenizer
        device: Target device
        sync_fn: Synchronization function
        batch_size: Batch size to test
        warmup_rounds: Number of warmup iterations
        test_rounds: Number of timed iterations
        seq_len: Sequence length for prefill mode (mutually exclusive with cache_len)
        cache_len: Cache length for decode mode (mutually exclusive with seq_len)

    Returns:
        List of timing measurements (in seconds)
    """
    if cache_len is not None:
        # === DECODE MODE ===
        # Step 1: Prefill to generate KV cache
        input_text = tokenizer.decode([tokenizer.eos_token_id] * cache_len)
        batch_input_texts = [input_text] * batch_size
        inputs = tokenizer(batch_input_texts, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs, use_cache=True)
            past_key_values = outputs.past_key_values

        # Step 2: Extract next token (matching tests/opt-125m.py line 69)
        next_token_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        # Step 3: Warmup decode
        for _ in range(warmup_rounds):
            with torch.no_grad():
                _ = model(
                    input_ids=next_token_id,
                    past_key_values=past_key_values,
                    use_cache=True
                )
            sync_fn()  # Force execution (prevent XLA dead code elimination)

        # Step 4: Time decode
        times = []
        for _ in range(test_rounds):
            sync_fn()
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(
                    input_ids=next_token_id,
                    past_key_values=past_key_values,
                    use_cache=True
                )
            sync_fn()
            end = time.perf_counter()
            times.append(end - start)

        # Print memory usage after decode
        if device.type == "mps":
            mem_gb = torch.mps.current_allocated_memory() / (1024 ** 3)
            print(f"  MPS memory after decode: {mem_gb:.2f} GB")
        elif device.type == "cuda":
            mem_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
            print(f"  CUDA memory after decode: {mem_gb:.2f} GB")

    else:
        # === PREFILL MODE ===
        input_text = tokenizer.decode([tokenizer.eos_token_id] * seq_len)
        batch_input_texts = [input_text] * batch_size
        inputs = tokenizer(batch_input_texts, return_tensors="pt").to(device)

        # Warmup
        for _ in range(warmup_rounds):
            with torch.no_grad():
                _ = model(**inputs, use_cache=True)
            sync_fn()  # Force execution (prevent XLA dead code elimination)

        # Timed runs
        times = []
        for _ in range(test_rounds):
            sync_fn()
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(**inputs, use_cache=True)
            sync_fn()
            end = time.perf_counter()
            times.append(end - start)

        # Print memory usage after prefill
        if device.type == "mps":
            mem_gb = torch.mps.current_allocated_memory() / (1024 ** 3)
            print(f"  MPS memory after prefill: {mem_gb:.2f} GB")
        elif device.type == "cuda":
            mem_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
            print(f"  CUDA memory after prefill: {mem_gb:.2f} GB")

    return times


def run_prefill_benchmark(
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
    sync_fn: Callable,
    batch_sizes: List[int],
    seq_lens: List[int],
    warmup_rounds: int,
    test_rounds: int
) -> List[Dict[str, Any]]:
    """Run prefill phase benchmark with OOM handling

    OOM Handling Strategy:
    - batch_sizes: iterate small to large (outer loop)
    - seq_lens: iterate small to large (inner loop)

    When OOM occurs at (batch_size=B, seq_len=S):
    - Break inner loop (skip remaining seq_lens for this batch_size)
    - Continue testing next batch_size

    Args:
        model: The language model
        tokenizer: The tokenizer
        device: Target device
        sync_fn: Synchronization function
        batch_sizes: List of batch sizes to test
        seq_lens: List of sequence lengths to test
        warmup_rounds: Number of warmup iterations
        test_rounds: Number of timed iterations

    Returns:
        List of result dictionaries
    """
    results = []

    print("\n=== Prefill Phase Benchmark ===")

    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            try:
                print(f"Testing batch_size={batch_size}, seq_len={seq_len}...", end=" ")
                times = run_single_config(
                    model, tokenizer, device, sync_fn,
                    batch_size, warmup_rounds, test_rounds,
                    seq_len=seq_len
                )
                mean_time = sum(times) / len(times)
                std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5

                result = {
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "times": times,
                    "mean": mean_time,
                    "std": std_time
                }
                results.append(result)
                # Print individual times for debugging
                times_str = ", ".join([f"{t:.6f}s" for t in times])
                print(f"✓ times=[{times_str}], mean={mean_time:.6f}s, std={std_time:.6f}s")
                clear_memory(device)  # Clear memory after each test for consistency

            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if is_oom_error(e):
                    print(f"✗ OOM")
                    clear_memory(device)
                    break  # Skip remaining seq_lens for this batch_size
                else:
                    raise

    return results


def run_decode_benchmark(
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
    sync_fn: Callable,
    batch_sizes: List[int],
    cache_lens: List[int],
    warmup_rounds: int,
    test_rounds: int
) -> List[Dict[str, Any]]:
    """Run decode phase benchmark with OOM handling

    Decode phase: 1 token input with KV cache from prefill

    OOM Handling Strategy:
    - batch_sizes: iterate small to large (outer loop)
    - cache_lens: iterate small to large (inner loop)

    When OOM occurs at (batch_size=B, cache_len=C):
    - Break inner loop (skip remaining cache_lens for this batch_size)
    - Continue testing next batch_size

    Args:
        model: The language model
        tokenizer: The tokenizer
        device: Target device
        sync_fn: Synchronization function
        batch_sizes: List of batch sizes to test
        cache_lens: List of KV cache lengths to test
        warmup_rounds: Number of warmup iterations
        test_rounds: Number of timed iterations

    Returns:
        List of result dictionaries
    """
    results = []

    print("\n=== Decode Phase Benchmark ===")

    for batch_size in batch_sizes:
        for cache_len in cache_lens:
            try:
                print(f"Testing batch_size={batch_size}, cache_len={cache_len}...", end=" ")
                times = run_single_config(
                    model, tokenizer, device, sync_fn,
                    batch_size, warmup_rounds, test_rounds,
                    cache_len=cache_len
                )
                mean_time = sum(times) / len(times)
                std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5

                result = {
                    "batch_size": batch_size,
                    "cache_len": cache_len,
                    "times": times,
                    "mean": mean_time,
                    "std": std_time
                }
                results.append(result)
                # Print individual times for debugging
                times_str = ", ".join([f"{t:.6f}s" for t in times])
                print(f"✓ times=[{times_str}], mean={mean_time:.6f}s, std={std_time:.6f}s")
                clear_memory(device)  # Clear memory after each test for consistency

            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if is_oom_error(e):
                    print(f"✗ OOM")
                    clear_memory(device)
                    break  # Skip remaining cache_lens for this batch_size
                else:
                    raise

    return results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Universal LM Performance Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default model on CPU
  python tests/lm_perf.py

  # Benchmark specific model with custom parameters
  python tests/lm_perf.py --model_id facebook/opt-125m --max_batch_size 128 --max_seq_len 512

  # Run only prefill phase
  python tests/lm_perf.py --phase prefill

  # Run only decode phase
  python tests/lm_perf.py --phase decode

  # Save results to file
  python tests/lm_perf.py --model_id gpt2 --output results.json

  # Use custom cache directory
  python tests/lm_perf.py --cache_dir /workspace/cache --model_id facebook/opt-125m
        """
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/opt-125m",
        help="HuggingFace model ID (default: facebook/opt-125m)"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Model cache directory (default: HuggingFace default)"
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=1024,
        help="Maximum batch size to test (default: 1024)"
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=1024,
        help="Maximum sequence length to test (default: 1024)"
    )
    parser.add_argument(
        "--warmup_rounds",
        type=int,
        default=2,
        help="Number of warmup rounds (default: 2)"
    )
    parser.add_argument(
        "--test_rounds",
        type=int,
        default=3,
        help="Number of test rounds (default: 3)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON format)"
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=["both", "prefill", "decode"],
        default="both",
        help="Which phase(s) to benchmark: both, prefill, or decode (default: both)"
    )
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    # Detect device
    device, sync_fn, device_type = get_device_and_sync()

    # Load model
    model, tokenizer = load_model(
        args.model_id, device, device_type, args.cache_dir
    )

    # Generate sweep parameters
    batch_sizes = generate_sweep_params(args.max_batch_size, min_val=1)
    seq_lens = generate_sweep_params(args.max_seq_len, min_val=32)

    print(f"\nSweep parameters:")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  Sequence lengths: {seq_lens}")
    print(f"  Warmup rounds: {args.warmup_rounds}")
    print(f"  Test rounds: {args.test_rounds}")
    print(f"  Phase: {args.phase}")

    # Run benchmarks based on selected phase
    prefill_results = None
    decode_results = None

    if args.phase in ["both", "prefill"]:
        prefill_results = run_prefill_benchmark(
            model, tokenizer, device, sync_fn,
            batch_sizes, seq_lens,
            args.warmup_rounds, args.test_rounds
        )

    if args.phase in ["both", "decode"]:
        decode_results = run_decode_benchmark(
            model, tokenizer, device, sync_fn,
            batch_sizes, seq_lens,  # Use same seq_lens as cache_lens
            args.warmup_rounds, args.test_rounds
        )

    # Prepare output
    results = {
        "model_id": args.model_id,
        "device_type": device_type,
        "device": str(device),
        "phase": args.phase,
        "sweep_params": {
            "batch_sizes": batch_sizes,
            "seq_lens": seq_lens
        },
        "config": {
            "warmup_rounds": args.warmup_rounds,
            "test_rounds": args.test_rounds
        }
    }

    # Add phase results only if they were run
    if prefill_results is not None:
        results["prefill_results"] = prefill_results
    if decode_results is not None:
        results["decode_results"] = decode_results

    # Save to file if specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {args.output}")
    else:
        print("\n=== Summary ===")
        if prefill_results is not None:
            print(f"Prefill: {len(prefill_results)} configurations tested")
        if decode_results is not None:
            print(f"Decode: {len(decode_results)} configurations tested")

    print("\n✓ Benchmark complete!")


if __name__ == "__main__":
    main()
