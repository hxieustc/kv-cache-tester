# single_prompt_tester.py

Measure baseline cache performance with simple cold start vs. cached prompt comparisons. This is the simplest tool and provides quick insights into whether caching is working at all.

## What it does

- Tests at doubling context sizes (1K, 2K, 4K, 8K, ...) or specific sizes
- For each size, sends a unique prompt (cold start) then repeats the same prompt (100% cached)
- Measures TTFT speedup from caching
- Optionally sends multiple concurrent prompts and/or repeats cached prompts

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `--api-endpoint` | Your inference server URL | **required** |
| `--context-sizes` | Specific context sizes to test (e.g., `8000 32000 64000`). Overrides min/max-tokens. | - |
| `--min-tokens` | Minimum context size (used with doubling if --context-sizes not specified) | 1000 |
| `--max-tokens` | Maximum context size (used with doubling if --context-sizes not specified) | 128000 |
| `--output-tokens` | Tokens to generate per request | 256 |
| `--num-iterations` | Iterations per context size | 5 |
| `--concurrent-prompts`, `-n` | Send N prompts simultaneously | 1 |
| `--cached-repeats`, `-r` | Repeat cached prompt N times | 1 |
| `--bust-requests`, `-b` | Fixed number of fill requests sent between cold and warm to evict the test prompt's KV from L1 GPU cache (servers with prefix caching). Mutually exclusive with `--auto-bust`. | 0 |
| `--auto-bust` | Auto-compute the bust count per context size from L1 cache geometry. Requires `--num-gpu-blocks` and `--block-size`. Mutually exclusive with `--bust-requests`. | false |
| `--num-gpu-blocks` | L1 GPU KV cache capacity in blocks. Required when `--auto-bust` is set. | - |
| `--block-size` | Tokens per GPU KV cache block. Required when `--auto-bust` is set. | - |
| `--bust-safety-factor` | Multiplier on the auto-computed bust count for headroom over output tokens, reserved blocks, and non-strict-LRU eviction. | 1.1 |
| `--metrics-endpoint` | Server endpoint for Prometheus metrics scraping (e.g. `http://host:8000`). Scrapes `/prometheus/metrics` (TRT-LLM) or `/metrics` (vLLM) before/after each warm request to isolate KV onboard/offload stats. | - |
| `--output-dir` | Where to save results | ./single_prompt_output |
| `--tokenizer` | HuggingFace tokenizer ID | Qwen/Qwen2.5-Coder-32B-Instruct |
| `--seed` | Random seed for reproducibility | - |
| `--verbose` | Enable debug logging | false |
| `--brief` | Agent-friendly minimal output | false |
| `--no-color` | Disable colored output (for light terminals) | false |

## Example Usage

```bash
# Basic test with defaults (doubling from 1K to 128K)
uv run python single_prompt_tester.py \
    --api-endpoint http://localhost:8000

# Test specific context sizes
uv run python single_prompt_tester.py \
    --api-endpoint http://localhost:8000 \
    --context-sizes 8000 32000 64000 \
    --output-dir specific_sizes

# Test with concurrent prompts
uv run python single_prompt_tester.py \
    --api-endpoint http://localhost:8000 \
    --context-sizes 32000 \
    --concurrent-prompts 10 \
    --output-dir concurrent_test

# Test with multiple cached repeats
uv run python single_prompt_tester.py \
    --api-endpoint http://localhost:8000 \
    --context-sizes 32000 \
    --cached-repeats 5 \
    --output-dir repeat_test

# Test up to 256K context with more iterations (doubling mode)
uv run python single_prompt_tester.py \
    --api-endpoint http://localhost:8000 \
    --min-tokens 2000 \
    --max-tokens 256000 \
    --num-iterations 10 \
    --output-dir baseline_results

# Test with custom tokenizer
uv run python single_prompt_tester.py \
    --api-endpoint http://localhost:8000 \
    --tokenizer meta-llama/Llama-3.3-70B-Instruct \
    --output-dir llama_test

# Auto-bust: compute bust count per ISL from L1 cache geometry
uv run python single_prompt_tester.py \
    --api-endpoint http://localhost:8000 \
    --context-sizes 1024 8192 65536 \
    --auto-bust \
    --num-gpu-blocks 1024 \
    --block-size 64 \
    --output-dir auto_bust_test
```

## Bust phase (L1 GPU cache eviction)

When the inference server has prefix caching enabled, the test prompt's KV blocks
stay in the L1 GPU cache after the cold request, so the "warm" path measures a
GPU-cache hit instead of a host→GPU onboard. To force the warm path to onboard
from host (the scenario most useful for offload benchmarking), the tester can
send N concurrent unique fill requests between cold and warm to evict the test
prompt's blocks.

You can set the bust count two ways:

- **Fixed (`--bust-requests N`)** — same N for every context size. Easy, but
  small ISLs need many bust requests while large ISLs need few, so a single
  value is rarely right across a sweep.
- **Auto (`--auto-bust`)** — compute the bust count per context size from L1
  cache geometry:

  ```
  blocks_per_request = ceil(ISL / block_size)
  bust_count         = ceil(num_gpu_blocks * safety_factor / blocks_per_request)
  ```

  The safety factor (default 1.1) adds headroom for output-token blocks,
  reserved blocks, and eviction policies that aren't strict LRU.

The two modes are mutually exclusive. Pair either with `--metrics-endpoint` to
verify post-hoc that an onboard transfer actually happened on the warm request
— if the bust count is too low, the tester fails fast with a clear error.

## Output Files

| File | Description |
|------|-------------|
| `single_prompt_performance.html` | Interactive graphs showing TTFT, TTLT, and speedup |
| `summary_table.html` | Table with statistical breakdown |
| `single_prompt_results_*.csv` | Raw data |
| `index.html` | Dashboard linking all results |

## When to use

- Quick smoke test to verify caching is working
- Understanding baseline performance of a single memory tier
- Measuring maximum achievable speedup at various context sizes
- Testing server behavior under concurrent load (with `-n`)

## Limitations

- Only tests 0% and 100% cache hit rates
- Fixed working set (all prompts cached)
