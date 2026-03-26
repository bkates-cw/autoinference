# autoinference

Autonomous vLLM inference optimization on CoreWeave H100s. An AI agent tunes serving configs, runs benchmarks, and iterates — designed to be driven by [autolab](https://github.com/kaiii66/autolab).

## Results (CoreWeave AI Hackathon, March 2026)

### Qwen3.5-35B-A3B (MoE: 35B total, 3B active)

| Metric | Baseline | Best | Improvement |
|---|---|---|---|
| Throughput | 5.25 req/s | **7.12 req/s** | **+35.6%** |
| GSM8K accuracy | 0.92 | 0.90 | Within guardrail |
| Format validation | 0.875 | 1.0 | Fixed |

**175 experiments** over ~15 hours. Key optimizations:
- `max-num-seqs=160` with `gpu-memory-utilization=0.94` (+2.6% from higher concurrency)
- `VLLM_FUSED_MOE_CHUNK_SIZE=32768` (+0.3% from fewer MoE kernel launches)
- Chat completions ThinkStripper to fix Qwen3 `<think>` token leakage in format canary
- Log suppression: `VLLM_LOGGING_LEVEL=WARNING` + `uvicorn-log-level=critical` (+0.3 req/s)
- `NVTE_ALLOW_NONDETERMINISTIC_ALGO=1` + `NVTE_FUSED_ATTN=1` (+0.12 req/s)

### Qwen3-4B

| Metric | Baseline | Best | Improvement |
|---|---|---|---|
| Throughput | 10.24 req/s | **12.66 req/s** | **+23.6%** |
| GSM8K accuracy | 0.86 | 0.86 | Unchanged |

**100 experiments** over ~8 hours. Key optimizations:
- `compilation-config: compile_sizes=[1,2,4,8]` (CUDA graph compilation for exact batch sizes)
- `max-num-seqs=8` (matched to benchmark concurrency)
- `max-model-len=2048` (freed KV cache for more throughput)

## How it works

The agent modifies `serve.py` (specifically the `vllm_args` dict and environment variables), then runs the full eval pipeline:

```
serve.py modified → vLLM starts → GSM8K accuracy → Format canary → guidellm throughput → Metrics printed
```

Results are logged to `results.tsv` and W&B. The agent keeps configs that improve throughput while maintaining accuracy guardrails, and reverts those that don't.

## Repo structure

```
serve.py       ← Agent edits this (vllm_args + serving code)
program.md     ← Agent instructions (optimization goals, constraints, strategy)
prepare.py     ← One-time setup (downloads model, datasets)
eval/          ← Benchmark code (do not modify)
results.tsv    ← Experiment log (created by agent)
pyproject.toml ← Python dependencies
```

## Running locally

```bash
# Install dependencies
uv sync

# One-time setup (downloads model weights + GSM8K dataset)
uv run prepare.py

# Run a single experiment
uv run serve.py

# Quick mode (smaller eval set, for testing)
uv run serve.py --quick
```

## Running with autolab

This repo is designed to be driven by [autolab](https://github.com/kaiii66/autolab):

```bash
# In the autolab repo
cp examples/autoinference/config.env config.env
# Set your WANDB_API_KEY, WANDB_ENTITY, ANTHROPIC_API_KEY

docker build -t autolab .
docker run -it \
  -v ~/.ssh/id_ed25519:/home/autolab/.ssh-mount/id_ed25519:ro \
  -v ~/.ssh/id_ed25519.pub:/home/autolab/.ssh-mount/id_ed25519.pub:ro \
  -v ~/.kube:/home/autolab/.kube:ro \
  -v $(pwd)/config.env:/home/autolab/app/config.env:ro \
  -e ANTHROPIC_API_KEY \
  -e WANDB_API_KEY \
  -e KUBECONFIG=/home/autolab/.kube/your-kubeconfig \
  autolab
```

A pre-built base image with model weights and dependencies cached is available in `examples/autoinference/base-image/` — reduces experiment startup from ~10 min to ~30 sec.

## Metrics

The eval pipeline prints key-value lines:

```
gsm8k_em: 0.9000           # GSM8K exact match accuracy
format_valid_rate: 1.0000   # Format canary pass rate (must be 1.0)
p95_ttft_ms: 86.4           # p95 time to first token
request_throughput: 7.12    # Requests/sec (primary target)
```

**Guardrails:** GSM8K within 5% of baseline, format_valid_rate = 1.0.

## Optimization surface

| Level | What to tune | Examples |
|---|---|---|
| **Config** | `vllm_args` dict | `max-num-batched-tokens`, `max-num-seqs`, `gpu-memory-utilization`, `kv-cache-dtype` |
| **Environment** | `os.environ` in `serve.py` | `VLLM_FUSED_MOE_CHUNK_SIZE`, `VLLM_ATTENTION_BACKEND`, `NVTE_*` flags |
| **Serving code** | Restructure how vLLM starts | Custom warm-up, prompt optimization, API endpoint selection |
| **VRAM** | Minimize `gpu-memory-utilization` | Binary search downward after throughput plateaus |

## Lessons learned

1. **Format canary is the #1 blocker.** Qwen3's `<think>` tokens break format validation. The fix: use `/v1/chat/completions` with `chat_template_kwargs: {"enable_thinking": False}` instead of `/v1/completions`.
2. **Cluster variance is real.** Throughput varies ~5% across runs on identical configs. Run multiple times before declaring a winner.
3. **Diminishing returns hit fast.** 80% of gains came from 3 structural changes. The remaining 100+ experiments yielded ~20% of the improvement.
4. **Log suppression matters.** vLLM's per-request INFO logging and uvicorn access logs cost measurable throughput under load (~0.3 req/s on 35B).
