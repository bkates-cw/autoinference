# Auto-Inference Optimization

Optimize **Qwen3.5-35B-A3B** (MoE: 35B total, 3B active) serving throughput on vLLM while maintaining output quality.

## The One File You Edit

`serve.py` — contains the `vllm_args` dict that configures vLLM. Modify parameters, run the script, read the scores.

## How to Run

```bash
python serve.py
```

This starts vLLM with your config, runs accuracy benchmarks (lm-eval-harness), format canary, and latency sweep (guidellm), prints all metrics, and shuts down.

## Metrics

The script prints key: value lines you can parse:

```
gsm8k_em: 0.8500        # GSM8K exact match accuracy
format_valid_rate: 1.0   # format canary pass rate (must be 1.0)
p95_ttft_ms: 245.3       # p95 time to first token
request_throughput: 12.5  # requests/sec
```

**Primary target:** maximize `request_throughput`
**Guardrails:** `gsm8k_em` within 5% of baseline, `format_valid_rate` = 1.0

## Commit Rule

Commit when:
1. Request throughput improved over baseline
2. p95 TTFT did not regress badly
3. GSM8K accuracy within 5% of baseline
4. Format canary = 100%

Otherwise revert and try different parameters.

## Parameters You Can Tune (in `vllm_args`)

| Parameter | What it does | Try these values |
|-----------|-------------|-----------------|
| `max-num-batched-tokens` | Token budget per step. Highest leverage. | 2048, 4096, 8192, 16384 |
| `max-num-seqs` | Max concurrent sequences in batch | 8, 16, 32, 64, 128 |
| `gpu-memory-utilization` | GPU memory fraction for KV cache | 0.80, 0.85, 0.90, 0.95 |
| `kv-cache-dtype` | KV cache precision. FP8 saves memory. | auto, fp8 |
| `quantization` | Weight precision | None, fp8, awq, gptq |
| `max-model-len` | Context cap. Lower = more concurrency. | 2048, 4096, 8192 |
| `enable-prefix-caching` | Reuse KV blocks for shared prefixes | True, False |
| `tensor-parallel-size` | GPU sharding (if multi-GPU) | 1, 2, 4 |
| `speculative-model` | Enable speculative decoding | "[ngram]" |
| `num-speculative-tokens` | Tokens to speculate | 2, 4, 8 |

## Strategy Tips

- Change 1-2 parameters at a time to attribute improvements.
- `max-num-batched-tokens` and `max-num-seqs` are usually highest leverage — start there.
- `enable-prefix-caching` is nearly free — try it early.
- `kv-cache-dtype: fp8` saves memory with minimal quality impact.
- Quantization gives big throughput gains but watch accuracy guardrails.
- Speculative decoding helps most at low QPS / memory-bound workloads.
- Lower `max-model-len` if you don't need long context — frees KV cache.
- Add comments to `serve.py` explaining WHY you chose certain values.

## What You Cannot Touch

- `prepare.py` — setup script
- `eval/` — benchmark and guardrail code
- `program.md` — this file

## Repo Structure

```
serve.py       ← YOU EDIT THIS (the one file)
program.md     ← you are reading this
prepare.py     ← one-time setup (model + dataset cache)
eval/          ← benchmark code (do not touch)
results/       ← experiment outputs
```
