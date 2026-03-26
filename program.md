# Auto-Inference Optimization

Optimize **Qwen3.5-35B-A3B** (MoE: 35B total, 3B active) serving throughput on vLLM while maintaining output quality.

## What You Can Modify

`serve.py` — you have **full control** over the serving side of this file. You are not limited to tweaking `vllm_args`. You can restructure how the vLLM server is started and configured. Some ideas beyond config tuning:

- **Prompt optimization**: Restructure few-shot prompts to maximize prefix sharing across requests
- **KV cache warming**: Pre-populate the cache with common prompt prefixes before benchmarking starts
- **Custom server launch**: Change how vLLM is started — environment variables, engine args, API server flags
- **Dynamic adaptation**: Profile GPU utilization and adjust settings at runtime
- **Chunked prefill tuning**: Enable and configure chunked prefill for better interleaving

**Do NOT modify the evaluation/benchmarking code** — the eval functions, metric printing, and benchmark harness must stay unchanged so results remain comparable across experiments.

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

## Logging Results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 6 columns:

```
commit	throughput	p95_ttft	gsm8k_em	status	description
```

1. git commit hash (short, 7 chars)
2. request_throughput (e.g. 12.50) — use 0.00 for crashes
3. p95_ttft_ms (e.g. 245.3) — use 0.0 for crashes
4. gsm8k_em (e.g. 0.8500) — use 0.0000 for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	throughput	p95_ttft	gsm8k_em	status	description
a1b2c3d	10.50	312.0	0.8500	keep	baseline
b2c3d4e	12.50	245.3	0.8500	keep	increase max-num-batched-tokens to 16384
c3d4e5f	11.00	450.1	0.8000	discard	fp8 quantization degraded accuracy
d4e5f6g	0.00	0.0	0.0000	crash	speculative decoding OOM
```

## Experiment Naming

Prefix all `EXPERIMENT_ID` values with a short model identifier derived from the `MODEL` variable in `serve.py` (e.g., `Qwen/Qwen3-4B` → `qwen3-4b`, `Qwen/Qwen3.5-35B-A3B` → `qwen3.5-35b-a3b`). Examples:
- Smoke test: `qwen3-4b-smoke`
- Experiments: `qwen3-4b-exp-01`, `qwen3-4b-exp-02`, etc.

## The Experiment Loop

The experiment runs on a dedicated branch named after the model (e.g. `autoinference/qwen3-4b-mar25`). Read the `MODEL` variable from `serve.py` to construct the branch name.

LOOP FOREVER:

1. Look at the git state: the current branch/commit you're on.
2. Tune `serve.py` — modify `vllm_args`, restructure serving code, or try deeper optimizations.
3. git commit.
4. Run the experiment: `uv run serve.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context).
5. Read out the results: `grep "^gsm8k_em:\|^format_valid_rate:\|^p95_ttft_ms:\|^request_throughput:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the error and attempt a fix.
7. Record the results in the TSV and commit it alongside any code changes.
8. If throughput improved AND guardrails pass, you "advance" the branch, keeping the git commit.
9. If throughput is equal or worse, or guardrails fail, `git reset --hard HEAD~1` to revert.

## Commit Rule

Keep a commit (status: `keep`) when ALL are true:
1. Request throughput improved over baseline
2. p95 TTFT did not regress badly
3. GSM8K accuracy within 5% of baseline
4. Format canary = 100%

Otherwise revert `serve.py` to the previous commit and try different parameters.

**Crashes**: If a run crashes (OOM, vLLM startup failure, etc.), use your judgment. If it's easy to fix (typo, bad parameter combo), fix and re-run. If the idea is fundamentally broken, log as `crash`, revert, and move on.

**NEVER STOP**: Once the loop has begun, do NOT pause to ask the human if you should continue. The human might be away and expects you to continue working *indefinitely* until manually stopped. If you run out of ideas, think harder — try combining previous near-misses, try more radical parameter combos, re-read the strategy tips.

## Optimization Surface

You can optimize at multiple levels. Start with config tuning, then go deeper.

### Level 1: vLLM Config (`vllm_args` dict)

| Parameter | What it does | Try these values |
|-----------|-------------|-----------------|
| `max-num-batched-tokens` | Token budget per step. Highest leverage. | 2048, 4096, 8192, 16384 |
| `max-num-seqs` | Max concurrent sequences in batch | 8, 16, 32, 64, 128 |
| `gpu-memory-utilization` | GPU memory fraction for KV cache | 0.80, 0.85, 0.90, 0.95 |
| `kv-cache-dtype` | KV cache precision. FP8 saves memory. | auto, fp8 |
| `quantization` | Weight precision | None, fp8, awq, gptq |
| `max-model-len` | Context cap. Lower = more concurrency. | 2048, 4096, 8192 |
| `enable-prefix-caching` | Reuse KV blocks for shared prefixes | True, False |
| `enable-chunked-prefill` | Interleave prefill and decode | True, False |
| `performance-mode` | Scheduler optimization target | throughput, balanced, interactivity |
| `speculative-config` | Speculative decoding (JSON) | ngram method with various token counts |

### Level 2: Serving Code Changes

Once config tuning plateaus, modify `serve.py` more aggressively:
- Restructure prompts to share common prefixes (prefix caching wins)
- Add KV cache warm-up phase before benchmarking
- Tune vLLM environment variables (e.g., `VLLM_ATTENTION_BACKEND`, `VLLM_USE_V1`)
- Custom engine args beyond what's in `vllm_args`

### Level 3: VRAM Minimization

After throughput plateaus (3 consecutive batches with no improvement), switch to minimizing VRAM:
1. Take the best throughput config
2. Binary search `gpu-memory-utilization` downward (0.90 → 0.85 → 0.80 → ...) until it crashes
3. Report the minimum VRAM that maintains ≥95% of best throughput
4. This tells you the cheapest GPU that could serve this model

## Strategy Tips

- Start with Level 1 config tuning — change 1-2 parameters at a time to attribute improvements.
- `max-num-batched-tokens` and `max-num-seqs` are usually highest leverage — start there.
- When config tuning stops improving, move to Level 2 code changes.
- After throughput plateaus, move to Level 3 VRAM minimization.
- Add comments to `serve.py` explaining WHY you chose certain values.

## W&B Experiment Overview Run

In addition to the per-experiment W&B runs created by `serve.py` (via `instructions.md`), maintain a **single overview W&B run** that aggregates all experiment results into one chart.

At the start of the experiment loop, create the overview run:

```python
import wandb, os
overview_run = wandb.init(
    entity=os.environ.get("WANDB_ENTITY", None),
    project=os.environ.get("WANDB_PROJECT", "research"),
    name="experiment-loop-overview",
    notes="Aggregated metrics across all experiments",
)
```

After collecting results from each experiment, log them to this run:

```python
overview_run.log({
    "request_throughput": throughput,
    "p95_ttft_ms": p95_ttft,
    "gsm8k_em": gsm8k_em,
})
```

This gives you one continuous chart showing how metrics evolve across experiments, while the individual per-experiment runs remain available for drill-down.

**How to do this from the shell**: Use a small Python script or inline snippet to log to the overview run after parsing each experiment's logs. Save the run ID so you can resume it across batches.

## What You Cannot Touch

- `prepare.py` — setup script
- `eval/` — benchmark and guardrail code
- `program.md` — this file

## Repo Structure

```
serve.py       ← YOU EDIT THIS (serving code — full control)
program.md     ← you are reading this (do not touch)
prepare.py     ← one-time setup (do not touch)
eval/          ← benchmark/eval code (do not touch)
results/       ← experiment outputs
```
