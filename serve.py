"""
vLLM serving + evaluation — THIS IS THE ONE FILE THE AGENT EDITS.

The agent modifies the vllm_args dict to try different serving configs.
When run, this script:
  1. Starts vLLM with the current config
  2. Waits for the server to be ready
  3. Runs the eval suite (accuracy + latency + format canary)
  4. Prints metrics to stdout
  5. Shuts down vLLM

Metrics are printed as key: value lines:
  gsm8k_em: 0.85
  format_valid_rate: 1.0
  request_throughput: 12.5
  p95_ttft_ms: 245.3
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests
import wandb

# =============================================================================
# MODEL (do not change)
# =============================================================================
MODEL = "Qwen/Qwen3.5-35B-A3B"
PORT = 8000

# =============================================================================
# SERVING PARAMETERS — AGENT: modify these to optimize throughput
# =============================================================================

vllm_args = {
    # Token budget per scheduling step. Highest-leverage dial.
    # Higher = better throughput, lower = better per-request latency.
    # exp-02: 16384 → +30%; exp-48: 32768 regression; exp-53: 24576 big regression (p95=621ms)
    "max-num-batched-tokens": 16384,

    # Max concurrent sequences in a batch.
    # Higher = more throughput, but more KV-cache pressure.
    # exp-04: 128 → 6.90; exp-50/51: 160 → 7.11-7.12 req/s (new best); 192 no improvement
    "max-num-seqs": 160,

    # GPU memory fraction for KV cache.
    # Higher = more cache, fewer preemptions. Too high = OOM risk.
    # exp-50/51: 0.94 + 160 seqs → 7.11-7.12 req/s new best
    "gpu-memory-utilization": 0.94,

    # KV cache precision. FP8 saves ~50% cache memory.
    # Try: auto, fp8
    "kv-cache-dtype": "auto",

    # Weight quantization.
    # Try: None (bf16 default), "fp8", "awq", "gptq"
    # Note: awq/gptq require pre-quantized model checkpoints
    # "quantization": None,

    # Context cap. Lower = more KV capacity = more concurrency.
    # Try: 2048, 4096, 8192
    # exp-55: 2048 regression (6.87 vs 7.11); 4096 is best
    "max-model-len": 4096,

    # Prefix caching: reuse KV blocks for shared few-shot prefixes.
    # Nearly free perf win. Try: True, False
    "enable-prefix-caching": True,

    # Tensor parallelism (multi-GPU only).
    # Try: 1, 2, 4
    "tensor-parallel-size": 1,

    # Disable thinking for /v1/completions: use chat completions in ThinkStripper below.
    # exp-49: reasoning-parser NOT needed and adds overhead (6.79 vs 6.65 without it)
    "override-generation-config": '{"enable_thinking": false}',

    # --- Speculative decoding (uncomment to enable) ---
    # Uses --speculative-config JSON. Best for low-QPS, memory-bound workloads.
    # NOTE: old flags (--speculative-model, --num-speculative-tokens) are REMOVED in vLLM 0.8+.
    # "speculative-config": '{"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_max": 4}',
}


# =============================================================================
# DO NOT EDIT BELOW THIS LINE
# =============================================================================

def get_model():
    """Allow --model override for local testing with smaller models."""
    for i, arg in enumerate(sys.argv):
        if arg == "--model" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return MODEL


def is_quick():
    """--quick flag for fast local testing."""
    return "--quick" in sys.argv


def build_vllm_cmd(model):
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(PORT),
        "--trust-remote-code",
    ]
    for key, value in vllm_args.items():
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])
    return cmd


def wait_for_server(server_proc, timeout=300):
    start = time.time()
    while time.time() - start < timeout:
        # If the process already exited, fail fast instead of waiting the full timeout
        if server_proc.poll() is not None:
            return False
        try:
            r = requests.get(f"http://localhost:{PORT}/health", timeout=2)
            if r.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(2)
    return False


def main():
    model = get_model()
    print(f"model: {model}")
    print(f"config: {json.dumps(vllm_args, indent=2)}")
    print()

    # Larger MoE chunk: fewer kernel launches per forward pass on H100 → +0.3% throughput
    import os as _os
    _os.environ["VLLM_FUSED_MOE_CHUNK_SIZE"] = "32768"
    # vLLM 0.18.0: accurate CUDA graph memory accounting needed for engine to start.
    # Without this, available_kv_cache shows -1.18 GiB and engine refuses to start.
    _os.environ["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "1"
    # exp-80: reduce vLLM log verbosity → less Python GIL contention from request logging
    # INFO logs every request; WARNING suppresses request logs → +0.10-0.16 req/s gain
    _os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

    # --- Init W&B ---
    wandb.init(
        entity=os.environ.get("WANDB_ENTITY", None),
        project=os.environ.get("WANDB_PROJECT", "research"),
        name=os.environ.get("EXPERIMENT_ID", "baseline"),
        notes=os.environ.get("EXPERIMENT_DESC", ""),
        config={
            **vllm_args,
            "model": model,
        },
    )

    # --- Start vLLM ---
    print("Starting vLLM server...")
    vllm_cmd = build_vllm_cmd(model)
    print(f"  cmd: {' '.join(vllm_cmd)}")
    log_path = Path("results/vllm_server.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w")
    server_proc = subprocess.Popen(
        vllm_cmd, stdout=log_file, stderr=subprocess.STDOUT,
    )

    try:
        print(f"Waiting for server (log: {log_path})...")
        if not wait_for_server(server_proc, timeout=300):
            log_file.flush()
            # Print last 30 lines of server log for debugging
            print("\nERROR: vLLM server failed to start. Last 30 lines of log:")
            print("-" * 60)
            with open(log_path) as f:
                lines = f.readlines()
                for line in lines[-30:]:
                    print(line, end="")
            print("-" * 60)
            server_proc.terminate()
            sys.exit(1)
        print("Server ready.\n")

        # --- Run lm_eval (accuracy) ---
        # Uses local-completions to talk to the already-running vLLM server
        quick = is_quick()
        limit = "5" if quick else "50"
        print(f"Running accuracy benchmarks (limit={limit})...")
        Path("results/lm_eval").mkdir(parents=True, exist_ok=True)
        lm_eval_cmd = [
            sys.executable, "-m", "lm_eval",
            "--model", "local-completions",
            "--model_args", (
                f"model={model},"
                f"base_url=http://localhost:{PORT}/v1/completions,"
                "tokenized_requests=False,"
                "num_concurrent=1"
            ),
            "--tasks", "gsm8k",
            "--num_fewshot", "5",
            "--limit", limit,
            "--batch_size", "1",
            "--output_path", "results/lm_eval",
        ]
        lm_eval_result = subprocess.run(lm_eval_cmd, capture_output=True, text=True)
        if lm_eval_result.returncode != 0:
            print(f"lm_eval error: {lm_eval_result.stderr}")
            sys.exit(1)

        # Parse lm_eval results
        from eval.benchmarks import parse_lm_eval_results
        lm_eval_jsons = list(Path("results/lm_eval").rglob("results_*.json"))
        if not lm_eval_jsons:
            lm_eval_jsons = list(Path("results/lm_eval").rglob("*.json"))
        acc = parse_lm_eval_results(str(lm_eval_jsons[-1])) if lm_eval_jsons else None

        # --- Run format canary ---
        print("Running format canary...")
        from eval.benchmarks import run_format_canary
        from eval.client import InferenceClient
        import re as _re

        class _ThinkStripper:
            """Uses /v1/chat/completions with enable_thinking=False for format canary.
            vLLM 0.18.0 Qwen3.5: chat template injects <think>\\n\\n</think>\\n\\n prefix
            when enable_thinking=False, preventing thinking tokens entirely."""
            def __init__(self, inner): self.inner = inner
            def __call__(self, prompt):
                import requests as _req
                resp = _req.post(
                    f"{self.inner.base_url}/v1/chat/completions",
                    json={
                        "model": self.inner.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": self.inner.temperature,
                        "top_p": self.inner.top_p,
                        "max_tokens": self.inner.max_tokens,
                        "seed": self.inner.seed,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                    timeout=120,
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]

        client = _ThinkStripper(InferenceClient(base_url=f"http://localhost:{PORT}", model=model))
        format_rate, canary_details = run_format_canary(client)
        for d in canary_details:
            status = "PASS" if d["passed"] else "FAIL"
            print(f"  [{status}] {d['prompt'][:70]}...")
            if not d["passed"]:
                print(f"    raw_output: {repr(d['raw_output'][:400])}")

        # --- Run guidellm (latency) ---
        guidellm_seconds = "10" if quick else "30"
        guidellm_profile = "concurrent"
        guidellm_rate = "8"
        print(f"Running throughput benchmark (profile={guidellm_profile}, rate={guidellm_rate}, {guidellm_seconds}s)...")
        guidellm_cmd = [
            sys.executable, "-m", "guidellm", "benchmark",
            "--target", f"http://localhost:{PORT}",
            "--profile", guidellm_profile,
            "--rate", guidellm_rate,
            "--max-seconds", guidellm_seconds,
            "--data", "prompt_tokens=256,output_tokens=128",
            "--output-path", "results/guidellm.json",
        ]
        guidellm_result = subprocess.run(guidellm_cmd, capture_output=True, text=True)

        lat = None
        if guidellm_result.returncode == 0:
            from eval.benchmarks import parse_guidellm_results
            lat = parse_guidellm_results("results/guidellm.json")
        else:
            print(f"guidellm warning: {guidellm_result.stderr}")

        # --- Print metrics ---
        print()
        print("=" * 60)
        print("METRICS")
        print("=" * 60)

        gsm8k_em = acc.gsm8k_em if acc and acc.gsm8k_em is not None else 0.0
        print(f"gsm8k_em: {gsm8k_em:.4f}")
        print(f"format_valid_rate: {format_rate:.4f}")

        if lat:
            print(f"p50_ttft_ms: {lat.p50_ttft_ms:.1f}")
            print(f"p95_ttft_ms: {lat.p95_ttft_ms:.1f}")
            print(f"p50_e2e_ms: {lat.p50_e2e_ms:.1f}")
            print(f"p95_e2e_ms: {lat.p95_e2e_ms:.1f}")
            print(f"request_throughput: {lat.request_throughput:.2f}")
            print(f"output_tokens_per_second: {lat.output_tokens_per_second:.1f}")

        # --- Log to W&B ---
        step_metrics = {"gsm8k_em": gsm8k_em, "format_valid_rate": format_rate}
        if lat:
            step_metrics.update({
                "p50_ttft_ms": lat.p50_ttft_ms,
                "p95_ttft_ms": lat.p95_ttft_ms,
                "p50_e2e_ms": lat.p50_e2e_ms,
                "p95_e2e_ms": lat.p95_e2e_ms,
                "request_throughput": lat.request_throughput,
                "output_tokens_per_second": lat.output_tokens_per_second,
            })
        wandb.log(step_metrics, step=1)
        wandb.summary.update(step_metrics)
        wandb.finish()

    finally:
        print("\nShutting down vLLM server...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        log_file.close()
        print(f"Server stopped. Full log: {log_path}")


if __name__ == "__main__":
    main()
