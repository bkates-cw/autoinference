"""
Format canary + guardrail/commit logic for auto-inference optimization.

Accuracy benchmarking is handled by lm-eval-harness.
Latency benchmarking is handled by guidellm.
This module handles:
  1. Format validation canary (8 strict-format prompts)
  2. Parsing lm_eval and guidellm outputs
  3. Guardrail checks and commit decisions
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# Data classes
# =============================================================================


@dataclass
class AccuracyResult:
    gsm8k_em: Optional[float] = None  # exact match, 0.0-1.0
    mmlu_acc: Optional[float] = None   # accuracy, 0.0-1.0


@dataclass
class LatencyResult:
    p50_ttft_ms: float = 0.0
    p95_ttft_ms: float = 0.0
    p50_e2e_ms: float = 0.0
    p95_e2e_ms: float = 0.0
    request_throughput: float = 0.0
    output_tokens_per_second: float = 0.0


@dataclass
class EvalResult:
    accuracy: AccuracyResult = field(default_factory=AccuracyResult)
    latency: LatencyResult = field(default_factory=LatencyResult)
    format_valid_rate: float = 1.0
    format_canary_details: list[dict] = field(default_factory=list)
    passed_guardrails: bool = True
    guardrail_violations: list[str] = field(default_factory=list)
    should_commit: bool = False


# =============================================================================
# Format validation canary
# =============================================================================

FORMAT_CANARY_PROMPTS = [
    {
        "prompt": 'Return a JSON object with exactly these fields: {"name": "test", "count": 42, "active": true}. Output only the JSON, nothing else.',
        "validate": lambda text: _validate_json_fields(text, {"name": str, "count": int, "active": bool}),
    },
    {
        "prompt": "List exactly 3 US state capitals, one per line, numbered 1-3. Output only the list.",
        "validate": lambda text: _validate_numbered_list(text, 3),
    },
    {
        "prompt": "Output exactly this string and nothing else: CANARY_CHECK_PASSED",
        "validate": lambda text: "CANARY_CHECK_PASSED" in text.strip(),
    },
    {
        "prompt": "Return a JSON array of exactly 5 integers from 1 to 10 in ascending order. Output only the JSON array.",
        "validate": lambda text: _validate_json_int_array(text, length=5),
    },
    {
        "prompt": "What is 7 * 8? Reply with only the number.",
        "validate": lambda text: "56" in text.strip(),
    },
    {
        "prompt": 'Return a JSON object with fields "city" (string), "population" (integer), and "country" (string) for Tokyo. Output only the JSON.',
        "validate": lambda text: _validate_json_fields(text, {"city": str, "population": int, "country": str}),
    },
    {
        "prompt": "List the days of the week, Monday through Friday only, as a comma-separated list. Output only the list.",
        "validate": lambda text: all(
            day.lower() in text.lower()
            for day in ["monday", "tuesday", "wednesday", "thursday", "friday"]
        ) and "saturday" not in text.lower(),
    },
    {
        "prompt": 'Respond with exactly: {"status": "ok"}',
        "validate": lambda text: _validate_json_fields(text, {"status": str}),
    },
]


def _validate_json_fields(text: str, expected_fields: dict) -> bool:
    match = re.search(r"\{[^{}]*\}", text.strip())
    if not match:
        return False
    try:
        data = json.loads(match.group())
        for field_name, field_type in expected_fields.items():
            if field_name not in data:
                return False
            if not isinstance(data[field_name], field_type):
                return False
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def _validate_numbered_list(text: str, expected_count: int) -> bool:
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    numbered = [line for line in lines if re.match(r"^\d+[.\)]\s", line)]
    return len(numbered) == expected_count


def _validate_json_int_array(text: str, length: int) -> bool:
    match = re.search(r"\[[^\[\]]*\]", text.strip())
    if not match:
        return False
    try:
        data = json.loads(match.group())
        return (
            isinstance(data, list)
            and len(data) == length
            and all(isinstance(x, int) for x in data)
        )
    except (json.JSONDecodeError, TypeError):
        return False


def run_format_canary(inference_fn) -> tuple[float, list[dict]]:
    """Run format canary prompts. Returns (valid_rate, details)."""
    results = []
    for canary in FORMAT_CANARY_PROMPTS:
        output = inference_fn(canary["prompt"])
        passed = canary["validate"](output)
        results.append({
            "prompt": canary["prompt"][:80] + "...",
            "passed": passed,
            "raw_output": output,
        })
    valid_rate = sum(1 for r in results if r["passed"]) / len(results)
    return valid_rate, results


# =============================================================================
# Parse lm_eval output
# =============================================================================


def parse_lm_eval_results(results_path: str) -> AccuracyResult:
    """
    Parse lm-eval-harness JSON output.

    lm_eval writes results to a JSON file with structure:
    {
      "results": {
        "gsm8k": {"exact_match,strict-match": 0.75, ...},
        "mmlu": {"acc,none": 0.65, ...}
      }
    }
    """
    with open(results_path) as f:
        data = json.load(f)

    results = data.get("results", {})
    acc = AccuracyResult()

    # GSM8K — look for exact_match metric
    for key in results:
        if "gsm8k" in key.lower():
            gsm8k = results[key]
            for metric_key in ["exact_match,strict-match", "exact_match,flexible-extract", "exact_match"]:
                if metric_key in gsm8k:
                    acc.gsm8k_em = gsm8k[metric_key]
                    break

    # MMLU — look for acc metric
    for key in results:
        if "mmlu" in key.lower():
            mmlu = results[key]
            for metric_key in ["acc,none", "acc"]:
                if metric_key in mmlu:
                    acc.mmlu_acc = mmlu[metric_key]
                    break

    return acc


# =============================================================================
# Parse guidellm output
# =============================================================================


def parse_guidellm_results(results_path: str) -> LatencyResult:
    """
    Parse guidellm benchmark JSON output.

    guidellm writes benchmarks.json with benchmark entries containing
    metrics like request_latency, time_to_first_token, etc. with
    percentile distributions.
    """
    with open(results_path) as f:
        data = json.load(f)

    lat = LatencyResult()

    # guidellm outputs a list of benchmarks (one per rate point in sweep)
    # Use the last benchmark (highest sustainable rate) or the single result
    benchmarks = data if isinstance(data, list) else data.get("benchmarks", [data])
    if not benchmarks:
        return lat

    # Use the benchmark with highest successful request rate
    bench = max(benchmarks, key=lambda b: b.get("request_rate", 0)) if len(benchmarks) > 1 else benchmarks[-1]

    # Extract metrics from the benchmark
    metrics = bench.get("metrics", bench)

    # TTFT
    ttft = metrics.get("time_to_first_token", metrics.get("ttft", {}))
    if isinstance(ttft, dict):
        percentiles = ttft.get("percentiles", {})
        lat.p50_ttft_ms = percentiles.get("p50", percentiles.get("50", 0)) * 1000
        lat.p95_ttft_ms = percentiles.get("p95", percentiles.get("95", 0)) * 1000

    # E2E / request latency
    e2e = metrics.get("request_latency", metrics.get("e2e", {}))
    if isinstance(e2e, dict):
        percentiles = e2e.get("percentiles", {})
        lat.p50_e2e_ms = percentiles.get("p50", percentiles.get("50", 0)) * 1000
        lat.p95_e2e_ms = percentiles.get("p95", percentiles.get("95", 0)) * 1000

    # Throughput
    lat.request_throughput = metrics.get("request_rate", metrics.get("completed_request_rate", 0))
    lat.output_tokens_per_second = metrics.get("output_tokens_per_second", metrics.get("output_token_throughput", 0))

    return lat


# =============================================================================
# Guardrail + commit decision
# =============================================================================


def check_guardrails(
    result: EvalResult,
    baseline: EvalResult,
    max_accuracy_degradation_pct: float = 5.0,
    goodput_improvement_min_pct: float = 1.0,
    p95_ttft_max_regression_pct: float = 10.0,
) -> EvalResult:
    """
    Check quality guardrails and commit criteria against baseline.
    Mutates and returns result.
    """
    # Format canary must be 100%
    if result.format_valid_rate < 1.0:
        result.passed_guardrails = False
        result.guardrail_violations.append(
            f"format_valid_rate: {result.format_valid_rate:.0%} (must be 100%)"
        )

    # GSM8K accuracy
    if result.accuracy.gsm8k_em is not None and baseline.accuracy.gsm8k_em is not None:
        if baseline.accuracy.gsm8k_em > 0:
            deg = (baseline.accuracy.gsm8k_em - result.accuracy.gsm8k_em) / baseline.accuracy.gsm8k_em * 100
            if deg > max_accuracy_degradation_pct:
                result.passed_guardrails = False
                result.guardrail_violations.append(
                    f"gsm8k: {deg:.1f}% degradation (limit: {max_accuracy_degradation_pct}%)"
                )

    # MMLU accuracy
    if result.accuracy.mmlu_acc is not None and baseline.accuracy.mmlu_acc is not None:
        if baseline.accuracy.mmlu_acc > 0:
            deg = (baseline.accuracy.mmlu_acc - result.accuracy.mmlu_acc) / baseline.accuracy.mmlu_acc * 100
            if deg > max_accuracy_degradation_pct:
                result.passed_guardrails = False
                result.guardrail_violations.append(
                    f"mmlu: {deg:.1f}% degradation (limit: {max_accuracy_degradation_pct}%)"
                )

    # Commit decision: guardrails pass + performance improves
    if result.passed_guardrails:
        base_throughput = baseline.latency.request_throughput
        new_throughput = result.latency.request_throughput
        base_ttft = baseline.latency.p95_ttft_ms
        new_ttft = result.latency.p95_ttft_ms

        throughput_improvement = (
            (new_throughput - base_throughput) / base_throughput * 100
            if base_throughput > 0 else 0
        )
        ttft_regression = (
            (new_ttft - base_ttft) / base_ttft * 100
            if base_ttft > 0 else 0
        )

        if throughput_improvement >= goodput_improvement_min_pct and ttft_regression <= p95_ttft_max_regression_pct:
            result.should_commit = True
        else:
            if throughput_improvement < goodput_improvement_min_pct:
                result.guardrail_violations.append(
                    f"throughput: +{throughput_improvement:.1f}% (need +{goodput_improvement_min_pct}%)"
                )
            if ttft_regression > p95_ttft_max_regression_pct:
                result.guardrail_violations.append(
                    f"p95_ttft: +{ttft_regression:.1f}% regression (limit: {p95_ttft_max_regression_pct}%)"
                )

    return result
