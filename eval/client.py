"""
Minimal inference client for format canary only.

Accuracy eval uses lm-eval-harness (talks to vLLM directly).
Latency eval uses guidellm (talks to vLLM endpoint).
This client is only used for the 8 format canary prompts.
"""

import requests
from dataclasses import dataclass


@dataclass
class InferenceClient:
    """Client for vLLM's OpenAI-compatible endpoint."""

    base_url: str = "http://localhost:8000"
    model: str = "Qwen/Qwen3.5-35B-A3B"

    # Locked generation params
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512
    seed: int = 42

    def __call__(self, prompt: str) -> str:
        response = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "model": self.model,
                "prompt": prompt,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
                "seed": self.seed,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["text"]
