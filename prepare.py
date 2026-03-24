"""
One-time setup for auto-inference optimization.
Called by experiment.yaml during SkyPilot setup phase.

    uv run prepare.py --skip-baseline

This will:
  1. Download and cache the model weights
  2. Pre-download benchmark datasets (GSM8K)
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Prepare auto-inference environment")
    parser.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("AUTO-INFERENCE OPTIMIZATION — SETUP")
    print("=" * 60)

    # 1. Download and cache model
    print("\n[1/2] Downloading model (this may take a while)...")
    from huggingface_hub import snapshot_download
    snapshot_download(args.model, ignore_patterns=["*.bin"])
    print("  Model cached.")

    # 2. Pre-download benchmark datasets
    print("\n[2/2] Caching benchmark datasets...")
    from datasets import load_dataset
    load_dataset("gsm8k", "main", split="test")
    print("  Datasets cached.")

    print("\n" + "=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
