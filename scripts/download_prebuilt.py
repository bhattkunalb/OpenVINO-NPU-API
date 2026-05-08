#!/usr/bin/env python3
"""
download_prebuilt.py - Download pre-built NPU-optimized OpenVINO models

Usage:
    python scripts/download_prebuilt.py <model_id> [output_dir]

Example:
    python scripts/download_prebuilt.py OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov
"""

import sys
import subprocess
from pathlib import Path

# Curated list of known-working NPU models
RECOMMENDED_MODELS = {
    "qwen2.5-1.5b": "OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov",
    "qwen2.5-3b": "OpenVINO/Qwen2.5-3B-Instruct-int4-ov",
    "phi-3-mini": "OpenVINO/Phi-3-mini-4k-instruct-int4-ov",
    "llama-3.2-1b": "OpenVINO/Llama-3.2-1B-Instruct-int4-ov",
    "gemma-2b": "OpenVINO/gemma-2b-it-int4-ov",
}

def download_model(model_id: str, output_dir: str | None = None) -> bool:
    """
    Download a pre-built model from HuggingFace.

    Args:
        model_id: The HuggingFace repository ID or alias.
        output_dir: Optional directory to save the model.

    Returns:
        True if the download succeeded, False otherwise.
    """
    if output_dir is None:
        # Auto-generate output dir from model name
        name = model_id.split("/")[-1].replace("-int4-ov", "").replace("-ov", "")
        output_dir = f"./models/{name}-npu"

    print(f"📥 Downloading {model_id} to {output_dir}...")

    cmd = [
        sys.executable, "-m", "huggingface_hub.cli",
        "download", model_id,
        "--local-dir", output_dir,
        "--local-dir-use-symlinks", "false"
    ]

    try:
        subprocess.run(cmd, capture_output=False, text=True, check=True)
        print(f"✅ Download complete: {output_dir}")
        print(f"📁 Files: {list(Path(output_dir).glob('openvino_*'))}")
        return True
    except subprocess.CalledProcessError:
        print("❌ Download failed. Try: pip install -U huggingface_hub")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("📦 Pre-built NPU Model Downloader")
        print("\nRecommended models:")
        for alias_name, hf_id in RECOMMENDED_MODELS.items():
            print(f"  {alias_name:20s} → {hf_id}")
        print("\nUsage: python download_prebuilt.py <model_id> [output_dir]")
        print("Example: python download_prebuilt.py OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov")
        sys.exit(1)

    target_model = sys.argv[1]
    target_dir = sys.argv[2] if len(sys.argv) > 2 else None

    # Resolve alias if provided
    if target_model in RECOMMENDED_MODELS:
        target_model = RECOMMENDED_MODELS[target_model]
        print(f"ℹ️  Using alias: {target_model}")

    SUCCESS = download_model(target_model, target_dir)
    sys.exit(0 if SUCCESS else 1)
