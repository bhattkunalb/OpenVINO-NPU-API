#!/usr/bin/env python3
"""
download_prebuilt.py - Download pre-built NPU-optimized OpenVINO models

This script downloads models from official Intel/OpenVINO collections.
No export or compilation required - models are ready for NPU inference.

Usage:
    python scripts/download_prebuilt.py <model_alias_or_id> [output_dir]

Examples:
    python scripts/download_prebuilt.py qwen2.5-1.5b
    python scripts/download_prebuilt.py OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov ./models/my-qwen
    python scripts/download_prebuilt.py phi-3-mini

Recommended Models:
    qwen2.5-1.5b    → OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov
    qwen2.5-3b      → OpenVINO/Qwen2.5-3B-Instruct-int4-ov
    phi-3-mini      → OpenVINO/Phi-3-mini-4k-instruct-int4-ov
    llama-3.2-1b    → OpenVINO/Llama-3.2-1B-Instruct-int4-ov
    gemma-2b        → OpenVINO/gemma-2b-it-int4-ov
"""

import sys
import subprocess
from pathlib import Path

# Curated list of known-working NPU models from official collections
RECOMMENDED_MODELS = {
    "qwen2.5-1.5b": "OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov",
    "qwen2.5-3b": "OpenVINO/Qwen2.5-3B-Instruct-int4-ov",
    "phi-3-mini": "OpenVINO/Phi-3-mini-4k-instruct-int4-ov",
    "llama-3.2-1b": "OpenVINO/Llama-3.2-1B-Instruct-int4-ov",
    "gemma-2b": "OpenVINO/gemma-2b-it-int4-ov",
}

def download_model(target_model_id: str, target_output_dir: str = None) -> bool:
    """Download a pre-built model using huggingface_hub CLI."""

    if target_output_dir is None:
        # Auto-generate output directory from model name
        name = target_model_id.split("/")[-1]
        # Clean up common suffixes for nicer folder names
        for suffix in ["-int4-ov", "-ov", "-instruct", "-it"]:
            name = name.replace(suffix, "")
        target_output_dir = f"./models/{name}-npu"

    print(f"📥 Downloading {target_model_id} to {target_output_dir}...")
    print("💡 This may take 5-15 minutes depending on your internet connection.\n")

    # Build command with Windows-compatible flags
    cmd = [
        sys.executable, "-m", "huggingface_hub.cli",
        "download", target_model_id,
        "--local-dir", target_output_dir,
        "--local-dir-use-symlinks", "false",  # Avoid symlink issues on Windows
        "--resume-download",  # Enable resumable downloads
    ]

    # Run with real-time output
    try:
        subprocess.run(
            cmd,
            capture_output=False,  # Show progress in real-time
            text=True,
            check=True
        )
        print(f"\n✅ Download complete: {Path(target_output_dir).resolve()}")

        # List key files to confirm success
        model_path = Path(target_output_dir)
        files = list(model_path.glob("openvino_*"))
        if files:
            print(f"📁 Model files: {', '.join(f.name for f in files[:3])}")
            if len(files) > 3:
                print(f"   ... and {len(files) - 3} more files")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Download failed with exit code {e.returncode}")
        print("💡 Try: pip install -U huggingface_hub")
        print("💡 Or check your internet connection and retry")
        return False
    except FileNotFoundError:
        print("\n❌ huggingface_hub CLI not found")
        print("💡 Install with: pip install huggingface_hub")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  Download interrupted. Partial files may remain in {target_output_dir}")
        return False

def print_help():
    """Print usage information and recommended models."""
    print("📦 Pre-built NPU Model Downloader")
    print("=" * 50)
    print("\n✅ These models are pre-converted, INT4 quantized,")
    print("   and tested on Intel Core Ultra NPUs.\n")

    print("Recommended models (use alias or full ID):")
    print("-" * 50)
    for alias, m_id in RECOMMENDED_MODELS.items():
        print(f"  {alias:15s} → {m_id}")

    print("\nUsage:")
    print("  python download_prebuilt.py <model_alias_or_id> [output_dir]")
    print("\nExamples:")
    print("  python download_prebuilt.py qwen2.5-1.5b")
    print("  python download_prebuilt.py OpenVINO/Phi-3-mini-4k-instruct-int4-ov")
    print("  python download_prebuilt.py llama-3.2-1b ./models/my-llama")
    print("\nTips:")
    print("  • Set $env:HF_HUB_ENABLE_HF_TRANSFER=1 for faster downloads")
    print("  • Downloads resume automatically if interrupted")
    print("  • Models are saved to ./models/<name>-npu by default")

if __name__ == "__main__":
    # Handle help flag
    if len(sys.argv) < 2 or sys.argv[1] in ["-h", "--help", "help"]:
        print_help()
        sys.exit(0)

    model_arg = sys.argv[1]
    arg_output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    # Resolve alias to full model ID if provided
    if model_arg in RECOMMENDED_MODELS:
        arg_model_id = RECOMMENDED_MODELS[model_arg]
        print(f"ℹ️  Using alias '{model_arg}' → {arg_model_id}\n")
    else:
        arg_model_id = model_arg
        # Warn if not in recommended list
        if arg_model_id not in RECOMMENDED_MODELS.values():
            print(f"⚠️  '{arg_model_id}' is not in the recommended list.")
            print("   Ensure it's a pre-built OpenVINO model for NPU.\n")

    # Execute download
    is_success = download_model(arg_model_id, arg_output_dir)

    if is_success:
        print("\n🎯 Next steps:")
        print(f"  1. Verify model: ls {arg_output_dir or './models/'}")
        print(f"  2. Update models.yaml with path: {arg_output_dir or './models/<name>-npu'}")
        print("  3. Start server: python -m uvicorn app.main:app --workers 1")
        print("  4. Test: See README.md for PowerShell examples")

    sys.exit(0 if is_success else 1)
