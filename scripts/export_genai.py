#!/usr/bin/env python3
"""Export a HuggingFace model to OpenVINO IR with GenAI compatibility.

This script wraps the optimum-cli export to ensure the correct flags
are used for openvino_genai.LLMPipeline compatibility on Intel NPU.

Key points:
  - Uses --task text-generation-with-past to produce a stateful model
    with beam_idx input (required by LLMPipeline).
  - INT4 weight compression is applied by default for NPU efficiency.

RECOMMENDED: For production NPU deployment, prefer pre-built official
models from the OpenVINO org on HuggingFace instead of local export:

    huggingface-cli download OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov \\
      --local-dir ./models/qwen2.5-1.5b-worker-ov-stateful

These models are verified to compile correctly on Intel NPU hardware.

Usage:
    python scripts/export_genai.py <model_id_or_path> <output_dir> [options]

Examples:
    # Export from HuggingFace hub (INT4, default)
    python scripts/export_genai.py Qwen/Qwen2.5-1.5B-Instruct \\
        ./models/qwen2.5-1.5b-worker-ov-stateful

    # Export from local cache with INT8
    python scripts/export_genai.py C:\\hf-cache\\qwen2.5-1.5b \\
        ./models/qwen2.5-1.5b-worker-ov-stateful --weight-format int8

    # Export with custom number of calibration samples
    python scripts/export_genai.py Qwen/Qwen2.5-3B-Instruct \\
        ./models/qwen2.5-3b-brain-ov-stateful --num-samples 64
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    """Parse args and invoke optimum-cli with the correct flags."""
    parser = argparse.ArgumentParser(
        description="Export a model to OpenVINO IR with GenAI compatibility.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "NOTE: The exported model must contain a 'beam_idx' input to be\n"
            "compatible with openvino_genai.LLMPipeline. This script ensures\n"
            "that by using --task text-generation-with-past.\n"
            "\n"
            "For NPU deployment, pre-built models from the OpenVINO org on\n"
            "HuggingFace are strongly recommended over local export."
        ),
    )
    parser.add_argument(
        "model", help="HuggingFace model ID or local path to model directory."
    )
    parser.add_argument(
        "output", help="Output directory for the exported OpenVINO model."
    )
    parser.add_argument(
        "--weight-format",
        choices=["fp32", "fp16", "int8", "int4"],
        default="int4",
        help="Weight compression format (default: int4).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Trust remote code in model repo (default: True).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=128,
        help="Number of calibration samples for INT4 quantization (default: 128).",
    )
    args = parser.parse_args()

    # Build the optimum-cli command
    cmd = [
        sys.executable, "-m", "optimum.commands.optimum_cli",
        "export", "openvino",
        "--model", args.model,
        "--task", "text-generation-with-past",
        "--weight-format", args.weight_format,
        args.output,
    ]

    if args.trust_remote_code:
        cmd.insert(-1, "--trust-remote-code")

    if args.weight_format == "int4":
        cmd.insert(-1, "--num-samples")
        cmd.insert(-1, str(args.num_samples))

    print(f"Running: {' '.join(cmd)}")
    print()
    print("This may take 10-30 minutes depending on model size and hardware.")
    print("If INT4 export fails with OOM, try --weight-format int8 or fp16.")
    print()

    result = subprocess.run(cmd, check=False)

    if result.returncode == 0:
        print()
        print(f"Export complete: {args.output}")
        print()
        print("Verify the model has beam_idx (required for GenAI):")
        print(f'  Select-String -Path "{args.output}\\openvino_model.xml"'
              ' -Pattern "beam_idx"')
        print()
        print("If beam_idx is present, add the model to models.yaml and start the server.")
    else:
        print()
        print("Export FAILED. Common fixes:")
        print("  - Increase Windows pagefile to 16-32 GB")
        print("  - Use --weight-format int8 or fp16 (less RAM)")
        print("  - Use a pre-built model instead:")
        print("    huggingface-cli download OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov \\")
        print("      --local-dir ./models/qwen2.5-1.5b-worker-ov-stateful")

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
