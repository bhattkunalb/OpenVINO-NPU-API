# OpenVINO NPU API

> Ultra-light OpenAI-compatible API for OpenVINO models on Intel NPU.

## Quickstart

Get up and running in 5 minutes with this copy-paste workflow:

```powershell
# 1. Environment Setup
python -m venv ov-env
ov-env\Scripts\activate
pip install -r requirements.txt

# 2. Download a Pre-Built Model (Recommended for NPU)
$env:PYTHONIOENCODING = "utf-8"
huggingface-cli download OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov `
  --local-dir ./models/qwen2.5-1.5b-worker-ov-stateful

# 3. Configure (Save as models.yaml)
@"
models:
  - name: qwen2.5-1.5b-worker-ov-stateful
    path: ./models/qwen2.5-1.5b-worker-ov-stateful
    task: chat
    device: NPU
    preprocess_fn: default_genai
    postprocess_fn: default_genai
"@ | Out-File -Encoding utf8 models.yaml

# 4. Start Server
python -m uvicorn app.main:app --host 0.0.0.0 --port 4647 --workers 1

# 5. Test API (in a new PowerShell window)
curl.exe -s http://localhost:4647/health | ConvertFrom-Json
```

## Prerequisites

> [!WARNING]
> **Python 3.11.x is strictly required.** Python 3.12+ and 3.14 are NOT supported by the OpenVINO export tooling and will cause `ModuleNotFoundError` during export.

- Windows 11 with PowerShell
- Intel Meteor Lake, Arrow Lake, or Lunar Lake CPU
- [Intel NPU Driver](https://www.intel.com/content/www/us/en/download/794636) installed and enabled

## Environment Setup

Always create a dedicated virtual environment to avoid dependency conflicts:

```powershell
# Create and activate venv
python -m venv ov-env
ov-env\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

## Model Acquisition Guide

There are two ways to get OpenVINO-optimized models for NPU inference.

### Option A: Pre-Built Models (Recommended)

The OpenVINO team publishes verified, NPU-compatible models on HuggingFace. These are pre-exported with the correct stateful configuration and `beam_idx` input required by `openvino_genai.LLMPipeline`.

> [!TIP]
> Pre-built models are strongly recommended for NPU deployment. They avoid local export OOM issues and NPU compiler crashes that can occur with custom exports.

```powershell
# Set encoding to avoid UnicodeEncodeError on Windows
$env:PYTHONIOENCODING = "utf-8"

# Qwen 2.5 1.5B (INT4, ~900 MB)
huggingface-cli download OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov `
  --local-dir ./models/qwen2.5-1.5b-worker-ov-stateful

# Qwen 2.5 3B (INT4, ~1.7 GB)
huggingface-cli download OpenVINO/Qwen2.5-3B-Instruct-int4-ov `
  --local-dir ./models/qwen2.5-3b-brain-ov-stateful
```

Available official models: search for `OpenVINO/` on [HuggingFace](https://huggingface.co/OpenVINO).

### Option B: Local Export (Advanced)

If you need a model that isn't available pre-built, export it locally using the included helper script.

> [!IMPORTANT]
> The critical flag is `--task text-generation-with-past`. This produces a stateful model with the `beam_idx` input required by `openvino_genai.LLMPipeline`. Using `--task text-generation` (without `-with-past`) will produce an incompatible model.

#### Using the Helper Script

```powershell
# Install export dependencies (in addition to requirements.txt)
pip install "optimum-intel[openvino,nncf]" datasets torch transformers

# Export Qwen 2.5 1.5B with INT4 compression
python scripts/export_genai.py Qwen/Qwen2.5-1.5B-Instruct `
  ./models/qwen2.5-1.5b-worker-ov-stateful

# Export with INT8 (uses less RAM)
python scripts/export_genai.py Qwen/Qwen2.5-1.5B-Instruct `
  ./models/qwen2.5-1.5b-worker-ov-stateful --weight-format int8
```

#### Using optimum-cli Directly

If you prefer to use `optimum-cli` directly, ensure you use the correct task flag:

```powershell
# Download to local cache first (avoids network issues during export)
$env:HF_HUB_ENABLE_HF_TRANSFER = "1"
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir C:\hf-cache\qwen2.5-1.5b

# Export with the CORRECT task flag
optimum-cli export openvino `
  --model C:\hf-cache\qwen2.5-1.5b `
  --weight-format int4 `
  --trust-remote-code `
  --task text-generation-with-past `
  ./models/qwen2.5-1.5b-worker-ov-stateful
```

#### Verifying the Export

After export, verify the model contains `beam_idx`:

```powershell
Select-String -Path ".\models\qwen2.5-1.5b-worker-ov-stateful\openvino_model.xml" `
  -Pattern "beam_idx"
```

If `beam_idx` is NOT found, the model is incompatible with `LLMPipeline`. Re-export with `--task text-generation-with-past`.

*Note: The output directory is a positional argument at the end of the command. Do not use an `--output` flag.*

## Configuration

Configure your models in `models.yaml`. Ensure the `name` matches the model you plan to request.

```yaml
models:
  - name: qwen2.5-1.5b-worker-ov-stateful
    path: ./models/qwen2.5-1.5b-worker-ov-stateful
    task: chat
    device: NPU
    preprocess_fn: default_genai
    postprocess_fn: default_genai

  - name: qwen2.5-3b-brain-ov-stateful
    path: ./models/qwen2.5-3b-brain-ov-stateful
    task: chat
    device: NPU
    preprocess_fn: default_genai
    postprocess_fn: default_genai
```

### Naming Convention

- `-ov-stateful` suffix: Model exported with `--task text-generation-with-past` (has `beam_idx`, GenAI-compatible).
- `-ov-stateless` suffix: Model exported with `--task text-generation` (no KV cache, slower, use only as fallback).

## Starting the Server

Activate your environment and start the uvicorn server.

> [!CAUTION]
> **Always use `--workers 1`**. The NPU context is held in-process and cannot be shared across multiple workers. Using more than 1 worker will cause instability and crashes.

```powershell
# Activate venv first:
ov-env\Scripts\activate

# Start server
python -m uvicorn app.main:app --host 0.0.0.0 --port 4647 --workers 1
```

## API Reference

### Health Check

```powershell
curl.exe -s http://localhost:4647/health | ConvertFrom-Json
```

### List Models

```powershell
curl.exe -s http://localhost:4647/v1/models | ConvertFrom-Json
```

### Chat Completions

**PowerShell Approach:**
To avoid JSON escaping issues in PowerShell, construct an object and convert it:

```powershell
$body = @{
    model = "qwen2.5-1.5b-worker-ov-stateful"
    messages = @(@{ role = "user"; content = "Hello" })
    max_tokens = 20
} | ConvertTo-Json -Depth 10 -Compress

Invoke-RestMethod -Uri "http://localhost:4647/v1/chat/completions" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

**cmd.exe Approach (Alternative):**

```cmd
curl.exe -s -X POST http://localhost:4647/v1/chat/completions ^
  -H "Content-Type: application/json" ^
  -d "{\"model\":\"qwen2.5-1.5b-worker-ov-stateful\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":20}"
```

## Troubleshooting

| Issue | Symptom | Fix |
| :--- | :--- | :--- |
| Wrong export task | `Stateful models without beam_idx input are not supported` | Re-export with `--task text-generation-with-past` or use `scripts/export_genai.py` |
| NPU compiler crash | `Failed to compile Model0_kv1152_FCEW000__0 for all devices in [NPU]` | Use a pre-built model from `OpenVINO/` org on HuggingFace |
| Python 3.14 incompatibility | `ModuleNotFoundError: optimum.exporters.base` | Use Python 3.11.x only |
| OpenVINO import error | `ModuleNotFoundError: openvino.runtime` | Update imports: `from openvino import Core` (not `from openvino.runtime import Core`) |
| TypeError in generate | `incompatible function arguments... ChatHistory` | Update `app/pipeline.py` to format messages as string prompts |
| PowerShell `curl` alias | `Cannot bind parameter 'Headers'` | Use `curl.exe` or `Invoke-RestMethod` |
| JSON escaping in PowerShell | `JSON decode error` | Use `ConvertTo-Json` or save to `.json` file |
| UnicodeEncodeError on Windows | `charmap codec can't encode characters` | Set `$env:PYTHONIOENCODING="utf-8"` before running commands |
| Error handler crash | `TypeError` in global exception handler | Ensure `JSONResponse` uses keyword args (`status_code=500, content=...`) |
| Models not loading | `"loaded_models":[]` in health response | Normal lazy loading; first request triggers compilation (~2-5s delay) |
| Export hangs/fails | `IncompleteRead`, `NameResolutionError` | Install `hf_transfer`, set `$env:HF_HUB_ENABLE_HF_TRANSFER="1"`, use local cache |
| Export OOM | `Failed to allocate ... bytes of memory` | Increase Windows pagefile to 16-32 GB, or use `--weight-format int8`/`fp16` |
| NPU not detected | `NPU not in available_devices` | Install Intel NPU driver: <https://www.intel.com/content/www/us/en/download/794636> |

### Troubleshooting Checklist

If things aren't working, check these common pitfalls:

#### Server won't start

- [ ] Python 3.11.x active? `python --version`
- [ ] Virtual environment activated? Prompt shows `(ov-env)`
- [ ] Dependencies installed? `pip list | Select-String -Pattern "openvino"`
- [ ] `app/main.py` exists? `ls app\main.py`

#### Model export fails

- [ ] Using `--task text-generation-with-past`? (NOT `text-generation`)
- [ ] Output dir is positional arg (no `--output`)?
- [ ] Local cache path used (not direct HF download)?
- [ ] Pagefile increased to 16+ GB?
- [ ] `hf_transfer` installed and `$env:HF_HUB_ENABLE_HF_TRANSFER="1"` set?

#### Inference returns 500 error

- [ ] Check server logs for OpenVINO errors
- [ ] Model compiled on first request? Wait 2-5 seconds
- [ ] NPU driver installed? `python -c "from openvino import Core; print(Core().available_devices)"`
- [ ] Exported model has `beam_idx`? `Select-String -Path ".\models\...\openvino_model.xml" -Pattern "beam_idx"`

#### curl/PowerShell issues

- [ ] Using `curl.exe` (not PowerShell alias)?
- [ ] JSON built with `ConvertTo-Json` or saved to file?
- [ ] Testing in `cmd.exe` instead of PowerShell?

## Production Hardening

The codebase has undergone a rigorous production-readiness review and hardening process:

- **Zero-Warning Stability**: Resolved all IDE import errors, PEP 8 line length violations, and linting warnings.
- **Environment Synchronization**: Unified project dependencies across `venv` and specialized `ov-env` to ensure consistent runtime behavior.
- **Thread Safety**: Refactored core generators and model access patterns to ensure safe multi-threaded inference under `uvicorn` lifespan management.
- **Strict Linting Compliance**: Implemented automated whitespace cleanup and standardized function signatures.
- **Standardized Error Handling**: OpenAI-compatible error response formatting ensured across all endpoints.

## Performance Notes

### Memory Requirements

| Model | Params | INT4 Disk | RAM for Export | Pagefile Recommended |
| :--- | :--- | :--- | :--- | :--- |
| Qwen 2.5 3B | 3B | ~1.7 GB | ~8 GB | 16 GB |
| Qwen 2.5 1.5B | 1.5B | ~0.9 GB | ~6 GB | 12 GB |
| Phi-3 Mini | 3.8B | ~2.2 GB | ~10 GB | 20 GB |
| Phi-3.5 Mini | 3.8B | ~2.2 GB | ~10 GB | 20 GB |

### Lazy Loading

Models are **lazy-loaded**. They will not appear in the `"loaded_models"` array of the `/health` endpoint until the *first* inference request is made. The first request will experience a compilation delay of ~2-5 seconds. Subsequent requests will be fast.

## Contributing

To add new models:

1. **Preferred**: Download a pre-built model from the [OpenVINO HuggingFace org](https://huggingface.co/OpenVINO).
2. **Alternative**: Export using `scripts/export_genai.py` with `--task text-generation-with-past`.
3. Verify `beam_idx` exists in the exported `openvino_model.xml`.
4. Add the model to `models.yaml`.
5. Test with the API.

Please report any issues or submit PRs for improvements!
