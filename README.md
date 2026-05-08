# OpenVINO NPU API

> Ultra-light OpenAI-compatible API for OpenVINO models on Intel NPU.

## ⚡ Quick Start (5 Minutes)

1. **Clone & Setup**

   ```powershell
   git clone https://github.com/bhattkunalb/OpenVINO-NPU-API.git
   cd OpenVINO-NPU-API
   python -m venv ov-env
   ov-env\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Download a Model**

   ```powershell
   python scripts/download_prebuilt.py qwen2.5-1.5b
   ```

3. **Start Server**

   ```powershell
   python -m uvicorn app.main:app --host 0.0.0.0 --port 4647 --workers 1
   ```

4. **Test**

   ```powershell
   curl.exe -s -X POST http://localhost:4647/v1/chat/completions `
     -H "Content-Type: application/json" `
     -d "{\"model\":\"qwen2.5-1.5b-npu\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":20}"
   ```

✅ Done! You have a local NPU inference API.

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

## 📦 Using Pre-Built NPU Models (Recommended)

This API is designed to work with **pre-converted, NPU-optimized models** from official sources. No export step required.

### Step 1: Download a Pre-Built Model

Choose from Intel's curated collection: [LLMs Optimized for NPU](https://huggingface.co/collections/OpenVINO/llms-optimized-for-npu)

```powershell
# Example: Download Qwen 2.5 1.5B INT4 for NPU
huggingface-cli download OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov --local-dir ./models/qwen2.5-1.5b-npu

# Example: Download Phi-3 Mini INT4 for NPU
huggingface-cli download OpenVINO/Phi-3-mini-4k-instruct-int4-ov --local-dir ./models/phi-3-mini-npu

# Example: Download Llama 3.2 1B INT4 for NPU
huggingface-cli download OpenVINO/Llama-3.2-1B-Instruct-int4-ov --local-dir ./models/llama-3.2-1b-npu
```

> ✅ These models are pre-quantized (INT4), pre-converted to OpenVINO IR, and tested on Intel Core Ultra NPUs.

### Step 2: Update `models.yaml`

Edit `models.yaml` to point to your downloaded model:

```yaml
models:
  - name: qwen2.5-1.5b-npu
    path: ./models/qwen2.5-1.5b-npu
    task: chat
    device: NPU
    preprocess_fn: default_genai
    postprocess_fn: default_genai
```

### Step 3: Start the Server

```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 4647 --workers 1
```

### Step 4: Test

```powershell
$body = @{
    model = "qwen2.5-1.5b-npu"
    messages = @(@{ role = "user"; content = "Hello" })
    max_tokens = 20
} | ConvertTo-Json -Depth 10 -Compress

Invoke-RestMethod -Uri "http://localhost:4647/v1/chat/completions" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

✅ Expected: JSON response with AI reply in ~3-5 seconds on NPU.

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

## 🔧 Troubleshooting Pre-Built Models

### Model not found / 404 error

- Ensure you downloaded the model first: `python scripts/download_prebuilt.py <model_id>`
- Verify the path in `models.yaml` matches the actual folder name
- Check `ls ./models/` to confirm files exist

### NPU not detected

```powershell
python -c "import openvino; print(openvino.Core().available_devices)"
```

- If `NPU` is missing: Install [Intel NPU driver](https://www.intel.com/content/www/us/en/download/794636)
- Ensure the hardware is enabled in Device Manager

### First request is slow (~5-10s)

- Normal: NPU compilation happens on first load. Subsequent requests are fast.

### Request fails with "beam_idx" or "StatefulToStateless" error

- You may have accidentally used a custom-exported model.
- **Fix**: Delete the custom model folder and download a pre-built one:

  ```powershell
  Remove-Item ./models/your-custom-model -Recurse -Force
  python scripts/download_prebuilt.py OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov
  ```

### Out of memory on NPU

- Pre-built INT4 models require ~1.5-4 GB NPU memory.
- Close other apps; ensure no other NPU processes are running.
- Try a smaller model (e.g., `llama-3.2-1b-npu`).

### Model loads but response is slow

- Check you're actually using NPU: server logs should show `Compiling on NPU`.
- If it says `CPU`, verify `device: NPU` in `models.yaml`.

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

1. Download a pre-built model from the [OpenVINO HuggingFace org](https://huggingface.co/OpenVINO).
2. Add the model to `models.yaml`.
3. Test with the API.

Please report any issues or submit PRs for improvements!
