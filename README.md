# OpenVINO NPU API

> 🚀 Superlight OpenAI-compatible API for OpenVINO models on Intel NPU.

## ⚡ Quickstart

Get up and running in 5 minutes with this copy-paste workflow:

```powershell
# 1. Environment Setup
python -m venv ov-env
ov-env\Scripts\activate
pip install "optimum[openvino]" "openvino>=2025.4.0" "openvino-genai>=2025.4.0" nncf openvino-tokenizers hf_transfer fastapi uvicorn

# 2. Download & Export (Qwen 2.5 3B)
$env:HF_HUB_ENABLE_HF_TRANSFER = "1"
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir C:\hf-cache\qwen2.5-3b
optimum-cli export openvino --model C:\hf-cache\qwen2.5-3b --weight-format int4 --trust-remote-code --task text-generation --stateful ./models/qwen2.5-3b-brain-ov-stateful

# 3. Configure (Save as models.yaml)
@"
models:
  - name: qwen2.5-3b-brain-ov-stateful
    path: ./models/qwen2.5-3b-brain-ov-stateful
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

## 📋 Prerequisites

> [!WARNING]
> **Python 3.11.x is strictly required.** Python 3.12+ and 3.14 are NOT supported by the OpenVINO export tooling and will cause `ModuleNotFoundError` during export.

- Windows 11 with PowerShell
- Intel Meteor Lake, Arrow Lake, or Lunar Lake CPU
- [Intel NPU Driver](https://www.intel.com/content/www/us/en/download/794636) installed and enabled

## 🛠️ Environment Setup

Always create a dedicated virtual environment to avoid dependency conflicts:

```powershell
# Create and activate venv
python -m venv ov-env
ov-env\Scripts\activate

# Install required packages
pip install "optimum[openvino]" "openvino>=2025.4.0" "openvino-genai>=2025.4.0" nncf openvino-tokenizers hf_transfer
```

## 📦 Model Export Guide

Exporting models locally ensures they run efficiently on the NPU.

> [!IMPORTANT]
> The `--stateful` flag is REQUIRED for OpenVINO GenAI compatibility. It adds the `beam_idx` input needed for KV caching.

### The Recommended Workflow (Download-to-Cache)

Avoid network interruptions by downloading to a local cache first, then exporting from there.

#### 1. Qwen 2.5 3B (Brain)

```powershell
# Enable fast downloads
$env:HF_HUB_ENABLE_HF_TRANSFER = "1"

# Download to cache
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir C:\hf-cache\qwen2.5-3b

# Export to OpenVINO (takes 10-30 mins)
optimum-cli export openvino `
  --model C:\hf-cache\qwen2.5-3b `
  --weight-format int4 `
  --trust-remote-code `
  --task text-generation `
  --stateful `
  ./models/qwen2.5-3b-brain-ov-stateful
```

#### 2. Qwen 2.5 1.5B (Worker)

```powershell
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir C:\hf-cache\qwen2.5-1.5b
optimum-cli export openvino `
  --model C:\hf-cache\qwen2.5-1.5b `
  --weight-format int4 `
  --trust-remote-code `
  --task text-generation `
  --stateful `
  ./models/qwen2.5-1.5b-worker-ov-stateful
```

#### 3. Phi-3 Mini

```powershell
huggingface-cli download microsoft/Phi-3-mini-4k-instruct --local-dir C:\hf-cache\phi-3-mini
optimum-cli export openvino `
  --model C:\hf-cache\phi-3-mini `
  --weight-format int4 `
  --trust-remote-code `
  --task text-generation `
  --stateful `
  ./models/phi-3-mini-ov-stateful
```

#### 4. Phi-3.5 Mini

```powershell
huggingface-cli download microsoft/Phi-3.5-mini-instruct --local-dir C:\hf-cache\phi-3.5-mini
optimum-cli export openvino `
  --model C:\hf-cache\phi-3.5-mini `
  --weight-format int4 `
  --trust-remote-code `
  --task text-generation `
  --stateful `
  ./models/phi-3.5-mini-ov-stateful
```

*Note: The output directory is a positional argument at the end of the command. Do not use an `--output` flag.*

## ⚙️ Configuration

Configure your models in `models.yaml`. Ensure the `name` ends with `-ov-stateful` and matches the `path`.

```yaml
models:
  - name: qwen2.5-3b-brain-ov-stateful
    path: ./models/qwen2.5-3b-brain-ov-stateful
    task: chat
    device: NPU
    preprocess_fn: default_genai
    postprocess_fn: default_genai

  - name: qwen2.5-1.5b-worker-ov-stateful
    path: ./models/qwen2.5-1.5b-worker-ov-stateful
    task: chat
    device: NPU
    preprocess_fn: default_genai
    postprocess_fn: default_genai
```

## 🚀 Starting the Server

Activate your environment and start the uvicorn server.

> [!CAUTION]
> **Always use `--workers 1`**. The NPU context is held in-process and cannot be shared across multiple workers. Using more than 1 worker will cause instability and crashes.

```powershell
# Activate venv first:
ov-env\Scripts\activate

# Start server
python -m uvicorn app.main:app --host 0.0.0.0 --port 4647 --workers 1
```

## 🔌 API Reference

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
    model = "qwen2.5-3b-brain-ov-stateful"
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
  -d "{\"model\":\"qwen2.5-3b-brain-ov-stateful\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":20}"
```

## 🔧 Troubleshooting

| Issue | Symptom | Fix |
| :--- | :--- | :--- |
| Python 3.14 incompatibility | `ModuleNotFoundError: optimum.exporters.base` | Use Python 3.11.x only |
| Missing `--stateful` flag | `Stateful models without beam_idx input are not supported` | Re-export with `--stateful` |
| PowerShell `curl` alias | `Cannot bind parameter 'Headers'` | Use `curl.exe` or `Invoke-RestMethod` |
| JSON escaping in PowerShell | `JSON decode error` | Use `ConvertTo-Json` or save to `.json` file |
| Error handler crash | `TypeError` in global exception handler | Fixed; ensure `JSONResponse` uses keyword args (`status_code=500`) |
| Models not loading | `"loaded_models":[]` in health response | Normal lazy loading; first request triggers compilation (~2-5s delay) |
| Export hangs/fails | `IncompleteRead`, `NameResolutionError` | Install `hf_transfer`, set `$env:HF_HUB_ENABLE_HF_TRANSFER="1"`, use local cache |
| Pagefile too small | `OSError: The paging file is too small (os error 1455)` | Increase Windows pagefile to 16-32 GB or close all other apps |
| NPU not detected | `NPU not in available_devices` | Install Intel NPU driver: <https://www.intel.com/content/www/us/en/download/794636> |

### Troubleshooting Checklist

If things aren't working, check these common pitfalls:

#### Server won't start

- [ ] Python 3.11.x active? `python --version`
- [ ] Virtual environment activated? Prompt shows `(ov-env)`
- [ ] Dependencies installed? `pip list | Select-String -Pattern "openvino"`
- [ ] `app/main.py` exists? `ls app\main.py`

#### Model export fails

- [ ] Using `--stateful` flag?
- [ ] Output dir is positional arg (no `--output`)?
- [ ] Local cache path used (not direct HF download)?
- [ ] Pagefile increased to 16+ GB?
- [ ] `hf_transfer` installed and `$env:HF_HUB_ENABLE_HF_TRANSFER="1"` set?

#### Inference returns 500 error

- [ ] Check server logs for OpenVINO errors
- [ ] Model compiled on first request? Wait 2-5 seconds
- [ ] NPU driver installed? `python -c "from openvino.runtime import Core; print(Core().available_devices)"`
- [ ] Error handler patched in `app/main.py`?

#### curl/PowerShell issues

- [ ] Using `curl.exe` (not PowerShell alias)?
- [ ] JSON built with `ConvertTo-Json` or saved to file?
- [ ] Testing in `cmd.exe` instead of PowerShell?

## ⏱️ Performance Notes

### Memory Requirements

| Model | Params | INT4 Disk | RAM for Export | Pagefile Recommended |
| :--- | :--- | :--- | :--- | :--- |
| Qwen 2.5 3B | 3B | ~1.7 GB | ~8 GB | 16 GB |
| Qwen 2.5 1.5B | 1.5B | ~0.9 GB | ~6 GB | 12 GB |
| Phi-3 Mini | 3.8B | ~2.2 GB | ~10 GB | 20 GB |
| Phi-3.5 Mini | 3.8B | ~2.2 GB | ~10 GB | 20 GB |

### Lazy Loading

Models are **lazy-loaded**. They will not appear in the `"loaded_models"` array of the `/health` endpoint until the *first* inference request is made. The first request will experience a compilation delay of ~2-5 seconds. Subsequent requests will be fast.

## 🤝 Contributing

To add new models, follow the Model Export Guide to export them to INT4 OpenVINO format with the `--stateful` flag, add them to `models.yaml`, and test with the API. Please report any issues or submit PRs for improvements!
