# OpenVINO NPU API 🧠⚡

Local, OpenAI-compatible API for running LLMs on Intel NPUs using pre-built, optimized models. No model export required.

## ⚡ Quick Start (5 Minutes)

### Prerequisites

- ✅ Windows 11 (23H2 or later)
- ✅ Intel Core Ultra processor with NPU
- ✅ **Python 3.11.x** (3.12+ and 3.14 are NOT supported)
- ✅ Intel NPU Driver: <https://www.intel.com/content/www/us/en/download/794636>

### Step 1: Clone & Setup

```powershell
# Create work directory (outside OneDrive to avoid sync locks)
mkdir C:\ov-npu
cd C:\ov-npu

# Clone repository
git clone https://github.com/bhattkunalb/OpenVINO-NPU-API.git .

# Create & activate virtual environment (Python 3.11.x required)
python -m venv ov-env
ov-env\Scripts\Activate.ps1

# If you get execution policy error, run this first:
# Set-ExecutionPolicy -Scope CurrentUser RemoteSigned -Force

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2: Download a Pre-Built Model

```powershell
# Enable faster downloads (optional but recommended)
$env:HF_HUB_ENABLE_HF_TRANSFER = "1"

# Download Qwen 2.5 1.5B INT4 for NPU (~1.5 GB)
python scripts/download_prebuilt.py qwen2.5-1.5b

# Wait for: "✅ Download complete: ./models/qwen2.5-1.5b-npu"
```

### Step 3: Start the Server

```powershell
# ⚠️ CRITICAL: --workers 1 is required for NPU stability
python -m uvicorn app.main:app --host 0.0.0.0 --port 4647 --workers 1
```

✅ Expected output:

```text
INFO: app.main | NPU confirmed: 'NPU'
INFO: app.model_manager | [qwen2.5-1.5b-npu] Compiling on NPU...
INFO: Uvicorn running on http://0.0.0.0:4647
```

### Step 4: Test Inference (New PowerShell Window)

```powershell
# Build request (PowerShell-safe JSON)
$body = @{
    model = "qwen2.5-1.5b-npu"
    messages = @(@{ role = "user"; content = "Say hello in one word." })
    max_tokens = 10
} | ConvertTo-Json -Depth 10 -Compress

# Send request
$response = Invoke-RestMethod -Uri "http://localhost:4647/v1/chat/completions" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body

# Show AI response
$response.choices[0].message.content
```

✅ Expected: `Hello` or `Hi` within ~5-10 seconds.

---

## 📦 Available Pre-Built Models

All models are INT4 quantized, pre-converted to OpenVINO, and tested on Intel Core Ultra NPUs.

| Alias | Model ID | Size | Use Case | Download Command |
| --- | --- | --- | --- | --- |
| `qwen2.5-1.5b` | `OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov` | ~1.5 GB | Balanced speed/quality | `python scripts/download_prebuilt.py qwen2.5-1.5b` |
| `qwen2.5-3b` | `OpenVINO/Qwen2.5-3B-Instruct-int4-ov` | ~3.0 GB | Complex reasoning | `python scripts/download_prebuilt.py qwen2.5-3b` |
| `phi-3-mini` | `OpenVINO/Phi-3-mini-4k-instruct-int4-ov` | ~2.2 GB | Fast verification | `python scripts/download_prebuilt.py phi-3-mini` |
| `llama-3.2-1b` | `OpenVINO/Llama-3.2-1B-Instruct-int4-ov` | ~1.0 GB | Lightweight tasks | `python scripts/download_prebuilt.py llama-3.2-1b` |

> 💡 Models are downloaded to `./models/<name>-npu/` by default.

---

## ⚙️ Configuration: models.yaml

Edit `models.yaml` to register models. **Indentation is critical** (use spaces, not tabs):

```yaml
models:
  - name: qwen2.5-1.5b-npu          # Must match API calls exactly
    path: ./models/qwen2.5-1.5b-npu  # Relative to project root
    task: chat
    input_type: text
    device: NPU                      # Options: NPU, CPU, GPU
    preprocess_fn: default_genai
    postprocess_fn: default_genai
    max_tokens: 1024
    context_length: 32768
```

### YAML Indentation Rules

```yaml
models:              # 0 spaces (root)
  - name: foo        # 2 spaces + dash (list item)
    path: ./foo      # 4 spaces (property)
    device: NPU      # 4 spaces (same level)
  - name: bar        # 2 spaces + dash (next item)
```

> ⚠️ Never mix tabs and spaces. Use an editor with YAML linting (VS Code recommended).

---

## 🔧 Troubleshooting

### Server won't start: YAML parsing error

```text
yaml.parser.ParserError: while parsing a block collection
```

**Fix:** Check `models.yaml` indentation. All list items must start with exactly 2 spaces + `-`. Use the copy-paste template above.

### Model not found / 404

```json
{"error": {"message": "Model 'xyz' not found"}}
```

**Fix:** 

1. Ensure model is downloaded: `ls ./models/`
2. Ensure `name` in API call matches `models.yaml` entry exactly
3. Restart server after editing `models.yaml`

### NPU not detected

```python
python -c "import openvino; print(openvino.Core().available_devices)"
# Output: ['CPU', 'GPU']  # Missing 'NPU'
```

**Fix:** Install Intel NPU driver from <https://www.intel.com/content/www/us/en/download/794636> and restart PC.

### First request is slow (~10-15s)

**Normal:** NPU compilation happens on first load. Subsequent requests are ~0.5-2s.

### Request fails with "beam_idx" or "StatefulToStateless" error

**Cause:** You may have used a custom-exported model (not pre-built).

**Fix:** Delete the custom model folder and download a pre-built one:

```powershell
Remove-Item ./models/your-custom-model -Recurse -Force
python scripts/download_prebuilt.py qwen2.5-1.5b
```

### Out of memory on NPU

**Fix:** 

- Close other applications
- Try a smaller model (`llama-3.2-1b-npu`)
- Ensure no other NPU processes are running

### PowerShell `curl` alias issues

**Fix:** Use `curl.exe` (with `.exe`) or `Invoke-RestMethod` as shown in testing examples.

---

## 🚀 Advanced Usage

### Streaming Responses

```powershell
$body = @{
    model = "qwen2.5-1.5b-npu"
    messages = @(@{ role = "user"; content = "Tell me a joke." })
    max_tokens = 100
    stream = $true
} | ConvertTo-Json -Depth 10 -Compress

# Note: Streaming requires custom client; see API docs
```

### Using Multiple Models

1. Download additional models:

   ```powershell
   python scripts/download_prebuilt.py phi-3-mini
   python scripts/download_prebuilt.py llama-3.2-1b
   ```

2. Uncomment their entries in `models.yaml`
3. Restart server
4. Switch models by changing `"model": "..."` in API calls

### Remote Access (Local Network)

```powershell
# Find your IP
ipconfig | Select-String "IPv4"

# Start server bound to that IP
python -m uvicorn app.main:app --host 192.168.1.XX --port 4647 --workers 1
```

> ⚠️ Only use on trusted networks. Add authentication for public exposure.

---

## 📊 Performance Expectations

| Model | Compile Time | First Inference | Subsequent | Tokens/sec (NPU) |
| --- | --- | --- | --- | --- |
| Llama 3.2 1B | ~8s | ~3s | ~0.5s | ~25-40 |
| Phi-3 Mini | ~10s | ~4s | ~0.8s | ~20-35 |
| Qwen 2.5 1.5B | ~12s | ~5s | ~1s | ~15-25 |
| Qwen 2.5 3B | ~18s | ~8s | ~1.5s | ~10-18 |

> 💡 Keep server running to avoid recompilation. NPU performance varies by workload and driver version.

---

## 🛠️ Development Notes

### Python Version Requirement

- **Required:** Python 3.11.x
- **Not Supported:** Python 3.12+, 3.14 (OpenVINO tooling incompatibility)
- Check: `python --version`

### Virtual Environment

- Always activate before running commands: `ov-env\Scripts\Activate.ps1`
- Verify: Prompt shows `(ov-env)` and `python -c "import sys; print(sys.prefix)"` contains `ov-env`

### Dependencies

- Installed via `requirements.txt`
- Key packages: `openvino>=2026.1.0`, `openvino-genai`, `fastapi`, `uvicorn`

### Code Fixes Applied

- ✅ OpenVINO imports updated: `import openvino as ov` (not `from openvino.runtime import`)
- ✅ Error handler fixed: `JSONResponse(status_code=500, content=...)`
- ✅ YAML loading with proper error handling

---

## 📚 Resources

- Intel NPU Driver: <https://www.intel.com/content/www/us/en/download/794636>
- Pre-Built Models Collection: <https://huggingface.co/collections/OpenVINO/llms-optimized-for-npu>
- OpenVINO Documentation: <https://docs.openvino.ai>
- Hugging Face Hub: <https://huggingface.co/docs/hub>

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Test changes with pre-built models only
4. Submit a pull request

> ⚠️ Do not add custom export workflows. This project is intentionally pre-built-models-only for reliability.

---

## 📄 License

MIT License. See LICENSE file for details.
