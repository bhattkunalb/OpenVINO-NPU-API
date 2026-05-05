# OpenVINO NPU Inference API

A superlight, production-ready **OpenAI-compatible inference API** for OpenVINO models running on Intel NPU hardware.

---

## Architecture

```
app/
├── main.py          # FastAPI app, lifespan, NPU validation, router mount
├── api.py           # Route handlers, SSE streaming, request/response wiring
├── model_manager.py # Thread-safe loader, LRU-style in-memory cache, warm-up
├── registry.py      # Pydantic config loader from models.yaml / models.json
├── schemas.py       # OpenAI-compatible Pydantic request/response models
├── preprocess.py    # Prompt construction, context truncation hooks
├── postprocess.py   # EOS cleanup, token estimation hooks
├── config.py        # Env-var parsing with typed constants
└── utils.py         # Shared: thread pool, SSE helpers, structured logging
models.yaml          # Model registry – add models here, zero code changes
requirements.txt
Dockerfile
```

---

## Startup Instructions

### 1. Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| openvino | ≥ 2024.3.0 |
| openvino-genai | ≥ 2024.3.0 |
| Intel NPU driver | Platform-specific (see below) |

**Intel NPU driver installation:**
- **Windows**: Install the [Intel NPU Driver for Windows](https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html)
- **Linux**: Install `intel-driver-compiler-npu` and `intel-fw-npu` from the Intel NPU Linux driver repo

Verify OpenVINO can see the NPU:
```python
import openvino as ov
print(ov.Core().available_devices)  # must include "NPU"
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure models

Edit `models.yaml` – set the `path` fields to your local OV model directories:
```yaml
- name: qwen3-2.5b
  path: /models/Qwen3-2.5B-OV      # directory with OV IR files + tokenizer
  task_type: chat
  device_preference: NPU
```

### 4. Start the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

> **Critical**: `--workers 1` is mandatory. NPU models are held in-process; multiple workers would each try to claim NPU context independently.

### 5. Environment variables

| Variable | Default | Description |
|---|---|---|
| `OPENVINO_API_MODEL_CONFIG` | `models.yaml` | Path to registry file |
| `OPENVINO_API_DEVICE` | `NPU` | OpenVINO device string |
| `OPENVINO_API_LOG_LEVEL` | `INFO` | Python logging level |
| `OPENVINO_API_THREAD_POOL_SIZE` | `4` | Inference thread pool workers |
| `OPENVINO_API_HOST` | `0.0.0.0` | Bind host |
| `OPENVINO_API_PORT` | `8000` | Bind port |

---

## API Reference

### Health check
```
GET /health
```

### List models
```
GET /v1/models
```

### Chat completions (OpenAI-compatible)
```
POST /v1/chat/completions
```

### Responses API
```
POST /v1/responses
```

### Embeddings (only active if an embedding model is registered)
```
POST /v1/embeddings
```

---

## Example `curl` Requests

### Chat completion (non-streaming)
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-2.5b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain what Intel NPU is in two sentences."}
    ],
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

### Chat completion (streaming)
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "model": "qwen2.5-1.5b",
    "messages": [{"role": "user", "content": "Count from 1 to 5."}],
    "stream": true,
    "max_tokens": 128
  }'
```

### Responses API
```bash
curl http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4-2b",
    "input": "What is the capital of France?",
    "max_output_tokens": 64
  }'
```

### Embeddings
```bash
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-m3",
    "input": ["Hello world", "OpenVINO on NPU is fast."]
  }'
```

---

## SSE Streaming Consumption

### Python (httpx)
```python
import httpx, json

with httpx.Client(timeout=120) as client:
    with client.stream(
        "POST",
        "http://localhost:8000/v1/chat/completions",
        json={
            "model": "qwen3-2.5b",
            "messages": [{"role": "user", "content": "Write a haiku about silicon."}],
            "stream": True,
        },
    ) as resp:
        for line in resp.iter_lines():
            if line.startswith("data: "):
                payload = line[6:]
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                delta = chunk["choices"][0]["delta"].get("content", "")
                print(delta, end="", flush=True)
```

### JavaScript (fetch / browser)
```js
const resp = await fetch("http://localhost:8000/v1/chat/completions", {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify({
    model: "qwen3-2.5b",
    messages: [{role: "user", content: "Hello!"}],
    stream: true,
  }),
});

const reader = resp.body.getReader();
const decoder = new TextDecoder();
while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  for (const line of decoder.decode(value).split("\n")) {
    if (line.startsWith("data: ") && line !== "data: [DONE]") {
      const chunk = JSON.parse(line.slice(6));
      process.stdout.write(chunk.choices[0].delta?.content ?? "");
    }
  }
}
```

---

## Adding a New Model (Zero Code Changes)

1. Export your model to OpenVINO IR format:
   ```bash
   # Using optimum-intel
   optimum-cli export openvino --model your-hf-repo --task text-generation-with-past /models/your-model-OV

   # Or using openvino-genai converter
   python -m openvino_genai.convert --model your-hf-repo --output /models/your-model-OV
   ```

2. Add an entry to `models.yaml`:
   ```yaml
   - name: my-new-model
     path: /models/your-model-OV
     task_type: chat        # chat | response | embedding
     device_preference: NPU
     max_tokens: 2048
     context_window: 8192
   ```

3. Restart the server. The model compiles and warms up on first request.

---

## Docker Deployment

```bash
# Build
docker build -t openvino-npu-api .

# Run (mount model dir and NPU device)
docker run -d \
  --device /dev/accel \
  -v /your/models:/models:ro \
  -e OPENVINO_API_MODEL_CONFIG=models.yaml \
  -p 8000:8000 \
  openvino-npu-api
```

---

## NPU Assumptions & OpenVINO Plugin Requirements

| Assumption | Detail |
|---|---|
| **No CPU fallback** | Service raises `RuntimeError` at startup if NPU plugin is absent. |
| **Single process** | NPU state is in-process. Run `--workers 1` always. |
| **Model format** | Models must be in OpenVINO IR (`.xml` + `.bin`) or GenAI compatible directory. |
| **Batch size** | Batch size = 1. No dynamic batching. |

---

## Log Format

Every request emits a structured log line:
```
2025-01-01T12:00:00 INFO     app.utils │ request_id=abc123 model=qwen3-2.5b device=NPU
  load_ms=0.1 infer_ms=842.3 total_ms=843.0 cache_hit=True status=ok
```