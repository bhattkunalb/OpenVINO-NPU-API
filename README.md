# OpenVINO NPU Inference API

A lightweight, production-ready **OpenAI-compatible inference API** for OpenVINO models running on Intel NPU hardware.

---

## Architecture

```text
app/
├── main.py          # FastAPI app, lifespan, NPU startup validation
├── api.py           # Route handlers, SSE streaming (dispatch only)
├── adapter.py       # OpenAI ↔ internal format translation (isolated)
├── pipeline.py      # preprocess → infer → postprocess contract
├── model_manager.py # Thread-safe loader, cache, warm-up, latency tracking
├── registry.py      # Pydantic config loader and validator
├── schemas.py       # OpenAI-compatible Pydantic request/response models
├── preprocess.py    # Prompt construction and context truncation
├── postprocess.py   # EOS stripping and token count estimation
├── config.py        # Environment variable parsing
└── utils.py         # SSE helpers, thread pool, structured logging
```

## Tech Stack

| Component | Library |
| :--- | :--- |
| Web framework | FastAPI 0.110+, Uvicorn |
| Inference | openvino>=2024.3.0 |
| Generation | openvino-genai>=2024.3.0 |
| Tokenization | openvino-tokenizers>=2024.3.0 |
| Validation | Pydantic>=2.0 |
| Config | PyYAML>=6.0 |

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure models

Edit `models.yaml` (see full schema below). No code changes needed.

### 3. Start the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

> **`--workers 1` is required.** The NPU context is held in-process and cannot be shared across workers.

---

## Environment Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `CONFIG_PATH` | `models.yaml` | Path to model registry YAML |
| `NPU_DEVICE_STRING` | `NPU` | OpenVINO device name |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `OPENVINO_API_THREAD_POOL_SIZE` | `4` | Max inference threads |
| `OPENVINO_API_HOST` | `0.0.0.0` | Bind host |
| `OPENVINO_API_PORT` | `8000` | Bind port |

---

## models.yaml Schema

```yaml
models:
  - name: my-model              # unique identifier (required)
    path: /models/my-model-OV   # path to OV IR dir (required)
    task: chat                  # chat | completion | embedding | vision
    input_type: text            # text | image | tensor
    device: NPU                 # NPU | CPU | GPU
    preprocess_fn: default_genai  # default_genai | custom:<module.fn>
    postprocess_fn: default_genai
    max_tokens: 2048
    context_length: 32768
```

### Adding a new model (zero code changes)

```yaml
# Append to models.yaml:
  - name: phi-3-mini
    path: /models/phi-3-mini-OV
    task: chat
    input_type: text
    device: NPU
    preprocess_fn: default_genai
    postprocess_fn: default_genai
    max_tokens: 2048
    context_length: 4096
```

Restart the server. The model is registered immediately.

---

## API Reference

### GET /health

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "loaded_models": ["qwen3-2.5b"],
  "registered_models": ["qwen3-2.5b", "qwen2.5-1.5b", "gemma4-2b", "bge-m3"]
}
```

---

### GET /v1/models

```bash
curl http://localhost:8000/v1/models
```

```json
{
  "object": "list",
  "data": [
    {"id": "qwen3-2.5b", "object": "model", "created": 1700000000, "owned_by": "local"},
    {"id": "qwen2.5-1.5b", "object": "model", "created": 1700000000, "owned_by": "local"}
  ]
}
```

---

### POST /v1/chat/completions

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-2.5b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is 2+2?"}
    ],
    "max_tokens": 128,
    "temperature": 0.7
  }'
```

```json
{
  "id": "chatcmpl-a1b2c3d4",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "qwen3-2.5b",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "2 + 2 = 4."},
      "finish_reason": "stop",
      "logprobs": null
    }
  ],
  "usage": {"prompt_tokens": 28, "completion_tokens": 9, "total_tokens": 37}
}
```

---

### POST /v1/chat/completions (SSE Streaming)

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-2.5b",
    "messages": [{"role": "user", "content": "Count to 5."}],
    "stream": true
  }'
```

Expected stream:

```text
data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","created":1700000000,"model":"qwen3-2.5b","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","created":1700000000,"model":"qwen3-2.5b","choices":[{"index":0,"delta":{"content":"1"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","created":1700000000,"model":"qwen3-2.5b","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

#### JavaScript EventSource example

```javascript
const es = new EventSource("/v1/chat/completions");
// Note: EventSource is GET-only; use fetch with ReadableStream for POST:

const resp = await fetch("/v1/chat/completions", {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify({
    model: "qwen3-2.5b",
    messages: [{role: "user", content: "Hello"}],
    stream: true
  })
});

const reader = resp.body.getReader();
const decoder = new TextDecoder();
while (true) {
  const {done, value} = await reader.read();
  if (done) break;
  const lines = decoder.decode(value).split("\n");
  for (const line of lines) {
    if (!line.startsWith("data: ") || line === "data: [DONE]") continue;
    const chunk = JSON.parse(line.slice(6));
    process.stdout.write(chunk.choices[0].delta.content ?? "");
  }
}
```

---

### POST /v1/responses

```bash
curl http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-2.5b", "input": "Summarize the Eiffel Tower in one sentence."}'
```

```json
{
  "id": "resp-a1b2c3",
  "object": "response",
  "created_at": 1700000000,
  "model": "qwen3-2.5b",
  "output": [
    {
      "type": "message",
      "role": "assistant",
      "content": [{"type": "output_text", "text": "The Eiffel Tower is an iconic iron lattice tower in Paris."}]
    }
  ],
  "usage": {"prompt_tokens": 12, "completion_tokens": 14, "total_tokens": 26}
}
```

---

### POST /v1/embeddings

Only available when an embedding model (task: embedding) is registered.

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "bge-m3", "input": ["Hello world", "OpenVINO rocks"]}'
```

```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [0.021, -0.034, ...]},
    {"object": "embedding", "index": 1, "embedding": [0.018, -0.029, ...]}
  ],
  "model": "bge-m3",
  "usage": {"prompt_tokens": 6, "completion_tokens": 0, "total_tokens": 6}
}
```

---

## NPU Assumptions & OpenVINO Plugin Requirements

| Assumption | Detail |
| :--- | :--- |
| **No CPU fallback** | Service raises `RuntimeError` at startup if NPU plugin is absent. |
| **Single worker** | NPU context is in-process. Always run with `--workers 1`. |
| **Model format** | OV IR (`.xml` + `.bin`) or GenAI-compatible directory. |
| **Batch size** | Fixed at 1. No dynamic batching. |
| **Plugin package** | `openvino-intel-npu` must be installed alongside `openvino`. |
| **BIOS/firmware** | Intel NPU must be enabled in BIOS. Driver: `intel-npu-driver`. |

---

## Docker

```dockerfile
# See Dockerfile in repo root
docker build -t openvino-npu-api .
docker run --rm \
  --device /dev/accel \
  -v /your/models:/models:ro \
  -e CONFIG_PATH=models.yaml \
  -p 8000:8000 \
  openvino-npu-api
```

---

## Log Format

Every request emits a structured log line:

```text
2025-01-01T12:00:00 INFO     app.utils │ request_id=abc123 model=qwen3-2.5b device=NPU load_ms=0.1 infer_ms=842.3 total_ms=843.0 cache_hit=True status=ok
```
