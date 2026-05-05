# OpenVINO NPU Inference API

> 🚀 Superlight OpenAI-compatible API for OpenVINO models on Intel NPU.
> Export models with `optimum-cli`, configure via `models.yaml`, run with `uvicorn`.
> 💻 **NPU Hardware Requirements**: Intel Meteor Lake (Core Ultra), Arrow Lake, or Lunar Lake CPUs with NPU enabled in BIOS.
> Discrete NPU (e.g., Intel® NPU Acceleration Library) also supported.
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

| Step | Command |
| :--- | :--- |
| Install | `pip install -r requirements.txt` |
| Export model | `optimum-cli export openvino --model Qwen/Qwen3-2B --weight-format int4 --trust-remote-code ./models/qwen3-2b-ov` |
| Configure | Edit `models.yaml` with your model path |
| Run | `uvicorn app.main:app --host 0.0.0.0 --port 4647 --workers 1` |
| Test | `curl http://localhost:4647/health` |

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
| `OPENVINO_API_PORT` | `4647` | Bind port |
| `OPENVINO_API_KEY` | `None` | Optional bearer token for all endpoints |

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
curl http://localhost:4647/health
```

```json
{
  "status": "ok",
  "loaded_models": ["qwen3-2.5b"],
  "registered_models": ["qwen3-2.5b", "qwen2.5-1.5b", "gemma4-2b", "bge-m3"]
}
```

### Checking Model Status

To see which models are correctly downloaded and registered by the API, use:

```bash
# Get full registry list
curl http://localhost:4647/v1/models | jq .data[].id

# Check health and currently loaded (warm) models
curl http://localhost:4647/health
```

---

## Client Integration & API Keys

By default, the API is open. To secure it, set the `OPENVINO_API_KEY` environment variable. All models share this key.

### OpenAI Style (Python)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:4647/v1",
    api_key="your-set-key-here" # or "anything" if not set
)

response = client.chat.completions.create(
    model="qwen3-2.5b",
    messages=[{"role": "user", "content": "Hello"}]
)
```

### OpenClaw / OpenJarvis Config

Configure these tools by pointing them to the local endpoint:

- **API Base**: `http://localhost:4647/v1`
- **API Key**: `sk-local-npu` (or whatever you set in `OPENVINO_API_KEY`)
- **Models**: Use the names from your `models.yaml` (e.g., `qwen3-2.5b`, `gemma4-2b`)

---

### GET /v1/models

```bash
curl http://localhost:4647/v1/models
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
curl http://localhost:4647/v1/chat/completions \
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
curl -N http://localhost:4647/v1/chat/completions \
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
curl http://localhost:4647/v1/responses \
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
curl http://localhost:4647/v1/embeddings \
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

## 🔄 Export Models to OpenVINO IR

This service requires models in OpenVINO Intermediate Representation (IR) format (`.xml` + `.bin`) or Hugging Face paths compatible with `openvino-genai`.

### Prerequisites

```bash
pip install -U "optimum[openvino]" nncf openvino-tokenizers
```

> 💡 **NPU Optimization**: Always use `--weight-format int4` for 2-4x memory reduction with minimal accuracy loss. Include `--trust-remote-code` for Qwen/Gemma architectures.

### ✅ Qwen 3.5 2B (Chat/Completion)

```bash
optimum-cli export openvino \
  --model Qwen/Qwen3-2B \
  --weight-format int4 \
  --trust-remote-code \
  --task text-generation-with-past \
  ./models/qwen3-2b-ov
```

### ✅ Qwen 2.5 1.5B (Worker/Utility)

```bash
optimum-cli export openvino \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --weight-format int4 \
  --trust-remote-code \
  --task text-generation-with-past \
  ./models/qwen2.5-1.5b-ov
```

### ✅ Gemma 4 2B (Verifier/Critique)

```bash
optimum-cli export openvino \
  --model google/gemma-2-2b-it \
  --weight-format int4 \
  --trust-remote-code \
  --task text-generation-with-past \
  ./models/gemma4-2b-ov
```

> ⚠️ **Gemma Note**: Use `gemma-2-2b-it` (Instruct-Tuned). The base `gemma-2-2b` lacks chat templates.

### 📋 Example `models.yaml` Entries

```yaml
models:
  - name: qwen3-2.5b
    path: ./models/qwen3-2b-ov
    task: chat
    input_type: text
    device: NPU
    preprocess_fn: default_genai
    postprocess_fn: default_genai
    max_tokens: 2048
    context_length: 32768

  - name: qwen2.5-1.5b
    path: ./models/qwen2.5-1.5b-ov
    task: completion
    input_type: text
    device: NPU
    preprocess_fn: default_genai
    postprocess_fn: default_genai
    max_tokens: 1024

  - name: gemma4-2b
    path: ./models/gemma4-2b-ov
    task: chat
    input_type: text
    device: NPU
    preprocess_fn: default_genai
    postprocess_fn: default_genai
    max_tokens: 512
```

### 🧪 Quick Validation

```bash
# Test model loads in OpenVINO
python -c "
from openvino.runtime import Core
core = Core()
model = core.read_model('./models/qwen3-2b-ov/openvino_model.xml')
compiled = core.compile_model(model, 'NPU')
print('✓ Model compiled successfully')
"
```

### 🚨 Troubleshooting Export

| Issue | Solution |
| :--- | :--- |
| `--trust-remote-code` error | Update: `pip install -U "optimum[openvino]"` |
| Out of memory during export | Add `--per-channel`: `--weight-format int4 --per-channel` |
| Missing tokenizer files | Ensure `tokenizer_config.json` is present in model dir |
| NPU compilation fails at runtime | Install plugin: `pip install openvino-intel-npu` |
| Chat template not applied | Verify `chat_template` exists in `tokenizer_config.json` |

> 🔐 **Security**: Always verify model hashes and sources before using pre-exported weights.


---

## 📡 Streaming (Server-Sent Events)

This service supports real-time token streaming via SSE for `/v1/chat/completions` and `/v1/responses`.

### Request
```bash
curl -N -X POST http://localhost:4647/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-2.5b",
    "messages": [{"role": "user", "content": "Explain SSE"}],
    "stream": true
  }'
```

### Response Format (Exact)
```text
Content-Type: text/event-stream

data: {"id":"uuid","object":"chat.completion.chunk","created":1234567890,"model":"qwen3-2.5b","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"uuid","object":"chat.completion.chunk","created":1234567890,"model":"qwen3-2.5b","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}

data:  [DONE]
```

### Client Examples

**JavaScript (EventSource)**
```javascript
const eventSource = new EventSource(
  "http://localhost:4647/v1/chat/completions",
  {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      model: "qwen3-2.5b",
      messages: [{role: "user", content: "Hi"}],
      stream: true
    })
  }
);

eventSource.onmessage = (e) => {
  if (e.data === " [DONE]") {
    eventSource.close();
    return;
  }
  const chunk = JSON.parse(e.data);
  processToken(chunk.choices[0].delta.content);
};
```

**Python (requests)**
```python
import requests, json

response = requests.post(
    "http://localhost:4647/v1/chat/completions",
    json={"model": "qwen3-2.5b", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
    stream=True
)

for line in response.iter_lines():
    if line:
        decoded = line.decode("utf-8")
        if decoded == " [DONE]":
            break
        if decoded.startswith("data: "):
            chunk = json.loads(decoded[6:])
            print(chunk["choices"][0]["delta"]["content"], end="", flush=True)
```

### ⚠️ Streaming Notes
- **No buffering**: Tokens are yielded as soon as generated by the NPU
- **Client disconnect**: Server gracefully stops generation if client closes connection
- **Error handling**: Errors mid-stream return a final `{"error": {...}}` chunk before `[DONE]`
- **Rate**: Streaming latency depends on NPU token generation speed (~10-30 tokens/sec for 2B models)

---

## 🎯 OpenAI API Compatibility

This service implements a strict subset of the OpenAI API for maximum client compatibility.

### ✅ Supported Fields & Behavior

| Field | Supported | Notes |
|-------|-----------|-------|
| `model` | ✅ | Must match a registered model name in `models.yaml` |
| `messages` | ✅ | Chat format: `[{"role": "user\|assistant\|system", "content": "..."}]` |
| `prompt` | ✅ | For `/v1/responses` completion-style requests |
| `stream` | ✅ | Boolean; enables SSE streaming when `true` |
| `max_tokens` | ✅ | Limits generation length (approximate) |
| `temperature` | ⚠️ | Accepted but may be ignored depending on model pipeline |
| `stop` | ⚠️ | Accepted but not enforced at API layer |
| `usage` | ✅ | Returned with approximate token counts |
| `error` format | ✅ | Matches OpenAI: `{"error": {"message", "type", "param", "code"}}` |

### ❌ Not Supported (By Design)
- `functions` / `function_call` (no tool calling at API layer)
- `response_format` (JSON mode) — handle in client/postprocessor
- `logprobs`, `top_logprobs` — not exposed by OpenVINO GenAI
- `n` > 1 (multiple completions) — batch size is always 1
- `user` field — no user tracking at infrastructure layer

### 🔄 Response Schema (Guaranteed)
All successful responses include:
```json
{
  "id": "uuid4-string",
  "object": "chat.completion" | "text_completion" | "embedding",
  "created": 1234567890,
  "model": "registered-model-name",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "..."},  // for chat
      // OR "text": "..." for completion
      "finish_reason": "stop" | "length" | null
    }
  ],
  "usage": {
    "prompt_tokens": 123,
    "completion_tokens": 456,
    "total_tokens": 579
  }
}
```

### 🧪 Compatibility Test
```bash
# Test with official OpenAI Python client
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:4647/v1",
    api_key="sk-local"  # or any string; auth is optional
)

response = client.chat.completions.create(
    model="qwen3-2.5b",
    messages=[{"role": "user", "content": "Hello"}]
)
print(response.choices[0].message.content)  # Should work identically to OpenAI API
```

> 💡 **Tip**: If a client fails, check that `model` exactly matches a name in `models.yaml` — this is the most common integration issue.

---

## 🔒 Model Manager Stability

The model manager ensures reliable, thread-safe inference under concurrent load.

### Concurrency Model
- **Single Uvicorn worker** (`--workers 1`): Required to avoid NPU resource contention
- **Per-model locking**: Each cached model has a `threading.Lock` to serialize inference calls
- **Async wrapping**: Blocking OpenVINO calls are wrapped in `asyncio.to_thread()` to avoid event loop blocking

### Cache Behavior
| Behavior | Details |
|----------|---------|
| **Lazy loading** | Models compile on first request, not at startup |
| **In-memory cache** | Compiled pipelines stored in dict; no eviction (RAM-bound) |
| **Warm-up** | Dummy inference (`generate("")`) runs post-compilation to initialize NPU kernels |
| **Cache key** | Model name from `models.yaml` (case-sensitive) |

### Memory Expectations
| Model | INT4 Size | RAM Usage (Runtime) |
|-------|-----------|---------------------|
| Qwen 3.5 2B | ~1.2 GB | ~2.5 GB (weights + KV cache) |
| Qwen 2.5 1.5B | ~0.9 GB | ~2.0 GB |
| Gemma 4 2B | ~1.2 GB | ~2.5 GB |

> 💡 **Tip**: Total RAM usage ≈ 2× model size due to KV cache and runtime overhead. Ensure 8 GB+ RAM for 2B models.

### Failure Recovery
| Scenario | Behavior |
|----------|----------|
| Model load fails | Returns `500 {"error": {"message": "Model load failed: ..."}}`; server continues running |
| NPU unavailable at runtime | Returns `503 {"error": {"message": "NPU device busy or unavailable"}}` |
| Concurrent requests to same model | Serialized via per-model lock; no corruption or race conditions |
| Client disconnect mid-inference | Generation stops gracefully; no resource leak |

### 🧪 Stability Test
```bash
# Concurrent requests to same model (should not crash)
for i in {1..5}; do
  curl -s -X POST http://localhost:4647/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen3-2.5b","messages":[{"role":"user","content":"Test '$i'"}]}' &
done
wait
echo "✓ All concurrent requests completed"
```

### ⚠️ Production Notes
- **Do not scale workers**: `--workers > 1` may cause NPU contention or out-of-memory errors
- **Monitor RAM**: Use `htop` or `psutil` to ensure sufficient headroom for KV cache growth
- **Warm-up matters**: First request after server start will be slower (~2-5s) due to compilation + warm-up
- **Model reload**: Changing `models.yaml` requires server restart (no hot-reload by design)


---

## Docker

```dockerfile
# See Dockerfile in repo root
docker build -t openvino-npu-api .
docker run --rm \
  --device /dev/accel \
  -v /your/models:/models:ro \
  -e CONFIG_PATH=models.yaml \
  -p 4647:4647 \
  openvino-npu-api
```

---

## Log Format

Every request emits a structured log line:

```text
2025-01-01T12:00:00 INFO     app.utils │ request_id=abc123 model=qwen3-2.5b device=NPU load_ms=0.1 infer_ms=842.3 total_ms=843.0 cache_hit=True status=ok
```
