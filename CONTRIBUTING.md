# Contributing to OpenVINO NPU API

Thank you for your interest in improving this project! Since this repository targets specific hardware (Intel NPU) and uses a custom inference stack, please follow these guidelines.

## 🛠 Development Environment

- **Python**: Use Python 3.11.x.
- **Hardware**: You should have access to an Intel NPU (Meteor Lake, Arrow Lake, or Lunar Lake) to verify changes.
- **Dependencies**: Install the core requirements:
  ```bash
  pip install -r requirements.txt
  ```

## 📦 Exporting Models

All models must be compatible with `openvino_genai.LLMPipeline`. 

1. **Always use stateful models**: Use the `--task text-generation-with-past` flag.
2. **Preferred Export**: Use the provided helper script:
   ```bash
   python scripts/export_genai.py <model_id> <output_dir>
   ```
3. **Verification**: Always verify that `beam_idx` is present in the `openvino_model.xml` file.

## 🧪 Testing

1. **Unit Tests**: Ensure any changes to `app/preprocess.py` or `app/adapter.py` do not break the ChatML formatting.
2. **NPU Validation**: Before submitting a PR, verify that the model compiles successfully on the NPU:
   - Start the server with `--workers 1`.
   - Send a test request and monitor the logs for `[NPU] Compiled in ... ms`.

## 📝 Documentation

- Update `README.md` if you add new features or configuration options.
- If you find a new edge case or error, please add it to the **Troubleshooting** table in the README.

## ⚖ Code Style

- Use **PEP 8** for Python code.
- Ensure all `subprocess` calls use `check=False` (or explicit error handling) and are well-documented.
- Maintain **ChatML** as the default formatting for all LLM prompts.
