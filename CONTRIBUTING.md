# Contributing to OpenVINO NPU API

Thank you for your interest in improving this project! Since this repository targets specific hardware (Intel NPU) and uses a custom inference stack, please follow these guidelines.

## 🛠 Development Environment

- **Python**: Use Python 3.11.x.
- **Hardware**: You should have access to an Intel NPU (Meteor Lake, Arrow Lake, or Lunar Lake) to verify changes.
- **Dependencies**: Install the core requirements:

  ```bash
  pip install -r requirements.txt
  ```

## 📦 Adding New Models

We prioritize stability by using pre-built OpenVINO models. To contribute a new model:

1. **Verify Source**: Ensure the model is available on Hugging Face in OpenVINO format (preferably in a collection like [llms-optimized-for-npu](https://huggingface.co/collections/OpenVINO/llms-optimized-for-npu)).
2. **Update download_prebuilt.py**: Add the new model ID and alias to the mapping in `scripts/download_prebuilt.py`.
3. **Registry Entry**: Add a commented-out template for the model in `models.yaml`.
4. **Test Alignment**: Verify that the model's `context_length` and `max_prompt_len` are correctly documented.

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
