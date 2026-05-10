"""Test script to bypass OpenClaw and test NPU API directly."""
from openai import OpenAI  # pylint: disable=import-error

client = OpenAI(
    base_url="http://localhost:4647/v1",
    api_key="sk-local-npu"
)

response = client.chat.completions.create(
    model="qwen2.5-1.5b-npu",
    messages=[{"role": "user", "content": "Write a haiku about coding."}],
    max_tokens=50,
    timeout=120  # Explicit timeout for NPU compilation
)

print(response.choices[0].message.content)
