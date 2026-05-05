"""Postprocessing hooks: normalize raw model output into clean API response text."""

from __future__ import annotations


# Tokens emitted by common models as end-of-sequence markers.
_EOS_TOKENS = ["<|endoftext|>", "</s>", "<|im_end|>", "<end_of_turn>", "<|eot_id|>"]


def strip_prompt_echo(generated: str, prompt: str) -> str:
    """
    Remove prompt prefix from generated text if echoed by the model.
    Safe no-op when the model does not echo.
    """
    if generated.startswith(prompt):
        return generated[len(prompt):]
    return generated


def clean_generation(text: str) -> str:
    """Strip EOS tokens and trailing whitespace from generated text."""
    for tok in _EOS_TOKENS:
        text = text.replace(tok, "")
    return text.strip()


def estimate_token_count(text: str) -> int:
    """
    Coarse token count approximation (≈4 chars/token).
    GenAI GenerationResult provides real counts; use those when available
    and call this only as a fallback for prompt-side estimates.
    """
    return max(1, len(text) // 4)
