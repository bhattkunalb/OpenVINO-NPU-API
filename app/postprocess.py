"""Postprocessing: normalize raw model output into clean API text."""

from __future__ import annotations

_EOS_TOKENS = ["<|endoftext|>", "</s>", "<|im_end|>", "<end_of_turn>", "<|eot_id|>"]


def strip_prompt_echo(generated: str, prompt: str) -> str:
    """Remove prompt prefix if echoed by the model."""
    return generated[len(prompt):] if generated.startswith(prompt) else generated


def clean_generation(text: str) -> str:
    """Strip EOS tokens and trailing whitespace."""
    for tok in _EOS_TOKENS:
        text = text.replace(tok, "")
    return text.strip()
