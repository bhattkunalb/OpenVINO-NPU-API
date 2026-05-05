"""Postprocessing hooks: normalise raw model output into API response fields."""

from __future__ import annotations


def strip_prompt_echo(generated: str, prompt: str) -> str:
    """
    Some OV IR models echo the prompt in their output.
    Strip it if present (safe no-op otherwise).
    """
    if generated.startswith(prompt):
        return generated[len(prompt):]
    return generated


def clean_generation(text: str) -> str:
    """Strip common artefacts: trailing EOS tokens, extra whitespace."""
    eos_tokens = ["<|endoftext|>", "</s>", "<|im_end|>", "<end_of_turn>", "<|eot_id|>"]
    for tok in eos_tokens:
        text = text.replace(tok, "")
    return text.strip()


def estimate_token_count(text: str) -> int:
    """
    Coarse token count heuristic (≈4 chars/token).
    OpenVINO GenAI provides real token counts via GenerationResult;
    use those when available and fall back here.
    """
    return max(1, len(text) // 4)
