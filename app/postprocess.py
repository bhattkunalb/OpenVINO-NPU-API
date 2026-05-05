"""Postprocessing: normalize raw model output into clean API text."""

from __future__ import annotations

_EOS_TOKENS = ["<|endoftext|>", "</s>", "<|im_end|>", "<end_of_turn>", "<|eot_id|>"]


def strip_prompt_echo(generated: str, prompt: str) -> str:
    """Remove prompt prefix if echoed by the model."""
    return generated[len(prompt):] if generated.startswith(prompt) else generated


def clean_generation(text: str) -> str:
    """Strip chat delimiters and extra whitespace from model output."""
    t = text.strip()
    for d in ["<|im_start|>", "<|im_end|>", "</s>", "<s>", "[INST]", "[/INST]"] + _EOS_TOKENS:
        t = t.replace(d, "")
    return t.strip()


def enforce_stop_strings(text: str, stop: list[str]) -> str:
    """Truncate text at the first occurrence of any stop string."""
    earliest = len(text)
    for s in stop:
        idx = text.find(s)
        if idx != -1 and idx < earliest:
            earliest = idx
    return text[:earliest]


class StreamStopManager:
    """Stateful buffer for identifying stop strings across streaming chunks."""

    def __init__(self, stop: list[str] | None):
        self.stop = stop or []
        self.buffer = ""
        self.stopped = False

    def process_token(self, token: str) -> str | None:
        """Add token to buffer and return 'safe' text that cannot be a stop prefix."""
        if self.stopped:
            return None
        self.buffer += token
        
        for s in self.stop:
            if s in self.buffer:
                self.stopped = True
                return self.buffer[:self.buffer.find(s)]
        
        # Keep enough in buffer to cover partial matches of any stop string
        max_stop_len = max((len(s) for s in self.stop), default=0)
        if max_stop_len <= 1:
            out = self.buffer
            self.buffer = ""
            return out
            
        safe_len = len(self.buffer) - (max_stop_len - 1)
        if safe_len > 0:
            out = self.buffer[:safe_len]
            self.buffer = self.buffer[safe_len:]
            return out
        return ""

    def flush(self) -> str:
        """Return remaining buffer if not stopped."""
        if self.stopped:
            return ""
        out = self.buffer
        self.buffer = ""
        return out
