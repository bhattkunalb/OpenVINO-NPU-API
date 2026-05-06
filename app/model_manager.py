"""Thread-safe model loader, in-memory cache, warm-up, and latency tracking.

Design:
  - One LLMPipeline per model, stored in _cache with a per-model Lock.
  - Lazy compilation on first request; warm-up fires after compilation.
  - _global_lock serializes cache mutations; per-model locks guard inference.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import openvino as ov
import openvino_genai as ov_genai

from app.registry import ModelEntry

log = logging.getLogger(__name__)


@dataclass
class CachedModel:
    """Compiled pipeline plus metadata for one registered model."""
    pipeline: Any
    entry: ModelEntry
    lock: threading.Lock = field(default_factory=threading.Lock)
    load_time_ms: float = 0.0
    infer_count: int = 0
    total_infer_ms: float = 0.0

    @property
    def avg_infer_ms(self) -> float:
        """Calculate mean latency over all recorded inferences."""
        return self.total_infer_ms / self.infer_count if self.infer_count else 0.0


class ModelManager:
    """Central model registry, lazy compiler, and thread-safe cache."""

    def __init__(self) -> None:
        self._cache: dict[str, CachedModel] = {}
        self._registry: dict[str, ModelEntry] = {}
        self._global_lock = threading.Lock()

    def register_entries(self, entries: list[ModelEntry]) -> None:
        """Add model definitions to the in-memory registry."""
        for e in entries:
            self._registry[e.name] = e
        log.info("Registered %d model(s).", len(entries))

    def get_cached(self, name: str) -> tuple[CachedModel, bool]:
        """Return (CachedModel, cache_hit). Compiles on first miss."""
        with self._global_lock:
            if name in self._cache:
                return self._cache[name], True
        entry = self._registry.get(name)
        if entry is None:
            raise KeyError(f"Model '{name}' not in registry.")
        compiled = self._load_and_warm(entry)
        with self._global_lock:
            if name not in self._cache:
                self._cache[name] = compiled
                return compiled, False
            return self._cache[name], True

    def record_inference(self, name: str, elapsed_ms: float) -> None:
        """Update latency stats for a model."""
        with self._global_lock:
            cm = self._cache.get(name)
        if cm:
            with cm.lock:
                cm.infer_count += 1
                cm.total_infer_ms += elapsed_ms

    def list_loaded(self) -> list[str]:
        """Return names of all models currently in memory."""
        with self._global_lock:
            return list(self._cache)

    def all_names(self) -> list[str]:
        """Return names of all models defined in the registry."""
        return list(self._registry)

    def get_entry(self, name: str) -> ModelEntry | None:
        """Fetch model metadata from the registry by name."""
        return self._registry.get(name)

    def _load_and_warm(self, entry: ModelEntry) -> CachedModel:
        """Compile and warm up a model on its target device."""
        log.info("[%s] Compiling on %s  path=%s", entry.name, entry.device, entry.path)
        t0 = time.perf_counter()
        if entry.task == "embedding":
            pipeline = self._load_embedding(entry.path, entry.device)
        else:
            pipeline = ov_genai.LLMPipeline(entry.path, entry.device)
        load_ms = (time.perf_counter() - t0) * 1000
        log.info("[%s] Compiled in %.1f ms. Warming up…", entry.name, load_ms)
        self._warmup(pipeline, entry)
        return CachedModel(pipeline=pipeline, entry=entry, load_time_ms=load_ms)

    @staticmethod
    def _load_embedding(model_path: str, device: str) -> Any:
        """Load embedding model via GenAI or raw Core fallback."""
        try:
            return ov_genai.EmbeddingModel(model_path, device)  # type: ignore[attr-defined]
        except (AttributeError, ImportError):
            core = ov.Core()
            xmls = list(Path(model_path).glob("*.xml"))
            if not xmls:
                raise FileNotFoundError(f"No .xml model in '{model_path}'") from None
            return core.compile_model(str(xmls[0]), device)

    @staticmethod
    def _warmup(pipeline: Any, entry: ModelEntry) -> None:
        """Fire one dummy inference to initialize NPU kernels (non-fatal)."""
        try:
            if entry.task == "embedding":
                (getattr(pipeline, "infer", None) or getattr(pipeline, "embed"))(["warmup"])
            else:
                cfg = ov_genai.GenerationConfig()
                cfg.max_new_tokens = 1
                pipeline.generate("warmup", cfg)
            log.info("[%s] Warm-up complete.", entry.name)
        except (RuntimeError, ValueError) as exc:
            log.warning("[%s] Warm-up failed (non-fatal): %s", entry.name, exc)


_MANAGER = ModelManager()


def get_manager() -> ModelManager:
    """Return the process-wide singleton ModelManager."""
    return _MANAGER
