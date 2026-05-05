"""
Thread-safe model loader, in-memory cache, warm-up, and latency tracking.

Architecture notes:
- One LLMPipeline (openvino_genai) per registered model, held in _cache.
- A per-model threading.Lock prevents concurrent compilation or inference on
  the same pipeline instance (OV GenAI pipelines are NOT thread-safe).
- Models are compiled lazily on first request, then kept alive for the
  process lifetime.
- Warm-up fires a single dummy generate() after compilation so NPU kernels
  are fully initialised before the first real request lands.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import openvino_genai as ov_genai

from app import config
from app.registry import ModelEntry

log = logging.getLogger(__name__)


@dataclass
class CachedModel:
    pipeline: Any           # ov_genai.LLMPipeline or EmbeddingModel
    entry: ModelEntry
    lock: threading.Lock = field(default_factory=threading.Lock)
    load_time_ms: float = 0.0
    infer_count: int = 0
    total_infer_ms: float = 0.0

    @property
    def avg_infer_ms(self) -> float:
        return self.total_infer_ms / self.infer_count if self.infer_count else 0.0


class ModelManager:
    def __init__(self) -> None:
        self._cache: dict[str, CachedModel] = {}
        self._registry: dict[str, ModelEntry] = {}
        self._global_lock = threading.Lock()

    def register_entries(self, entries: list[ModelEntry]) -> None:
        for e in entries:
            self._registry[e.name] = e
        log.info("Registered %d model(s) in manager.", len(entries))

    def get_cached(self, model_name: str) -> tuple[CachedModel, bool]:
        """Return (CachedModel, cache_hit). Compiles on first access (blocking)."""
        with self._global_lock:
            if model_name in self._cache:
                return self._cache[model_name], True

        entry = self._registry.get(model_name)
        if entry is None:
            raise KeyError(f"Model '{model_name}' not in registry.")

        cached = self._load_and_warm(entry)

        with self._global_lock:
            if model_name not in self._cache:
                self._cache[model_name] = cached
                return cached, False
            return self._cache[model_name], True

    def record_inference(self, model_name: str, elapsed_ms: float) -> None:
        with self._global_lock:
            cm = self._cache.get(model_name)
        if cm:
            cm.infer_count += 1
            cm.total_infer_ms += elapsed_ms

    def list_loaded(self) -> list[str]:
        with self._global_lock:
            return list(self._cache.keys())

    def all_names(self) -> list[str]:
        return list(self._registry.keys())

    def get_entry(self, name: str) -> Optional[ModelEntry]:
        return self._registry.get(name)

    def _load_and_warm(self, entry: ModelEntry) -> CachedModel:
        device = self._resolve_device(entry.device_preference)
        log.info("[%s] Loading on device=%s path=%s", entry.name, device, entry.path)

        t0 = time.perf_counter()
        if entry.task_type == "embedding":
            pipeline = self._load_embedding(entry.path, device)
        else:
            pipeline = ov_genai.LLMPipeline(entry.path, device)
        load_ms = (time.perf_counter() - t0) * 1000

        log.info("[%s] Compiled in %.1f ms. Running warm-up...", entry.name, load_ms)
        self._warmup(pipeline, entry)

        return CachedModel(pipeline=pipeline, entry=entry, load_time_ms=load_ms)

    @staticmethod
    def _resolve_device(preference: str) -> str:
        """Validate device plugin is available. Raises – never falls back to CPU."""
        import openvino as ov
        core = ov.Core()
        available = core.available_devices
        log.debug("OpenVINO available devices: %s", available)

        if preference in available:
            return preference
        for dev in available:
            if dev.startswith(preference):
                log.info("Device '%s' matched as '%s'.", preference, dev)
                return dev

        raise RuntimeError(
            f"Requested device '{preference}' not available. "
            f"Available: {available}. "
            "Install the Intel NPU driver. This service does NOT fall back to CPU."
        )

    @staticmethod
    def _load_embedding(model_path: str, device: str) -> Any:
        try:
            return ov_genai.EmbeddingModel(model_path, device)  # type: ignore[attr-defined]
        except AttributeError:
            import openvino as ov
            core = ov.Core()
            xmls = list(Path(model_path).glob("*.xml"))
            if not xmls:
                raise FileNotFoundError(f"No .xml found in {model_path}")
            return core.compile_model(str(xmls[0]), device)

    @staticmethod
    def _warmup(pipeline: Any, entry: ModelEntry) -> None:
        try:
            if entry.task_type == "embedding":
                if hasattr(pipeline, "infer"):
                    pipeline.infer(["Hello"])
                elif hasattr(pipeline, "embed"):
                    pipeline.embed(["Hello"])
            else:
                cfg = ov_genai.GenerationConfig()
                cfg.max_new_tokens = 1
                pipeline.generate("Hello", cfg)
            log.info("[%s] Warm-up complete.", entry.name)
        except Exception as exc:  # noqa: BLE001
            log.warning("[%s] Warm-up failed (non-fatal): %s", entry.name, exc)


_manager: Optional[ModelManager] = None


def get_manager() -> ModelManager:
    global _manager
    if _manager is None:
        _manager = ModelManager()
    return _manager
