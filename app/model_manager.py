"""
Thread-safe model loader, in-memory cache, warm-up, and latency tracking.

Design:
  - One LLMPipeline (openvino_genai) per registered model, stored in _cache.
  - Per-model threading.Lock prevents concurrent inference on the same pipeline
    (OV GenAI pipelines are NOT thread-safe across threads).
  - Models are compiled lazily on first request, then held for process lifetime.
  - Warm-up: one dummy generate() fires after compilation to initialize NPU kernels.
  - _global_lock serializes cache mutations; per-model locks guard inference.
"""

from __future__ import annotations

import importlib
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import openvino as ov
import openvino_genai as ov_genai

from app.registry import ModelEntry

log = logging.getLogger(__name__)


@dataclass
class CachedModel:
    """
    Compiled pipeline plus metadata for a single registered model.
    Holds a per-model lock because OV GenAI LLMPipeline is not thread-safe.
    """

    pipeline: Any           # ov_genai.LLMPipeline or compiled Core model
    entry: ModelEntry
    lock: threading.Lock = field(default_factory=threading.Lock)
    load_time_ms: float = 0.0
    infer_count: int = 0
    total_infer_ms: float = 0.0

    @property
    def avg_infer_ms(self) -> float:
        """Average inference latency in milliseconds."""
        return self.total_infer_ms / self.infer_count if self.infer_count else 0.0


class ModelManager:
    """
    Central manager for model registration, lazy compilation, and thread-safe access.

    Public interface:
      register_entries(entries)   – populate registry from config
      get_cached(name)            – return (CachedModel, cache_hit); compiles on miss
      get_entry(name)             – raw registry lookup
      record_inference(name, ms)  – update latency stats
      list_loaded()               – names of compiled models
      all_names()                 – names of all registered models
    """

    def __init__(self) -> None:
        """Initialize empty registry and cache."""
        self._cache: dict[str, CachedModel] = {}
        self._registry: dict[str, ModelEntry] = {}
        self._global_lock = threading.Lock()

    def register_entries(self, entries: list[ModelEntry]) -> None:
        """Add model definitions to the in-memory registry."""
        for entry in entries:
            self._registry[entry.name] = entry
        log.info("Registered %d model(s).", len(entries))

    def get_cached(self, model_name: str) -> tuple[CachedModel, bool]:
        """
        Return (CachedModel, cache_hit).
        On a cache miss, compile the model on the configured device (blocking).
        Uses double-checked locking to prevent concurrent compilation of the same model.
        """
        with self._global_lock:
            if model_name in self._cache:
                return self._cache[model_name], True

        entry = self._registry.get(model_name)
        if entry is None:
            raise KeyError(f"Model '{model_name}' not in registry.")

        compiled = self._load_and_warm(entry)

        with self._global_lock:
            # Another thread may have compiled while we were loading
            if model_name not in self._cache:
                self._cache[model_name] = compiled
                return compiled, False
            return self._cache[model_name], True

    def record_inference(self, model_name: str, elapsed_ms: float) -> None:
        """Atomically update inference latency stats for the given model."""
        with self._global_lock:
            cm = self._cache.get(model_name)
        if cm:
            cm.infer_count += 1
            cm.total_infer_ms += elapsed_ms

    def list_loaded(self) -> list[str]:
        """Return names of all currently compiled (in-memory) models."""
        with self._global_lock:
            return list(self._cache.keys())

    def all_names(self) -> list[str]:
        """Return names of all registered models (compiled or not)."""
        return list(self._registry.keys())

    def get_entry(self, name: str) -> Optional[ModelEntry]:
        """Look up raw registry entry by name."""
        return self._registry.get(name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_and_warm(self, entry: ModelEntry) -> CachedModel:
        """Compile the model on the target device and run a warm-up pass."""
        device = self._resolve_device(entry.device)
        log.info("[%s] Compiling on device=%s  path=%s", entry.name, device, entry.path)

        t0 = time.perf_counter()
        if entry.task == "embedding":
            pipeline = self._load_embedding(entry.path, device)
        else:
            pipeline = ov_genai.LLMPipeline(entry.path, device)
        load_ms = (time.perf_counter() - t0) * 1000

        log.info("[%s] Compiled in %.1f ms. Running warm-up inference…", entry.name, load_ms)
        self._warmup(pipeline, entry)

        return CachedModel(pipeline=pipeline, entry=entry, load_time_ms=load_ms)

    @staticmethod
    def _resolve_device(device: str) -> str:
        """
        Validate that the requested device plugin is available.
        Raises RuntimeError and NEVER falls back to CPU.
        """
        core = ov.Core()
        available = core.available_devices
        log.debug("OpenVINO available devices: %s", available)

        if device in available:
            return device
        # Handle sub-device enumeration (e.g. "NPU.0")
        for dev in available:
            if dev.startswith(device):
                log.info("Device '%s' matched as '%s'.", device, dev)
                return dev

        raise RuntimeError(
            f"Requested device '{device}' is not available. "
            f"Available: {available}. "
            "Install the Intel NPU driver and openvino-intel-npu package. "
            "This service does NOT fall back to CPU."
        )

    @staticmethod
    def _load_embedding(model_path: str, device: str) -> Any:
        """
        Load an embedding model via GenAI EmbeddingModel API.
        Falls back to raw Core.compile_model if the API is absent in the installed SDK.
        """
        try:
            return ov_genai.EmbeddingModel(model_path, device)  # type: ignore[attr-defined]
        except AttributeError as exc:
            log.warning(
                "ov_genai.EmbeddingModel not found (%s); falling back to Core.compile_model.", exc
            )
            core = ov.Core()
            xmls = list(Path(model_path).glob("*.xml"))
            if not xmls:
                raise FileNotFoundError(
                    f"No .xml model file found in '{model_path}'"
                ) from exc
            return core.compile_model(str(xmls[0]), device)

    @staticmethod
    def _warmup(pipeline: Any, entry: ModelEntry) -> None:
        """
        Fire a single dummy inference to initialize NPU kernels.
        Failures are logged as warnings and do NOT abort startup.
        """
        try:
            if entry.task == "embedding":
                if hasattr(pipeline, "infer"):
                    pipeline.infer(["warmup"])
                elif hasattr(pipeline, "embed"):
                    pipeline.embed(["warmup"])
            else:
                cfg = ov_genai.GenerationConfig()
                cfg.max_new_tokens = 1
                pipeline.generate("warmup", cfg)
            log.info("[%s] Warm-up complete.", entry.name)
        except (RuntimeError, ValueError) as exc:
            log.warning("[%s] Warm-up inference failed (non-fatal): %s", entry.name, exc)
        except Exception as exc:  # noqa: BLE001
            log.warning("[%s] Unexpected warm-up error (non-fatal): %s", entry.name, exc)


# ---------------------------------------------------------------------------
# Custom preprocess/postprocess fn resolver (config-driven, no plugin system)
# ---------------------------------------------------------------------------

def resolve_fn(spec: str) -> Optional[Callable[..., Any]]:
    """
    Resolve a function reference from config.
    'default_genai' → None (caller uses built-in GenAI path).
    'custom:module.submodule.fn_name' → imported function object.
    """
    if spec == "default_genai":
        return None
    if spec.startswith("custom:"):
        ref = spec.split(":", 1)[1]
        module_path, fn_name = ref.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, fn_name)
    raise ValueError(f"Unknown fn spec: '{spec}'. Use 'default_genai' or 'custom:module.fn'")


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_manager: Optional[ModelManager] = None


def get_manager() -> ModelManager:
    """Return the process-wide singleton ModelManager instance."""
    global _manager
    if _manager is None:
        _manager = ModelManager()
    return _manager
