"""Model registry for Simple AI Chat.

The registry is the single source of truth for which models exist, what they
can do, how they are launched, and roughly how much VRAM they need.  It is
loaded from ``config/models.yaml`` plus ``config/hardware.yaml``.

VRAM estimates are deliberately simple and conservative::

    vram = weights + mmproj (if on GPU) + kv(ctx, kv_type) + overhead

The ``ModelManager`` replaces an estimate with a measured value after the
first successful load, so planning self-calibrates on the real card.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

GIB = 1024 ** 3

# KV-cache size relative to f16 for llama.cpp cache types (bits per element / 16).
KV_TYPE_FACTOR: dict[str, float] = {
    "f16": 1.0,
    "bf16": 1.0,
    "q8_0": 8.5 / 16,
    "q5_1": 6.0 / 16,
    "q5_0": 5.5 / 16,
    "q4_1": 5.0 / 16,
    "q4_0": 4.5 / 16,
    "iq4_nl": 4.5 / 16,
}

KNOWN_BACKENDS = frozenset({"llama-prism", "llama", "sdcpp", "qwentts", "transformers", "cloud"})


class Residency(str, Enum):
    """How long a model is allowed to stay in VRAM."""

    PINNED = "pinned"        # the brain; evicted only as a last resort, then restored
    WARM = "warm"            # kept while there is room; LRU eviction
    TRANSIENT = "transient"  # unloaded as soon as its job finishes
    CPU = "cpu"              # never uses the VRAM budget


class RegistryError(ValueError):
    """Raised when the model registry configuration is invalid."""


@dataclass(frozen=True)
class Hardware:
    gpu_name: str = "RTX 4070 Ti SUPER"
    gpu_total_gib: float = 16.0
    reserved_gib: float = 1.0
    safety_margin_gib: float = 0.5
    ram_total_gib: float = 40.0
    ram_reserved_gib: float = 10.0
    models_dir: str = "models"
    binaries: Mapping[str, str] = field(default_factory=dict)
    base_port: int = 8100

    @property
    def vram_budget_gib(self) -> float:
        return max(0.0, self.gpu_total_gib - self.reserved_gib - self.safety_margin_gib)

    @property
    def ram_budget_gib(self) -> float:
        return max(0.0, self.ram_total_gib - self.ram_reserved_gib)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Hardware":
        gpu = data.get("gpu", {}) or {}
        ram = data.get("ram", {}) or {}
        paths = data.get("paths", {}) or {}
        return cls(
            gpu_name=str(gpu.get("name", cls.gpu_name)),
            gpu_total_gib=float(gpu.get("total_gib", cls.gpu_total_gib)),
            reserved_gib=float(gpu.get("reserved_gib", cls.reserved_gib)),
            safety_margin_gib=float(gpu.get("safety_margin_gib", cls.safety_margin_gib)),
            ram_total_gib=float(ram.get("total_gib", cls.ram_total_gib)),
            ram_reserved_gib=float(ram.get("reserved_gib", cls.ram_reserved_gib)),
            models_dir=str(paths.get("models_dir", cls.models_dir)),
            binaries=dict(paths.get("binaries", {}) or {}),
            base_port=int(data.get("base_port", cls.base_port)),
        )


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    backend: str
    capabilities: tuple[str, ...]
    residency: Residency = Residency.WARM
    priority: int = 50
    rank: int = 100
    weights_gib: float = 0.0
    mmproj_gib: float = 0.0
    mmproj_on_gpu: bool = True
    kv_kib_per_token_f16: float = 0.0
    ctx: int = 0
    kv_type: str = "f16"
    overhead_gib: float = 0.0
    ram_gib: float = 0.0
    exclusive: bool = False
    enabled: bool = True
    repo: str | None = None
    files: Mapping[str, str] = field(default_factory=dict)
    launch: Mapping[str, Any] = field(default_factory=dict)
    license: str = ""
    notes: str = ""

    @property
    def on_gpu(self) -> bool:
        return self.residency is not Residency.CPU

    def kv_gib(self, ctx: int | None = None, kv_type: str | None = None) -> float:
        tokens = self.ctx if ctx is None else ctx
        factor = KV_TYPE_FACTOR[kv_type or self.kv_type]
        return tokens * self.kv_kib_per_token_f16 * 1024 * factor / GIB

    def vram_gib(self, ctx: int | None = None, kv_type: str | None = None) -> float:
        """Estimated VRAM in GiB; 0 for CPU-resident models."""
        if not self.on_gpu:
            return 0.0
        mmproj = self.mmproj_gib if self.mmproj_on_gpu else 0.0
        return self.weights_gib + mmproj + self.kv_gib(ctx, kv_type) + self.overhead_gib

    def provides(self, capability: str) -> bool:
        return capability in self.capabilities

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelSpec":
        try:
            model_id = str(data["id"])
        except KeyError as exc:
            raise RegistryError("model entry is missing 'id'") from exc

        backend = str(data.get("backend", ""))
        if backend not in KNOWN_BACKENDS:
            raise RegistryError(f"{model_id}: unknown backend {backend!r}")

        capabilities = tuple(str(c) for c in data.get("capabilities", ()) or ())
        if not capabilities:
            raise RegistryError(f"{model_id}: at least one capability is required")

        try:
            residency = Residency(str(data.get("residency", Residency.WARM.value)))
        except ValueError as exc:
            raise RegistryError(f"{model_id}: unknown residency {data.get('residency')!r}") from exc

        kv_type = str(data.get("kv_type", "f16"))
        if kv_type not in KV_TYPE_FACTOR:
            raise RegistryError(f"{model_id}: unknown kv_type {kv_type!r}")

        return cls(
            id=model_id,
            name=str(data.get("name", model_id)),
            backend=backend,
            capabilities=capabilities,
            residency=residency,
            priority=int(data.get("priority", 50)),
            rank=int(data.get("rank", 100)),
            weights_gib=float(data.get("weights_gib", 0.0)),
            mmproj_gib=float(data.get("mmproj_gib", 0.0)),
            mmproj_on_gpu=bool(data.get("mmproj_on_gpu", True)),
            kv_kib_per_token_f16=float(data.get("kv_kib_per_token_f16", 0.0)),
            ctx=int(data.get("ctx", 0)),
            kv_type=kv_type,
            overhead_gib=float(data.get("overhead_gib", 0.0)),
            ram_gib=float(data.get("ram_gib", 0.0)),
            exclusive=bool(data.get("exclusive", False)),
            enabled=bool(data.get("enabled", True)),
            repo=data.get("repo"),
            files=dict(data.get("files", {}) or {}),
            launch=dict(data.get("launch", {}) or {}),
            license=str(data.get("license", "")),
            notes=str(data.get("notes", "")),
        )


class Registry:
    """Validated collection of model specs, toggles, and profiles."""

    def __init__(
        self,
        models: Iterable[ModelSpec],
        hardware: Hardware | None = None,
        toggles: Mapping[str, Any] | None = None,
        profiles: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.hardware = hardware or Hardware()
        self.toggles: dict[str, Any] = dict(toggles or {})
        self.profiles: dict[str, dict[str, Any]] = {k: dict(v) for k, v in (profiles or {}).items()}
        self._models: dict[str, ModelSpec] = {}
        for spec in models:
            if spec.id in self._models:
                raise RegistryError(f"duplicate model id {spec.id!r}")
            self._models[spec.id] = spec

        pinned = [m.id for m in self._models.values() if m.enabled and m.residency is Residency.PINNED]
        pinned_total = sum(self._models[m].vram_gib() for m in pinned)
        if pinned_total > self.hardware.vram_budget_gib:
            raise RegistryError(
                f"pinned models need {pinned_total:.2f} GiB but the VRAM budget is "
                f"{self.hardware.vram_budget_gib:.2f} GiB"
            )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_dicts(
        cls, models_doc: Mapping[str, Any], hardware_doc: Mapping[str, Any] | None = None
    ) -> "Registry":
        models = [ModelSpec.from_dict(entry) for entry in models_doc.get("models", []) or []]
        hardware = Hardware.from_dict(hardware_doc or {})
        return cls(
            models,
            hardware=hardware,
            toggles=models_doc.get("toggles", {}) or {},
            profiles=models_doc.get("profiles", {}) or {},
        )

    @classmethod
    def from_yaml(cls, models_path: str | Path, hardware_path: str | Path | None = None) -> "Registry":
        import yaml

        models_doc = yaml.safe_load(Path(models_path).read_text(encoding="utf-8")) or {}
        hardware_doc = {}
        if hardware_path is not None:
            hardware_doc = yaml.safe_load(Path(hardware_path).read_text(encoding="utf-8")) or {}
        return cls.from_dicts(models_doc, hardware_doc)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, model_id: str) -> ModelSpec:
        try:
            return self._models[model_id]
        except KeyError as exc:
            raise KeyError(f"unknown model {model_id!r}") from exc

    def __contains__(self, model_id: object) -> bool:
        return model_id in self._models

    def __iter__(self):
        return iter(self._models.values())

    def __len__(self) -> int:
        return len(self._models)

    def enabled(self) -> list[ModelSpec]:
        return [m for m in self._models.values() if m.enabled]

    def providers(self, capability: str) -> list[ModelSpec]:
        """Enabled models providing *capability*, best (lowest rank) first."""
        found = [m for m in self.enabled() if m.provides(capability)]
        return sorted(found, key=lambda m: (m.rank, -m.priority, m.id))

    def best_for(self, capability: str) -> ModelSpec | None:
        providers = self.providers(capability)
        return providers[0] if providers else None

    def toggle(self, name: str, default: Any = False) -> Any:
        return self.toggles.get(name, default)
