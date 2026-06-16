from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PipelineCheckItem:
    name: str
    ok: bool
    message: str
    path: Path | None = None


@dataclass(frozen=True)
class PipelinePreflightResult:
    pipeline: str
    ok: bool
    items: tuple[PipelineCheckItem, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)

    def markdown(self) -> str:
        mark = "OK" if self.ok else "Blocked"
        lines = [f"**{mark}: {self.pipeline} pipeline**"]
        lines.extend(
            f"- {'OK' if item.ok else 'Missing'} **{item.name}:** {item.message}"
            for item in self.items
        )
        if self.warnings:
            lines.append("\n**Warnings**")
            lines.extend(f"- {warning}" for warning in self.warnings)
        return "\n".join(lines)


_ONNX_REQUIRED_MODELS = {
    "text_encoder": Path("text_encoder") / "model.onnx",
    "unet": Path("unet") / "model.onnx",
    "vae_decoder": Path("vae_decoder") / "model.onnx",
}

_TOKENIZER_HINTS = (
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer_config.json",
)


def preflight_onnx_pipeline(
    model_dir: str | Path,
    *,
    provider_preference: str = "auto",
    available_providers: list[str] | None = None,
) -> PipelinePreflightResult:
    """Check whether an ONNX txt2img model folder is usable.

    This function is intentionally light: it never creates ONNX Runtime
    sessions and it does not import torch or diffusers.
    """
    root = Path(model_dir).expanduser().resolve()
    items: list[PipelineCheckItem] = []
    warnings: list[str] = []
    metadata: dict[str, str] = {"model_dir": str(root), "provider_preference": provider_preference}

    items.append(
        PipelineCheckItem(
            "model folder",
            root.is_dir(),
            str(root) if root.exists() else f"Folder does not exist: {root}",
            root,
        )
    )

    for name, rel in _ONNX_REQUIRED_MODELS.items():
        path = root / rel
        items.append(
            PipelineCheckItem(
                name,
                path.is_file(),
                str(path) if path.is_file() else f"Expected {rel.as_posix()}",
                path,
            )
        )

    tokenizer_dir = root / "tokenizer"
    tokenizer_files = [tokenizer_dir / hint for hint in _TOKENIZER_HINTS]
    tokenizer_ok = tokenizer_dir.is_dir() and any(path.is_file() for path in tokenizer_files)
    items.append(
        PipelineCheckItem(
            "tokenizer",
            tokenizer_ok,
            str(tokenizer_dir) if tokenizer_ok else "Expected local tokenizer assets in tokenizer/",
            tokenizer_dir,
        )
    )

    providers = available_providers
    if providers is None:
        providers = _load_available_onnx_providers(warnings)
    metadata["available_providers"] = ", ".join(providers) if providers else ""
    provider_ok, provider_message = _provider_status(provider_preference, providers)
    items.append(PipelineCheckItem("execution provider", provider_ok, provider_message))

    if provider_preference in {"directml", "cpu"}:
        warnings.append("DirectML and CPU are compatibility paths. NVIDIA CUDA remains the preferred ONNX provider on this machine.")
    warnings.append("ONNX is optional. Diffusers remains the required baseline pipeline.")

    return PipelinePreflightResult(
        pipeline="ONNX",
        ok=all(item.ok for item in items),
        items=tuple(items),
        warnings=tuple(warnings),
        metadata=metadata,
    )


def preflight_diffusers_pipeline() -> PipelinePreflightResult:
    """Check the mandatory Diffusers baseline without importing heavy models."""
    warnings: list[str] = []
    items = [
        _import_check("diffusers", "Required baseline inference package."),
        _import_check("transformers", "Required for tokenizers/text encoders; must stay <5."),
        _import_check("torch", "Required runtime tensor package."),
        _import_check("safetensors", "Preferred local model file format."),
    ]
    try:
        import transformers

        version = tuple(int(part) for part in transformers.__version__.split(".")[:2])
        if version >= (5, 0):
            items.append(
                PipelineCheckItem(
                    "transformers version",
                    False,
                    f"transformers {transformers.__version__} is unsupported; use >=4.44,<5.",
                )
            )
        else:
            items.append(PipelineCheckItem("transformers version", True, transformers.__version__))
    except Exception as exc:
        warnings.append(f"Could not inspect transformers version: {exc}")

    return PipelinePreflightResult(
        pipeline="Diffusers",
        ok=all(item.ok for item in items),
        items=tuple(items),
        warnings=tuple(warnings),
    )


def _import_check(module_name: str, message: str) -> PipelineCheckItem:
    try:
        __import__(module_name)
        return PipelineCheckItem(module_name, True, message)
    except Exception as exc:
        return PipelineCheckItem(module_name, False, f"{message} Import failed: {exc}")


def _load_available_onnx_providers(warnings: list[str]) -> list[str]:
    try:
        from aiwf.infrastructure.onnx.session import get_available_providers

        return list(get_available_providers())
    except Exception as exc:
        warnings.append(f"ONNX Runtime provider probe failed: {exc}")
        return []


def _provider_status(preference: str, providers: list[str]) -> tuple[bool, str]:
    provider_set = set(providers)
    if not providers:
        return False, "onnxruntime is not installed or no providers are available."
    if preference == "auto":
        for provider in ("CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"):
            if provider in provider_set:
                return True, f"auto will use {provider}."
        return False, f"No supported provider in: {', '.join(providers)}"
    required = {
        "cuda": "CUDAExecutionProvider",
        "directml": "DmlExecutionProvider",
        "cpu": "CPUExecutionProvider",
    }.get(preference)
    if required is None:
        return False, f"Unknown provider preference: {preference}"
    if required not in provider_set:
        return False, f"{required} is not available. Installed providers: {', '.join(providers)}"
    return True, f"{required} is available."
