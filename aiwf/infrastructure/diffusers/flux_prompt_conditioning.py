from __future__ import annotations

"""Explicit FLUX prompt-conditioning modes for AIWF Studio.

The default teacher path remains unchanged. The two smaller-encoder paths are
opt-in research routes and fail closed when their local assets are incomplete.
"""

import hashlib
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file
from torch import nn
from transformers.modeling_outputs import BaseModelOutput

from aiwf.core.domain.errors import ModelNotFoundError

FLUX_CONDITIONING_TEACHER = "teacher"
FLUX_CONDITIONING_DISTILLT5_CONTROL = "distillt5-control"
FLUX_CONDITIONING_UNIVERSAL = "universal"
FLUX_CONDITIONING_MODES = (
    FLUX_CONDITIONING_TEACHER,
    FLUX_CONDITIONING_DISTILLT5_CONTROL,
    FLUX_CONDITIONING_UNIVERSAL,
)

DEFAULT_UNIVERSAL_CORE_MODEL = "google/umt5-base"
DEFAULT_UNIVERSAL_CORE_REVISION = "3d0f0ce00e52a86f64385e1b5d0660999c5f96da"
FLUX_SEQUENCE_WIDTH = 4096
FLUX_POOLED_WIDTH = 768


def normalize_flux_conditioning_mode(value: str | None) -> str:
    normalized = (value or FLUX_CONDITIONING_TEACHER).strip().lower()
    if normalized not in FLUX_CONDITIONING_MODES:
        supported = ", ".join(FLUX_CONDITIONING_MODES)
        raise ValueError(
            f"Unsupported FLUX prompt-conditioning mode {value!r}. "
            f"Expected one of: {supported}."
        )
    return normalized


def _optional_path(value: str | Path | None) -> Path | None:
    if value is None or not str(value).strip():
        return None
    return Path(value).expanduser()


def _artifact_revision(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    resolved = path.expanduser().resolve()
    candidates = [resolved]
    if resolved.is_dir():
        candidates = [
            resolved / "adapter_config.json",
            resolved / "adapter.safetensors",
            resolved / "config.json",
            resolved / "model.safetensors",
            resolved / "tokenizer.json",
            resolved / "spiece.model",
        ]
    revision: list[dict[str, Any]] = []
    for candidate in candidates:
        try:
            stat = candidate.stat()
        except OSError:
            continue
        revision.append(
            {
                "path": str(candidate.resolve()),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    if not revision:
        revision.append({"path": str(resolved), "missing": True})
    return revision


@dataclass(frozen=True)
class FluxPromptConditioningConfig:
    mode: str = FLUX_CONDITIONING_TEACHER
    distillt5_path: Path | None = None
    distillt5_tokenizer_path: Path | None = None
    universal_adapter_path: Path | None = None
    universal_core_model: str = DEFAULT_UNIVERSAL_CORE_MODEL
    universal_core_revision: str | None = DEFAULT_UNIVERSAL_CORE_REVISION

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", normalize_flux_conditioning_mode(self.mode))
        object.__setattr__(self, "distillt5_path", _optional_path(self.distillt5_path))
        object.__setattr__(
            self,
            "distillt5_tokenizer_path",
            _optional_path(self.distillt5_tokenizer_path),
        )
        object.__setattr__(
            self,
            "universal_adapter_path",
            _optional_path(self.universal_adapter_path),
        )

    @classmethod
    def from_environment(cls) -> "FluxPromptConditioningConfig":
        return cls(
            mode=os.environ.get("AIWF_FLUX_PROMPT_CONDITIONING_MODE"),
            distillt5_path=os.environ.get("AIWF_FLUX_DISTILLT5_PATH"),
            distillt5_tokenizer_path=os.environ.get(
                "AIWF_FLUX_DISTILLT5_TOKENIZER_PATH"
            ),
            universal_adapter_path=os.environ.get(
                "AIWF_FLUX_UNIVERSAL_ADAPTER_PATH"
            ),
            universal_core_model=(
                os.environ.get("AIWF_FLUX_UNIVERSAL_CORE_MODEL")
                or DEFAULT_UNIVERSAL_CORE_MODEL
            ),
            universal_core_revision=(
                os.environ.get("AIWF_FLUX_UNIVERSAL_CORE_REVISION")
                or DEFAULT_UNIVERSAL_CORE_REVISION
            ),
        )

    def updated(self, **changes: Any) -> "FluxPromptConditioningConfig":
        return replace(self, **changes)

    def cache_scope(
        self,
        component_paths: Mapping[str, str | Path] | None = None,
    ) -> str:
        payload: dict[str, Any] = {"mode": self.mode}
        if self.mode == FLUX_CONDITIONING_TEACHER:
            payload["components"] = {
                key: _artifact_revision(Path(value))
                for key, value in sorted((component_paths or {}).items())
                if key in {"clip_l", "t5xxl"}
            }
        elif self.mode == FLUX_CONDITIONING_DISTILLT5_CONTROL:
            payload["distillt5"] = _artifact_revision(self.distillt5_path)
            payload["tokenizer"] = _artifact_revision(
                self.distillt5_tokenizer_path
            )
            clip_path = (component_paths or {}).get("clip_l")
            payload["clip_l"] = (
                _artifact_revision(Path(clip_path)) if clip_path else []
            )
        else:
            payload["adapter"] = _artifact_revision(
                self.universal_adapter_path
            )
            payload["core_model"] = self.universal_core_model
            payload["core_revision"] = self.universal_core_revision
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class FluxT5ProjectedEncoder(nn.Module):
    """Local equivalent of DistillT5's T5EncoderWithProjection."""

    def __init__(self, config) -> None:  # noqa: ANN001
        super().__init__()
        from transformers import T5EncoderModel

        self.encoder = T5EncoderModel(config)
        self.final_projection = nn.Sequential(
            nn.Linear(
                int(config.project_in_dim),
                int(config.project_out_dim),
                bias=False,
            ),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(
                int(config.project_out_dim),
                int(config.project_out_dim),
                bias=False,
            ),
        )

    def forward(  # noqa: ANN201
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **_: Any,
    ):
        result = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=False,
        )
        projected = self.final_projection(result[0])
        if not return_dict:
            return (projected,)
        return BaseModelOutput(last_hidden_state=projected)


def load_distillt5_control(
    config: FluxPromptConditioningConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[nn.Module, Any, torch.device]:
    """Load the proven local DistillT5 control without network access."""

    from transformers import T5Config, T5TokenizerFast

    model_root = config.distillt5_path
    tokenizer_root = config.distillt5_tokenizer_path
    if model_root is None or not model_root.is_dir():
        raise ModelNotFoundError(
            "FLUX distillt5-control needs a local DistillT5 directory. "
            "Set AIWF_FLUX_DISTILLT5_PATH or install it under "
            "models/flux/Textencoder/DistillT5."
        )
    if tokenizer_root is None or not tokenizer_root.is_dir():
        raise ModelNotFoundError(
            "FLUX distillt5-control needs a local T5-base tokenizer directory. "
            "Set AIWF_FLUX_DISTILLT5_TOKENIZER_PATH or install it under "
            "models/flux/Textencoder/t5-v1_1-base."
        )
    state_path = model_root / "model.safetensors"
    if not state_path.is_file():
        raise ModelNotFoundError(
            f"DistillT5 weights are missing: {state_path}"
        )

    model_config = T5Config.from_pretrained(
        str(model_root),
        local_files_only=True,
    )
    if not hasattr(model_config, "project_in_dim"):
        model_config.project_in_dim = 768
    if not hasattr(model_config, "project_out_dim"):
        model_config.project_out_dim = FLUX_SEQUENCE_WIDTH
    if int(model_config.project_out_dim) != FLUX_SEQUENCE_WIDTH:
        raise ModelNotFoundError(
            "DistillT5 project_out_dim must be 4096 for the FLUX control route."
        )

    model = FluxT5ProjectedEncoder(model_config)
    state = load_file(str(state_path), device="cpu")
    if (
        "encoder.shared.weight" in state
        and "encoder.encoder.embed_tokens.weight" not in state
    ):
        state["encoder.encoder.embed_tokens.weight"] = state[
            "encoder.shared.weight"
        ]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ModelNotFoundError(
            "DistillT5 checkpoint mismatch: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    model = model.eval().requires_grad_(False).to(device=device, dtype=dtype)
    tokenizer = T5TokenizerFast.from_pretrained(
        str(tokenizer_root),
        local_files_only=True,
        legacy=True,
    )
    return model, tokenizer, device


def load_universal_conditioner(
    config: FluxPromptConditioningConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
):
    """Load the UMT5 adapter route from local assets only."""

    adapter_root = config.universal_adapter_path
    if adapter_root is None or not adapter_root.is_dir():
        raise ModelNotFoundError(
            "FLUX universal conditioning needs a trained local adapter directory. "
            "Set AIWF_FLUX_UNIVERSAL_ADAPTER_PATH to a directory containing "
            "adapter_config.json and adapter.safetensors."
        )
    for name in ("adapter_config.json", "adapter.safetensors"):
        if not (adapter_root / name).is_file():
            raise ModelNotFoundError(
                f"FLUX universal adapter is incomplete: missing {adapter_root / name}"
            )

    core_path = Path(config.universal_core_model).expanduser()
    if not core_path.is_dir():
        raise ModelNotFoundError(
            "The first FLUX universal pass is local-only. "
            "AIWF_FLUX_UNIVERSAL_CORE_MODEL must point to a complete local "
            "google/umt5-base snapshot; no model download is attempted."
        )
    try:
        from ute.integrations import load_flux_conditioner
    except ImportError as exc:
        raise ModelNotFoundError(
            "Flux universal conditioning needs the `ute.integrations` runtime "
            "package, but AIWF does not currently provide a verified installation "
            "source for it. Do not install a package by name alone. Use Flux "
            "Teacher or DistillT5-Control until a verified integration is configured."
        ) from exc

    return load_flux_conditioner(
        adapter_root,
        core_model=str(core_path.resolve()),
        core_revision=None,
        device=device,
        dtype=dtype,
    )


def validate_flux_conditioning(
    conditioning,
    *,
    expected_batch: int,
    expected_sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    sequence = getattr(conditioning, "sequence", None)
    pooled = getattr(conditioning, "pooled", None)
    expected_sequence_shape = (
        expected_batch,
        expected_sequence_length,
        FLUX_SEQUENCE_WIDTH,
    )
    expected_pooled_shape = (expected_batch, FLUX_POOLED_WIDTH)
    if not isinstance(sequence, torch.Tensor) or tuple(sequence.shape) != expected_sequence_shape:
        shape = tuple(sequence.shape) if isinstance(sequence, torch.Tensor) else None
        raise RuntimeError(
            "Universal FLUX sequence conditioning has the wrong shape: "
            f"expected {expected_sequence_shape}, got {shape}."
        )
    if not isinstance(pooled, torch.Tensor) or tuple(pooled.shape) != expected_pooled_shape:
        shape = tuple(pooled.shape) if isinstance(pooled, torch.Tensor) else None
        raise RuntimeError(
            "Universal FLUX pooled conditioning has the wrong shape: "
            f"expected {expected_pooled_shape}, got {shape}."
        )
    if not torch.isfinite(sequence).all() or not torch.isfinite(pooled).all():
        raise RuntimeError("Universal FLUX conditioning contains NaN or Inf values.")
    return sequence, pooled
