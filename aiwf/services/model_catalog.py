from __future__ import annotations

import re
from pathlib import Path

from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.domain.models import Checkpoint, LoraInfo
from aiwf.infrastructure.diffusers.model_arch import architecture_label
from aiwf.infrastructure.diffusers.loras import resolve_lora
from aiwf.infrastructure.safetensors_metadata import (
    CHECKPOINT_HEADER_KEYS,
    LORA_HEADER_KEYS,
    file_size_label,
    format_metadata_block,
    read_safetensors_metadata,
    suggest_lora_keywords,
)
from aiwf.services.generation import GenerationService

RE_LORA_KEYWORD = re.compile(r"\*lora:([a-zA-Z0-9_.\-]+)")


class ModelCatalogService:
    """Checkpoint/LoRA catalog, header metadata, and prompt keyword expansion."""

    def __init__(
        self,
        generation: GenerationService,
        flags: RuntimeFlags,
        settings: UserSettings,
    ) -> None:
        self._generation = generation
        self.flags = flags
        self.settings = settings

    def list_checkpoints(self) -> list[Checkpoint]:
        return self._generation.list_checkpoints()

    def refresh_checkpoints(self) -> list[Checkpoint]:
        return self._generation.refresh_checkpoint_catalog()

    def list_loras(self) -> list[LoraInfo]:
        return self._generation.list_loras()

    def refresh_loras(self) -> list[LoraInfo]:
        invalidate = getattr(self._generation.backend, "invalidate_loras", None)
        if callable(invalidate):
            invalidate()
        return self._generation.list_loras()

    def checkpoint_details(self, checkpoint: Checkpoint | None) -> str:
        if checkpoint is None:
            return "_Select a checkpoint to view file info and header metadata._"

        metadata = read_safetensors_metadata(checkpoint.path)
        lines = [
            f"### {checkpoint.title}",
            f"**File:** `{checkpoint.filename}`  ",
            f"**Size:** {file_size_label(checkpoint.path)}  ",
            f"**Path:** `{checkpoint.path}`  ",
            f"**Kind:** {checkpoint.kind}",
        ]
        if checkpoint.hash:
            lines.append(f"**Fingerprint:** `{checkpoint.hash}`")

        header_lines = format_metadata_block(metadata, CHECKPOINT_HEADER_KEYS)
        if header_lines:
            lines.append("\n**Header metadata**")
            lines.extend(header_lines)
        else:
            lines.append(
                "\n_No safetensors header metadata found. "
                "`.ckpt` files and some exports omit training info._"
            )

        lines.append(
            "\n**How to use**  \n"
            "Pick this model in **Studio → Model**. "
            "For inpaint workflows, prefer `[inpaint]` or `[SDXL inpaint]` checkpoints — "
            "architecture is auto-detected from model weights (same signals as A1111: 9-channel UNet + SDXL conditioner)."
        )
        return "\n".join(lines)

    def lora_details(self, lora: LoraInfo | None) -> str:
        if lora is None:
            return "_Select a LoRA to view metadata and configure keyword shortcuts._"

        metadata = read_safetensors_metadata(lora.path)
        alias = self.alias_for_lora(lora.id)
        strength = self.settings.lora_defaults.get(lora.id, 1.0)
        keywords = self.settings.lora_keywords.get(lora.id) or suggest_lora_keywords(metadata)

        lines = [
            f"### {lora.title}",
            f"**File:** `{lora.filename}`  ",
            f"**Size:** {file_size_label(lora.path)}  ",
            f"**Path:** `{lora.path}`  ",
            f"**Architecture:** {architecture_label(lora.architecture)}  ",
            f"**Shortcut:** `{self.keyword_token(alias or lora.id)}`  ",
            f"**Default strength:** {strength:.2f}",
        ]
        if lora.recommended_subdir:
            lines.append(f"**Recommended folder:** `{lora.recommended_subdir}`")
        if keywords:
            lines.append(f"**Trigger words:** {keywords}")

        header_lines = format_metadata_block(metadata, LORA_HEADER_KEYS)
        if header_lines:
            lines.append("\n**Header metadata**")
            lines.extend(header_lines)
        elif lora.path.endswith(".safetensors"):
            lines.append("\n_No training metadata in the safetensors header._")

        lines.append(
            "\n**How to use**  \n"
            f"Type `{self.keyword_token(alias or lora.id)}` in the Studio prompt — "
            "on Generate it expands to the LoRA tag plus your trigger words.  \n"
            "You can also write `<lora:name:strength>` directly in the prompt."
        )
        return "\n".join(lines)

    def keyword_token(self, alias_or_id: str) -> str:
        return f"*lora:{alias_or_id}"

    def alias_for_lora(self, lora_id: str) -> str | None:
        lowered_id = lora_id.lower()
        for alias, mapped_id in self.settings.lora_aliases.items():
            if mapped_id == lora_id or mapped_id.lower() == lowered_id:
                return alias
        return None

    def resolve_lora_keyword(self, token: str, catalog: list[LoraInfo]) -> LoraInfo | None:
        lowered = token.strip().lower()
        if not lowered:
            return None

        mapped_id = self.settings.lora_aliases.get(lowered)
        if mapped_id:
            match = resolve_lora(catalog, mapped_id)
            if match is not None:
                return match

        direct = resolve_lora(catalog, token)
        if direct is not None:
            return direct

        for lora in catalog:
            if lowered in lora.id.lower() or lowered in lora.filename.lower():
                return lora
        return None

    def lora_strength(self, lora_id: str) -> float:
        return float(self.settings.lora_defaults.get(lora_id, 1.0))

    def lora_keywords(self, lora_id: str, metadata: dict[str, str] | None = None) -> str:
        saved = self.settings.lora_keywords.get(lora_id, "").strip()
        if saved:
            return saved
        if metadata is None:
            metadata = {}
        return suggest_lora_keywords(metadata)

    def set_lora_config(
        self,
        lora_id: str,
        *,
        alias: str | None = None,
        strength: float | None = None,
        keywords: str | None = None,
    ) -> None:
        if alias is not None:
            cleaned = alias.strip().lower()
            self.settings.lora_aliases = {
                key: value
                for key, value in self.settings.lora_aliases.items()
                if value != lora_id and key != cleaned
            }
            if cleaned:
                self.settings.lora_aliases[cleaned] = lora_id

        if strength is not None:
            self.settings.lora_defaults[lora_id] = max(0.0, min(2.0, float(strength)))

        if keywords is not None:
            cleaned_keywords = keywords.strip()
            if cleaned_keywords:
                self.settings.lora_keywords[lora_id] = cleaned_keywords
            else:
                self.settings.lora_keywords.pop(lora_id, None)

    def expand_prompt_keywords(self, prompt: str) -> str:
        """Expand *lora:alias tokens into <lora:id:weight> plus configured trigger words."""
        if not prompt or "*lora:" not in prompt:
            return prompt

        catalog = self.list_loras()

        def replace(match: re.Match[str]) -> str:
            token = match.group(1)
            lora = self.resolve_lora_keyword(token, catalog)
            if lora is None:
                return match.group(0)

            metadata = read_safetensors_metadata(lora.path)
            strength = self.lora_strength(lora.id)
            keywords = self.lora_keywords(lora.id, metadata)
            lora_tag = f"<lora:{lora.id}:{strength:g}>"
            if keywords:
                return f"{keywords} {lora_tag}"
            return lora_tag

        expanded = RE_LORA_KEYWORD.sub(replace, prompt)
        return re.sub(r"\s{2,}", " ", expanded).strip()

    def lora_choices(self) -> list[tuple[str, str]]:
        choices: list[tuple[str, str]] = []
        for lora in self.list_loras():
            alias = self.alias_for_lora(lora.id)
            label = f"{lora.title}"
            if alias:
                label += f" (*lora:{alias})"
            choices.append((label, lora.id))
        return choices

    def checkpoint_choices(self) -> list[tuple[str, str]]:
        choices: list[tuple[str, str]] = []
        for checkpoint in self.list_checkpoints():
            if checkpoint.architecture == "sdxl_inpaint":
                suffix = " [SDXL inpaint]"
            elif checkpoint.architecture == "sdxl":
                suffix = " [SDXL]"
            elif checkpoint.kind == "inpaint":
                suffix = " [inpaint]"
            else:
                suffix = ""
            choices.append((f"{checkpoint.title}{suffix}", checkpoint.id))
        return choices

    def find_checkpoint(self, checkpoint_id: str | None) -> Checkpoint | None:
        if not checkpoint_id:
            return None
        for checkpoint in self.list_checkpoints():
            if checkpoint.id == checkpoint_id:
                return checkpoint
        return None

    def find_lora(self, lora_id: str | None) -> LoraInfo | None:
        if not lora_id:
            return None
        for lora in self.list_loras():
            if lora.id == lora_id:
                return lora
        return None

    def models_folder_help(self) -> str:
        models_dir = self.flags.resolved_models_dir()
        ckpt_dir = self.flags.resolved_ckpt_dir()
        lora_dir = models_dir / "Loras"
        legacy_lora_dir = models_dir / "Lora"
        return (
            f"**Checkpoint folders**  \n"
            f"- `{ckpt_dir}`  \n"
            f"- `{models_dir}`  \n\n"
            f"**LoRA folders**  \n"
            f"- `{lora_dir}`  \n"
            f"- `{legacy_lora_dir}`  \n"
            "\n"
            "Drop `.safetensors` or `.ckpt` files in these folders, then click **Refresh**."
        )
