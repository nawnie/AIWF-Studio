from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path

from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.domain.prompt_dynamics import resolve_dynamic_prompt
from aiwf.core.domain.prompt_style import PromptStyle, apply_prompt_style
from aiwf.services.model_catalog import ModelCatalogService


class PromptProcessorService:
    """Wildcard/variant expansion, prompt files, and LoRA keyword preprocessing."""

    def __init__(
        self,
        flags: RuntimeFlags,
        settings: UserSettings,
        models: ModelCatalogService,
    ) -> None:
        self.flags = flags
        self.settings = settings
        self.models = models

    def prompts_dir(self) -> Path:
        return (self.flags.data_dir / self.settings.prompts_dir).resolve()

    def wildcards_dir(self) -> Path:
        return (self.flags.data_dir / self.settings.wildcards_dir).resolve()

    def ensure_dirs(self) -> None:
        self.prompts_dir().mkdir(parents=True, exist_ok=True)
        self.wildcards_dir().mkdir(parents=True, exist_ok=True)

    def ensure_default_styles(self) -> bool:
        from aiwf.core.domain.style_presets import ensure_default_prompt_styles

        return ensure_default_prompt_styles(self.settings)

    def list_prompt_files(self) -> list[tuple[str, str]]:
        root = self.prompts_dir()
        if not root.exists():
            return []
        files: list[tuple[str, str]] = []
        for path in sorted(root.rglob("*.txt")):
            if not path.is_file():
                continue
            rel = path.relative_to(root).as_posix()
            files.append((rel, rel))
        return files

    def read_prompt_file(self, relative_path: str | None, *, rng: random.Random | None = None) -> str:
        if not relative_path:
            return ""
        path = self.prompts_dir() / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Prompt file not found: {relative_path}")
        rng = rng or random.Random()
        lines = [
            line.strip()
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if not lines:
            return ""
        if len(lines) == 1:
            return lines[0]
        return rng.choice(lines)

    def preview_prompt_file(self, relative_path: str | None, *, max_lines: int = 6) -> str:
        if not relative_path:
            return "_Select a prompt file to preview._"
        path = self.prompts_dir() / relative_path
        if not path.is_file():
            return f"_File not found: `{relative_path}`_"
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()[:max_lines]
        body = "\n".join(f"> {line}" for line in lines)
        more = ""
        total = len(path.read_text(encoding="utf-8", errors="ignore").splitlines())
        if total > max_lines:
            more = f"\n_…{total - max_lines} more lines_"
        return f"**`{relative_path}`**\n{body}{more}"

    def folder_help(self) -> str:
        return (
            f"**Prompt files** → `{self.prompts_dir()}`  \n"
            f"**Wildcards** → `{self.wildcards_dir()}`"
        )

    def list_styles(self) -> list[PromptStyle]:
        return list(self.settings.prompt_styles)

    def style_choices(self) -> list[tuple[str, str]]:
        return [(style.name, style.name) for style in self.list_styles()]

    def find_style(self, name: str | None) -> PromptStyle | None:
        if not name:
            return None
        for style in self.list_styles():
            if style.name == name:
                return style
        return None

    def save_style(self, style: PromptStyle, *, ctx_save: Callable[[], None] | None = None) -> list[PromptStyle]:
        styles = [item for item in self.settings.prompt_styles if item.name != style.name]
        styles.append(style)
        self.settings.prompt_styles = sorted(styles, key=lambda item: item.name.lower())
        if ctx_save is not None:
            ctx_save()
        return self.list_styles()

    def delete_style(self, name: str, *, ctx_save: Callable[[], None] | None = None) -> list[PromptStyle]:
        self.settings.prompt_styles = [item for item in self.settings.prompt_styles if item.name != name]
        if ctx_save is not None:
            ctx_save()
        return self.list_styles()

    def reset_style_to_default(self, name: str, *, ctx_save: Callable[[], None] | None = None) -> PromptStyle | None:
        from aiwf.core.domain.style_presets import get_builtin_style

        preset = get_builtin_style(name)
        if preset is None:
            return None
        self.save_style(preset, ctx_save=ctx_save)
        return preset

    def prepare_prompt(
        self,
        text: str,
        *,
        negative_text: str = "",
        prompt_file: str | None = None,
        use_prompt_file: bool = False,
        style_name: str | None = None,
        style_override: PromptStyle | None = None,
        seed: int | None = None,
    ) -> tuple[str, str]:
        rng = random.Random(seed) if seed is not None and seed >= 0 else random.Random()

        if style_override is not None and (style_override.prompt.strip() or style_override.negative_prompt.strip()):
            style = style_override
        else:
            style = self.find_style(style_name)
        prompt_text, negative_text = apply_prompt_style(style, text, negative_text)

        parts: list[str] = []
        if use_prompt_file and prompt_file:
            parts.append(self.read_prompt_file(prompt_file, rng=rng))
        if prompt_text.strip():
            parts.append(prompt_text.strip())
        prompt = " ".join(part for part in parts if part).strip()

        prompt = resolve_dynamic_prompt(prompt, self.wildcards_dir(), rng)
        negative = resolve_dynamic_prompt(negative_text.strip(), self.wildcards_dir(), rng) if negative_text.strip() else ""
        prompt = self.models.expand_prompt_keywords(prompt)
        if negative:
            negative = self.models.expand_prompt_keywords(negative)
        return prompt, negative