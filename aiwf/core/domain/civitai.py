"""
aiwf/core/domain/civitai.py

Domain models for CivitAI browse results.

These are pure data classes — no HTTP, no filesystem, no torch.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CivitAIModelVersion:
    """A single release of a CivitAI model."""

    id: int
    name: str
    base_model: str          # "SD 1.5", "SDXL 1.0", "Wan Video", etc.
    download_url: str
    size_kb: int = 0
    created_at: str = ""
    trigger_words: list[str] = field(default_factory=list)
    preview_images: list[str] = field(default_factory=list)  # image URLs for carousel

    def size_label(self) -> str:
        mb = self.size_kb / 1024
        if mb >= 1024:
            return f"{mb / 1024:.1f} GB"
        return f"{mb:.0f} MB"


@dataclass(frozen=True)
class CivitAIModel:
    """A model listing on CivitAI."""

    id: int
    name: str
    type: str               # "Checkpoint" | "LORA" | "TextualInversion" | ...
    nsfw: bool
    description: str = ""
    tags: list[str] = field(default_factory=list)
    stats_downloads: int = 0
    stats_rating: float = 0.0
    creator: str = ""
    versions: list[CivitAIModelVersion] = field(default_factory=list)

    @property
    def url(self) -> str:
        return f"https://civitai.com/models/{self.id}"

    @property
    def latest_version(self) -> CivitAIModelVersion | None:
        return self.versions[0] if self.versions else None

    def preview_image_url(self) -> str | None:
        """Return the first available preview image URL from the latest version."""
        ver = self.latest_version
        if ver and ver.preview_images:
            return ver.preview_images[0]
        return None

    def all_preview_images(self) -> list[str]:
        """Return all preview image URLs across all versions (deduplicated)."""
        seen: set[str] = set()
        result: list[str] = []
        for ver in self.versions:
            for url in ver.preview_images:
                if url and url not in seen:
                    seen.add(url)
                    result.append(url)
        return result

    def summary_markdown(self, *, show_nsfw: bool = False) -> str:
        """Return a short Markdown card for display in the UI."""
        if self.nsfw and not show_nsfw:
            return f"**[NSFW — hidden]** [View on CivitAI]({self.url})"
        v = self.latest_version
        lines: list[str] = [f"### [{self.name}]({self.url})"]
        lines.append(f"*{self.type}*  ·  ⬇ {self.stats_downloads:,}")
        if v:
            lines.append(f"Base: `{v.base_model}`  ·  Size: {v.size_label()}")
            if v.trigger_words:
                triggers = ", ".join(f"`{t}`" for t in v.trigger_words[:6])
                lines.append(f"Triggers: {triggers}")
        if self.tags:
            lines.append("Tags: " + " · ".join(self.tags[:8]))
        return "\n\n".join(lines)


@dataclass
class CivitAISearchResult:
    """Paginated results from a CivitAI search query."""

    models: list[CivitAIModel]
    total_count: int
    next_cursor: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None
