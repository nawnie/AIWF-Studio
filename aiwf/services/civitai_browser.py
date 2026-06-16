"""
aiwf/services/civitai_browser.py

Thin client for the CivitAI v1 REST API and local installed-model listing.

Design rules
------------
* Zero heavy imports — only stdlib + domain types.
* Uses ``urllib.request`` (always available; consistent with model_info_lookup).
* Returns None or empty results on network errors, 404s, or missing tokens.
* Never blocks the caller on a slow/unavailable CivitAI endpoint.
* ``list_installed`` reads the local models directory; never touches network.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

from aiwf.core.domain.civitai import (
    CivitAIModel,
    CivitAIModelVersion,
    CivitAISearchResult,
)

if TYPE_CHECKING:
    from aiwf.core.config.settings import RuntimeFlags

logger = logging.getLogger(__name__)

_BASE_URL = "https://civitai.com/api/v1"
_DEFAULT_TIMEOUT = 12  # seconds

# Map CivitAI type strings to human labels
_TYPE_LABELS: dict[str, str] = {
    "Checkpoint": "Checkpoint",
    "LORA": "LoRA",
    "LoCon": "LyCORIS",
    "TextualInversion": "Embedding",
    "Hypernetwork": "Hypernetwork",
    "AestheticGradient": "Aesthetic Gradient",
    "Controlnet": "ControlNet",
    "Upscaler": "Upscaler",
    "VAE": "VAE",
    "Poses": "Poses",
    "Wildcards": "Wildcards",
    "Workflows": "Workflows",
    "Other": "Other",
}

# Installed model extensions recognised by the browser
_INSTALLED_EXTENSIONS = {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf"}


class CivitAIBrowser:
    """Browse CivitAI models and list locally installed ones."""

    def __init__(self, api_token: str | None = None) -> None:
        self._token = api_token or os.environ.get("CIVITAI_API_TOKEN", "").strip() or None

    # ------------------------------------------------------------------
    # Network helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, params: dict | None = None) -> dict | None:
        url = f"{_BASE_URL}{path}"
        if params:
            url = url + "?" + urllib.parse.urlencode(params)
        headers: dict[str, str] = {"Accept": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=_DEFAULT_TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            logger.debug("CivitAI API %s → HTTP %s", path, exc.code)
            return None
        except Exception as exc:
            logger.debug("CivitAI API %s error: %s", path, exc)
            return None

    def is_available(self) -> bool:
        """Return True if CivitAI API responds (no token required for public search)."""
        data = self._get("/models", {"limit": 1})
        return data is not None

    # ------------------------------------------------------------------
    # Model parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_version(raw: dict) -> CivitAIModelVersion:
        files = raw.get("files") or []
        primary = next((f for f in files if f.get("primary")), files[0] if files else {})
        size_kb = int((primary.get("sizeKB") or 0))
        download_url = primary.get("downloadUrl") or ""
        triggers = raw.get("trainedWords") or []
        # Collect safe preview image URLs (skip NSFW-flagged images)
        images = raw.get("images") or []
        preview_urls = [
            img["url"] for img in images
            if img.get("url") and not img.get("nsfw", False)
        ]
        return CivitAIModelVersion(
            id=raw.get("id", 0),
            name=raw.get("name", ""),
            base_model=raw.get("baseModel", ""),
            download_url=download_url,
            size_kb=size_kb,
            created_at=raw.get("createdAt", ""),
            trigger_words=triggers,
            preview_images=preview_urls,
        )

    @staticmethod
    def _parse_model(raw: dict) -> CivitAIModel:
        versions = [CivitAIBrowser._parse_version(v) for v in (raw.get("modelVersions") or [])]
        stats = raw.get("stats") or {}
        return CivitAIModel(
            id=raw.get("id", 0),
            name=raw.get("name", ""),
            type=raw.get("type", "Other"),
            nsfw=bool(raw.get("nsfw", False)),
            description=(raw.get("description") or "").strip(),
            tags=raw.get("tags") or [],
            stats_downloads=int(stats.get("downloadCount") or 0),
            stats_rating=float(stats.get("rating") or 0.0),
            creator=(raw.get("creator") or {}).get("username", ""),
            versions=versions,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str = "",
        *,
        types: list[str] | None = None,
        nsfw: bool = False,
        sort: str = "Highest Rated",
        limit: int = 20,
        cursor: str | None = None,
    ) -> CivitAISearchResult:
        """Search CivitAI models. Returns empty result on network failure."""
        params: dict = {"limit": limit, "sort": sort, "nsfw": str(nsfw).lower()}
        if query:
            params["query"] = query
        if types:
            params["types"] = ",".join(types)
        if cursor:
            params["cursor"] = cursor

        data = self._get("/models", params)
        if data is None:
            return CivitAISearchResult(
                models=[],
                total_count=0,
                error="CivitAI API unreachable — check network or try again later.",
            )

        raw_items = data.get("items") or []
        metadata = data.get("metadata") or {}
        return CivitAISearchResult(
            models=[self._parse_model(r) for r in raw_items],
            total_count=int(metadata.get("totalItems") or len(raw_items)),
            next_cursor=metadata.get("nextCursor"),
        )

    def get_model(self, model_id: int) -> CivitAIModel | None:
        """Fetch a single model by ID. Returns None on failure."""
        data = self._get(f"/models/{model_id}")
        if data is None:
            return None
        return self._parse_model(data)

    def gallery_images(
        self, result: "CivitAISearchResult"
    ) -> list[tuple[str, str]]:
        """Return (image_url, caption) pairs from a search result for gr.Gallery.

        Each entry is the first safe preview image of each model.  Models with
        no preview images are silently skipped.  The index in this list matches
        the model at the same index that *has* a preview image — callers should
        maintain a parallel index list if they need to map clicks back to models.
        """
        from aiwf.core.domain.civitai import CivitAISearchResult as _R  # local import OK
        pairs: list[tuple[str, str]] = []
        for model in result.models:
            url = model.preview_image_url()
            if url:
                caption = f"{model.name} ({model.type})"
                pairs.append((url, caption))
        return pairs

    def gallery_index_map(
        self, result: "CivitAISearchResult"
    ) -> list[int]:
        """Return the indices into result.models that have preview images.

        gallery_images() and gallery_index_map() are parallel — item i in the
        gallery corresponds to result.models[gallery_index_map()[i]].
        """
        return [
            i for i, model in enumerate(result.models)
            if model.preview_image_url()
        ]

    def list_installed(self, flags: "RuntimeFlags") -> list[dict]:
        """
        Return a list of dicts describing locally installed model files.

        Each dict has: name, path, category, size_mb, extension.
        Reads from ``flags.resolved_models_dir()`` — no network calls.
        """
        models_dir = flags.resolved_models_dir()
        if not models_dir.is_dir():
            return []

        results: list[dict] = []
        for child in sorted(models_dir.rglob("*")):
            if not child.is_file():
                continue
            if child.suffix.lower() not in _INSTALLED_EXTENSIONS:
                continue
            # category from the immediate parent folder name
            category = child.parent.name
            try:
                size_mb = child.stat().st_size / (1024 * 1024)
            except OSError:
                size_mb = 0.0
            results.append(
                {
                    "name": child.stem,
                    "path": str(child),
                    "category": category,
                    "size_mb": round(size_mb, 1),
                    "extension": child.suffix.lower(),
                }
            )
        return results

    def installed_summary(self, flags: "RuntimeFlags") -> str:
        """Return a Markdown table of installed models for display."""
        items = self.list_installed(flags)
        if not items:
            return "_No model files found in the models directory._"

        # Group by category
        by_cat: dict[str, list[dict]] = {}
        for item in items:
            by_cat.setdefault(item["category"], []).append(item)

        lines: list[str] = []
        for cat, cat_items in sorted(by_cat.items()):
            lines.append(f"**{cat}** ({len(cat_items)} file{'s' if len(cat_items) != 1 else ''})")
            for m in sorted(cat_items, key=lambda x: x["name"].lower()):
                ext = m["extension"]
                size = f"{m['size_mb']:.1f} MB"
                lines.append(f"- `{m['name']}{ext}` — {size}")
            lines.append("")

        return "\n".join(lines).strip()
