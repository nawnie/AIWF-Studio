from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path

from PIL import Image, PngImagePlugin

from aiwf import __version__
from aiwf.core.domain.generation import GenerationRequest
from aiwf.core.domain.models import Checkpoint
from aiwf.core.infotext import format_infotext, parse_infotext
from aiwf.core.tags import parse_tags, parse_tags_from_params


class MetadataService:
    def build_infotext(
        self,
        request: GenerationRequest,
        seed: int,
        checkpoint: Checkpoint,
        *,
        output_width: int | None = None,
        output_height: int | None = None,
    ) -> str:
        return format_infotext(
            request,
            seed,
            checkpoint,
            output_width=output_width,
            output_height=output_height,
        )

    def read_infotext(self, image: Image.Image) -> str | None:
        if not hasattr(image, "text"):
            return None
        return image.text.get("parameters")

    def file_fingerprint(self, path: str | Path) -> str | None:
        """Quick local fingerprint for metadata labels without hashing huge files."""
        try:
            resolved = Path(path)
            stat = resolved.stat()
            digest = hashlib.sha256()
            digest.update(str(stat.st_size).encode())
            digest.update(str(int(stat.st_mtime)).encode())
            with resolved.open("rb") as handle:
                digest.update(handle.read(1024 * 1024))
                if stat.st_size > 1024 * 1024:
                    handle.seek(-1024 * 1024, 2)
                    digest.update(handle.read(1024 * 1024))
            return digest.hexdigest()[:10]
        except OSError:
            return None

    def enrich_infotext(
        self,
        infotext: str,
        *,
        model_hash: str | None = None,
        vae_name: str | None = None,
        vae_hash: str | None = None,
        lora_hashes: dict[str, str] | None = None,
        app_version: str | None = None,
    ) -> str:
        additions: list[str] = []
        if model_hash:
            additions.append(f"Model hash: {model_hash}")
        if vae_name:
            additions.append(f"VAE: {vae_name}")
        if vae_hash:
            additions.append(f"VAE hash: {vae_hash}")
        if lora_hashes:
            pairs = [f"{name}: {value}" for name, value in lora_hashes.items() if value]
            if pairs:
                additions.append("Lora hashes: " + "; ".join(pairs))
        if app_version:
            additions.append(f"AIWF Studio: {app_version}")

        if not additions:
            return infotext
        clean = (infotext or "").rstrip()
        suffix = ", ".join(additions)
        if not clean:
            return suffix
        return f"{clean}, {suffix}"

    def read_tags(self, image: Image.Image) -> list[str]:
        infotext = self.read_infotext(image)
        if infotext:
            tags = parse_tags_from_params(parse_infotext(infotext))
            if tags:
                return tags

        if not hasattr(image, "text"):
            return []

        raw = image.text.get("aiwf")
        if not raw:
            return []

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return []

        stored = payload.get("tags")
        if isinstance(stored, list):
            return parse_tags(" ".join(str(tag) for tag in stored))
        if isinstance(stored, str):
            return parse_tags(stored)
        return []

    def embed(self, image: Image.Image, infotext: str, *, tags: list[str] | None = None) -> Image.Image:
        meta = PngImagePlugin.PngInfo()
        meta.add_text("parameters", infotext)
        payload: dict[str, object] = {"generator": "aiwf-studio", "version": __version__}
        if tags:
            payload["tags"] = tags
        meta.add_text("aiwf", json.dumps(payload))
        buffer = BytesIO()
        image.save(buffer, format="PNG", pnginfo=meta)
        buffer.seek(0)
        return Image.open(buffer)
