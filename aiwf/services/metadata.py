from __future__ import annotations

import json
from io import BytesIO

from PIL import Image, PngImagePlugin

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
        payload: dict[str, object] = {"generator": "aiwf-webui", "version": "0.2.0"}
        if tags:
            payload["tags"] = tags
        meta.add_text("aiwf", json.dumps(payload))
        buffer = BytesIO()
        image.save(buffer, format="PNG", pnginfo=meta)
        buffer.seek(0)
        return Image.open(buffer)