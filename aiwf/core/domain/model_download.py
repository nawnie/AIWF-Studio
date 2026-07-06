from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ModelCategory = Literal[
    "checkpoint",
    "lora",
    "vae",
    "controlnet",
    "preprocessor",
    "upscaler",
    "esrgan",
    "gfpgan",
    "codeformer",
    "faceswap",
    "embedding",
    "hypernetwork",
    "wan_safetensor",
    "wan_gguf",
    "wan_diffusers",
    "wan_lora",
    "wan_vae",
    "wan_text_encoder",
    "flux_unet_safetensor",
    "flux_unet_gguf",
    "flux_text_encoder",
    "flux_vae",
    "flux2_unet_safetensor",
    "flux2_unet_gguf",
    "flux2_components",
    "flux2_diffusers",
    "z_image_unet_safetensor",
    "z_image_unet_gguf",
    "z_image_components",
    "krea2_unet_safetensor",
    "krea2_text_encoder",
    "krea2_vae",
    "krea2_diffusers",
    "anima_unet_safetensor",
    "anima_text_encoder",
    "anima_vae",
    "qwen_image_diffusers",
    "qwen_image_nunchaku",
    "sana_diffusers",
    "sana_video_diffusers",
    "ltx_checkpoint",
    "ltx_gguf",
    "ltx_upscaler",
    "ltx_lora",
    "ltx_vae",
    "ltx_audio_vae",
    "ltx_text_encoder",
    "llm_gguf",
    "llm_safetensor",
    "rife",
    "sam",
    "other",
]

ModelSource = Literal["huggingface", "civitai", "direct"]


@dataclass(frozen=True)
class CatalogEntry:
    """Trusted model-manager entry, not arbitrary user download input.

    Download services use category/source to choose the destination and fetcher.
    Keep enough provenance here to write receipts and re-check upstream sources.
    """

    key: str
    title: str
    category: ModelCategory
    source: ModelSource
    size_mb: int | None = None
    repo_id: str = ""
    filename: str = ""
    civitai_model_id: int | None = None
    civitai_version_id: int | None = None
    url: str = ""
    notes: str = ""
    # Hugging Face snapshot downloads are directory-shaped assets; ordinary
    # entries resolve to a single file under the category destination.
    snapshot: bool = False
    coming_soon: bool = False

    def choice_label(self, *, installed: bool = False) -> str:
        size = f" · {self.size_mb}MB" if self.size_mb else ""
        mark = "  ✓ installed" if installed else ""
        return f"{self.title} [{self.category}]{size}{mark}"
