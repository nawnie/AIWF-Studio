from __future__ import annotations

import datetime
import json
import logging
import os
import re
import shutil
import tempfile
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from aiwf.core.config.settings import RuntimeFlags
from aiwf.core.domain.model_download import CatalogEntry, ModelCategory, ModelSource
from aiwf.infrastructure.download.stream import stream_download
from aiwf.api.security import is_private_url
from aiwf.services.model_download_catalog import MODEL_DOWNLOAD_CATALOG
from aiwf.services.model_files import indexed_safetensors_shards_ready, indexed_weight_shards_ready

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]

HF_HOSTS = ("huggingface.co", "hf.co")
CIVITAI_HOSTS = ("civitai.com", "civitai.green")

CATEGORY_LABELS: dict[ModelCategory, str] = {
    "checkpoint": "Checkpoint",
    "sd_singlefile_config": "Stable Diffusion support config",
    "lora": "LoRA",
    "vae": "VAE",
    "controlnet": "ControlNet",
    "preprocessor": "ControlNet preprocessor",
    "upscaler": "Upscaler",
    "esrgan": "ESRGAN upscaler",
    "gfpgan": "GFPGAN restorer",
    "codeformer": "CodeFormer restorer",
    "faceswap": "Face swap",
    "embedding": "Embedding / Textual inversion",
    "hypernetwork": "Hypernetwork",
    "wan_safetensor": "Wan transformer (.safetensors)",
    "wan_gguf": "Wan transformer (.gguf)",
    "wan_diffusers": "Wan Diffusers folder",
    "wan_lora": "Wan LoRA",
    "wan_vae": "Wan VAE",
    "wan_text_encoder": "Wan text encoder (UMT5-XXL)",
    "flux_unet_safetensor": "Flux UNet / transformer (.safetensors)",
    "flux_unet_gguf": "Flux UNet / transformer (.gguf)",
    "flux_text_encoder": "Flux text encoder",
    "flux_vae": "Flux VAE",
    "flux_tokenizer": "Flux tokenizer files",
    "flux2_unet_safetensor": "Flux.2 Klein transformer (.safetensors)",
    "flux2_unet_gguf": "Flux.2 Klein transformer (.gguf)",
    "flux2_components": "Flux.2 Klein components",
    "flux2_diffusers": "Flux.2 Klein Diffusers pipeline",
    "flux_kontext_diffusers": "Flux Kontext Diffusers pipeline",
    "z_image_unet_safetensor": "Z-Image transformer (.safetensors)",
    "z_image_unet_gguf": "Z-Image transformer (.gguf)",
    "z_image_components": "Z-Image components",
    "krea2_unet_safetensor": "Krea 2 transformer (.safetensors)",
    "krea2_text_encoder": "Krea 2 Qwen3-VL text encoder",
    "krea2_vae": "Krea 2 Qwen Image VAE",
    "krea2_diffusers": "Krea 2 Diffusers pipeline",
    "anima_unet_safetensor": "Anima transformer (.safetensors)",
    "anima_text_encoder": "Anima Qwen text encoder",
    "anima_vae": "Anima Qwen Image VAE",
    "qwen_image_diffusers": "Qwen Image Diffusers pipeline",
    "qwen_image_nunchaku": "Qwen Image Nunchaku transformer",
    "sana_diffusers": "Sana Diffusers pipeline",
    "sana_video_diffusers": "Sana Video Diffusers pipeline",
    "ltx_checkpoint": "LTX 2.3 checkpoint",
    "ltx_gguf": "LTX 2.3 GGUF support asset",
    "ltx_upscaler": "LTX 2.3 upscaler",
    "ltx_lora": "LTX 2.3 LoRA",
    "ltx_vae": "LTX 2.3 video VAE",
    "ltx_audio_vae": "LTX 2.3 audio VAE",
    "ltx_text_encoder": "LTX 2.3 Gemma text encoder",
    "ltx_tokenizer": "LTX 2B T5 tokenizer",
    "llm_gguf": "LLM GGUF",
    "llm_safetensor": "LLM safetensors",
    "rife": "RIFE (frame interpolation)",
    "sam": "SAM (segmentation)",
    "other": "Other (models root)",
}

# Header identities used only before an explicit auto-placement. Entries without
# a strong model-header identity remain in place for the user to sort manually.
_CATALOG_HEADER_IDENTITIES: dict[str, set[tuple[str, str]]] = {
    "wan_safetensor": {(arch, role) for arch in ("wan-transformer", "wan-transformer-fp8") for role in ("high-noise", "low-noise", "unknown")},
    "wan_gguf": {(arch, role) for arch in ("wan-transformer", "wan-transformer-fp8") for role in ("high-noise", "low-noise", "unknown")},
    "wan_lora": {("wan-lora", "lora")},
    "wan_vae": {("wan-vae", "vae")},
    "wan_text_encoder": {("umt5-encoder", "text-encoder")},
    "flux_unet_safetensor": {("flux-transformer", role) for role in ("high-noise", "low-noise", "unknown")},
    "flux_unet_gguf": {("flux-transformer", role) for role in ("high-noise", "low-noise", "unknown")},
    "flux_text_encoder": {(arch, "text-encoder") for arch in ("clip", "t5xxl-encoder")},
    "flux_vae": {("flux-vae", "vae")},
    "flux2_unet_safetensor": {("flux2-klein-transformer", role) for role in ("high-noise", "low-noise", "unknown")},
    "flux2_unet_gguf": {("flux2-klein-transformer", role) for role in ("high-noise", "low-noise", "unknown")},
    "z_image_unet_safetensor": {("z-image-transformer", "unknown")},
    "z_image_unet_gguf": {("z-image-transformer", "unknown")},
    "ltx_lora": {("ltx-lora", "lora")},
    "ltx_checkpoint": {("ltx-transformer", "unknown")},
    "ltx_vae": {("ltx-vae", "vae")},
    "ltx_audio_vae": {("ltx-audio-vae", "vae")},
    "ltx_text_encoder": {("gemma-llm", "text-encoder")},
    "rife": {("rife", "upscaler")},
    "sam": {("sam", "unknown")},
}

CATEGORY_FOLDERS: dict[ModelCategory, tuple[str, ...]] = {
    "checkpoint": ("Stable-diffusion",),
    "sd_singlefile_config": ("Support", "DiffusersConfigs"),
    "lora": ("Loras",),
    "vae": ("VAE",),
    "controlnet": ("ControlNet",),
    "preprocessor": ("ControlNet", "Annotators"),
    "upscaler": ("RealESRGAN",),
    "esrgan": ("ESRGAN",),
    "gfpgan": ("GFPGAN",),
    "codeformer": ("Codeformer",),
    "faceswap": ("insightface",),
    "embedding": ("embeddings",),
    "hypernetwork": ("hypernetworks",),
    "wan_safetensor": ("wan", "Safetensor"),
    "wan_gguf": ("wan", "GGUF"),
    "wan_diffusers": ("wan", "Diffusers"),
    "wan_lora": ("wan", "lora"),
    "wan_vae": ("VAE",),
    "wan_text_encoder": ("Textencoder",),
    "flux_unet_safetensor": ("flux", "UNet"),
    "flux_unet_gguf": ("flux", "GGUF"),
    "flux_text_encoder": ("flux", "Textencoder"),
    "flux_vae": ("flux", "VAE"),
    "flux_tokenizer": ("flux", "tokenizer"),
    "flux2_unet_safetensor": ("flux2", "UNet"),
    "flux2_unet_gguf": ("flux2", "GGUF"),
    "flux2_components": ("flux2", "Components"),
    "flux2_diffusers": ("flux2", "Diffusers"),
    # The Flux Kontext GGUF resolver looks for this canonical component path.
    # Keep catalog installs aligned with sorter recommendations and resolver.
    "flux_kontext_diffusers": ("flux", "Components"),
    "flux_kontext_components": ("flux", "Components"),
    "z_image_unet_safetensor": ("z-image", "UNet"),
    "z_image_unet_gguf": ("z-image", "GGUF"),
    "z_image_components": ("z-image", "Components"),
    "krea2_unet_safetensor": ("krea2", "UNet"),
    "krea2_text_encoder": ("krea2", "Textencoder"),
    "krea2_vae": ("krea2", "VAE"),
    "krea2_diffusers": ("krea2", "Diffusers"),
    "anima_unet_safetensor": ("anima", "UNet"),
    "anima_text_encoder": ("anima", "Textencoder"),
    "anima_vae": ("anima", "VAE"),
    "qwen_image_diffusers": ("qwen-image", "Diffusers"),
    "qwen_image_nunchaku": ("qwen-image", "Nunchaku"),
    "sana_diffusers": ("sana", "Diffusers"),
    "sana_video_diffusers": ("sana-video", "Diffusers"),
    "ltx_checkpoint": ("ltx", "checkpoints"),
    "ltx_gguf": ("ltx", "GGUF"),
    "ltx_upscaler": ("ltx", "upscalers"),
    "ltx_lora": ("ltx", "loras"),
    "ltx_vae": ("ltx", "vae"),
    "ltx_audio_vae": ("ltx", "audio_vae"),
    "ltx_text_encoder": ("ltx", "text_encoder"),
    "ltx_tokenizer": ("ltx", "tokenizer"),
    "llm_gguf": ("LLM", "GGUF"),
    "llm_safetensor": ("LLM",),
    "rife": ("rife",),
    "sam": ("sam",),
    "other": (),
}

CATEGORY_EXTENSION_RULES: dict[ModelCategory, tuple[str, ...]] = {
    "checkpoint": (".safetensors", ".ckpt", ".pt"),
    "lora": (".safetensors", ".ckpt", ".pt"),
    "vae": (".safetensors", ".ckpt", ".pt"),
    "controlnet": (".safetensors", ".bin", ".pt", ".pth"),
    "preprocessor": (".safetensors", ".bin", ".ckpt", ".onnx", ".pt", ".pth"),
    "upscaler": (".pth", ".safetensors"),
    "esrgan": (".pth", ".safetensors"),
    "gfpgan": (".pth",),
    "codeformer": (".pth",),
    "faceswap": (".onnx",),
    "embedding": (".pt", ".safetensors", ".bin"),
    "hypernetwork": (".pt", ".safetensors"),
    "wan_safetensor": (".safetensors",),
    "wan_gguf": (".gguf",),
    "wan_lora": (".safetensors", ".pt", ".pth"),
    "wan_vae": (".safetensors",),
    "wan_text_encoder": (".safetensors", ".gguf"),
    "flux_unet_safetensor": (".safetensors",),
    "flux_unet_gguf": (".gguf",),
    "flux_text_encoder": (".safetensors",),
    "flux_vae": (".safetensors",),
    "flux2_unet_safetensor": (".safetensors",),
    "flux2_unet_gguf": (".gguf",),
    "flux2_components": (".safetensors", ".json", ".txt"),
    "flux2_diffusers": (".safetensors", ".json", ".txt", ".model"),
    "z_image_unet_safetensor": (".safetensors",),
    "z_image_unet_gguf": (".gguf",),
    "z_image_components": (".safetensors", ".json", ".txt"),
    "krea2_unet_safetensor": (".safetensors",),
    "krea2_text_encoder": (".safetensors",),
    "krea2_vae": (".safetensors",),
    "krea2_diffusers": (".safetensors", ".json", ".txt", ".model"),
    "anima_unet_safetensor": (".safetensors",),
    "anima_text_encoder": (".safetensors",),
    "anima_vae": (".safetensors",),
    "qwen_image_diffusers": (".safetensors", ".json", ".txt", ".model"),
    "qwen_image_nunchaku": (".safetensors",),
    "sana_diffusers": (".safetensors", ".json", ".txt", ".model"),
    "sana_video_diffusers": (".safetensors", ".json", ".txt", ".model"),
    "ltx_checkpoint": (".safetensors",),
    "ltx_gguf": (".gguf",),
    "ltx_upscaler": (".safetensors",),
    "ltx_lora": (".safetensors",),
    "ltx_vae": (".safetensors",),
    "ltx_audio_vae": (".safetensors",),
    "ltx_text_encoder": (".safetensors", ".json", ".model", ".txt"),
    "ltx_tokenizer": (".json", ".model", ".txt"),
    "llm_gguf": (".gguf",),
    "llm_safetensor": (".safetensors",),
    "rife": (".pth",),
    "sam": (".pth",),
}


@dataclass(frozen=True)
class ParsedRemote:
    source: ModelSource
    url: str
    filename: str
    repo_filename: str = ""
    local_filename: str = ""
    repo_id: str = ""
    civitai_model_id: int | None = None
    civitai_version_id: int | None = None
    snapshot: bool = False
    snapshot_allow_patterns: tuple[str, ...] = ()


def _civitai_token() -> str | None:
    return os.environ.get("CIVITAI_API_TOKEN") or os.environ.get("CIVITAI_TOKEN")


def split_hf_url(text: str) -> tuple[str, str]:
    """Parse a Hugging Face URL into ``(repo_id, inferred_file_path)``."""
    parsed = urllib.parse.urlparse(text.strip())
    if parsed.netloc.removeprefix("www.") not in HF_HOSTS:
        raise ValueError("Not a Hugging Face URL.")

    parts = [part for part in parsed.path.split("/") if part]
    if parts[:1] == ["models"] and len(parts) == 1:
        raise ValueError(
            "That link is the Hugging Face browse page, not a downloadable model. "
            "Open it in your browser, pick a model, then paste that model's page URL or `user/model` here."
        )
    if len(parts) < 2:
        raise ValueError("Hugging Face URL must include org/model (e.g. runwayml/stable-diffusion-v1-5).")

    repo_id = f"{parts[0]}/{parts[1]}"
    inferred = ""
    for marker in ("resolve", "tree", "blob"):
        if marker in parts:
            idx = parts.index(marker)
            file_parts = parts[idx + 2 :]
            if file_parts:
                inferred = "/".join(file_parts)
            break
    return repo_id, inferred.lstrip("/") if inferred else ""


def _parse_hf_reference(url_or_repo: str, filename: str = "", *, allow_snapshot: bool = False) -> ParsedRemote:
    text = (url_or_repo or "").strip()
    if not text:
        raise ValueError("Hugging Face repo or URL is required.")

    if text.startswith("http"):
        parsed = urllib.parse.urlparse(text)
        if parsed.netloc.removeprefix("www.") not in HF_HOSTS:
            raise ValueError("Not a Hugging Face URL.")
        if "resolve" in text:
            repo_id, inferred = split_hf_url(text)
            resolved_path = (filename or inferred).strip().lstrip("/")
            if not resolved_path:
                raise ValueError("Hugging Face file URL must include a filename after /resolve/<revision>/.")
            return ParsedRemote(
                source="huggingface",
                url=text,
                filename=Path(resolved_path).name,
                repo_filename=resolved_path,
                repo_id=repo_id,
            )
        repo_id, inferred = split_hf_url(text)
        file_path = (filename or inferred).strip().lstrip("/")
        if not file_path:
            if allow_snapshot:
                return ParsedRemote(
                    source="huggingface",
                    url=f"https://huggingface.co/{repo_id}",
                    filename="",
                    repo_filename="",
                    repo_id=repo_id,
                    snapshot=True,
                )
            raise ValueError(
                f"Repo `{repo_id}` needs a filename. On Hugging Face open the model → Files tab, "
                "copy a file name (e.g. model.safetensors), and paste it in **Hugging Face file path**."
            )
        url = f"https://huggingface.co/{repo_id}/resolve/main/{file_path}"
        return ParsedRemote(
            source="huggingface",
            url=url,
            filename=Path(file_path).name,
            repo_filename=file_path,
            repo_id=repo_id,
        )

    repo_id = text.rstrip("/")
    if "/" not in repo_id:
        raise ValueError("Hugging Face repo must look like user/model.")
    file_path = filename.strip().lstrip("/")
    if not file_path:
        if allow_snapshot:
            return ParsedRemote(
                source="huggingface",
                url=f"https://huggingface.co/{repo_id}",
                filename="",
                repo_filename="",
                repo_id=repo_id,
                snapshot=True,
            )
        raise ValueError("Enter a filename or subpath for the Hugging Face repo.")
    url = f"https://huggingface.co/{repo_id}/resolve/main/{file_path}"
    return ParsedRemote(
        source="huggingface",
        url=url,
        filename=Path(file_path).name,
        repo_filename=file_path,
        repo_id=repo_id,
    )


_RE_CIVITAI_MODEL = re.compile(r"/models/(\d+)", re.I)
_RE_CIVITAI_VERSION = re.compile(r"/(?:modelVersions|api/download/models)/(\d+)", re.I)


def _fetch_civitai_json(path: str, *, token: str | None = None) -> dict[str, Any]:
    url = f"https://civitai.com/api/v1{path}"
    headers: dict[str, str] = {"User-Agent": "aiwf-studio/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _civitai_file_from_version(version: dict[str, Any]) -> tuple[str, str]:
    files = version.get("files") or []
    if not files:
        raise ValueError("CivitAI version has no downloadable files.")
    preferred = next((item for item in files if item.get("primary")), files[0])
    download_url = preferred.get("downloadUrl") or preferred.get("download_url")
    if not download_url:
        raise ValueError("CivitAI file is missing a download URL.")
    name = preferred.get("name") or Path(urllib.parse.urlparse(download_url).path).name
    return str(download_url), str(name)


def _resolve_civitai_download(
    *,
    model_id: int | None,
    version_id: int | None,
    token: str | None,
) -> ParsedRemote:
    if version_id is not None:
        payload = _fetch_civitai_json(f"/model-versions/{version_id}", token=token)
        url, filename = _civitai_file_from_version(payload)
        return ParsedRemote(
            source="civitai",
            url=url,
            filename=filename,
            civitai_model_id=payload.get("modelId"),
            civitai_version_id=version_id,
        )

    if model_id is None:
        raise ValueError("CivitAI model or version id is required.")

    payload = _fetch_civitai_json(f"/models/{model_id}", token=token)
    versions = payload.get("modelVersions") or []
    if not versions:
        raise ValueError("CivitAI model has no published versions.")
    version = versions[0]
    url, filename = _civitai_file_from_version(version)
    return ParsedRemote(
        source="civitai",
        url=url,
        filename=filename,
        civitai_model_id=model_id,
        civitai_version_id=version.get("id"),
    )


def _parse_civitai_reference(url_or_id: str) -> ParsedRemote:
    text = (url_or_id or "").strip()
    if not text:
        raise ValueError("CivitAI model URL or version id is required.")

    token = _civitai_token()
    model_id: int | None = None
    version_id: int | None = None

    if text.isdigit():
        version_id = int(text)
    elif text.startswith("http"):
        parsed = urllib.parse.urlparse(text)
        if parsed.netloc.removeprefix("www.") not in CIVITAI_HOSTS:
            raise ValueError("Not a CivitAI URL.")
        version_match = _RE_CIVITAI_VERSION.search(parsed.path)
        model_match = _RE_CIVITAI_MODEL.search(parsed.path)
        query_version = urllib.parse.parse_qs(parsed.query).get("modelVersionId", [None])[0]
        if query_version and str(query_version).isdigit():
            # Model page links carry the selected version as ?modelVersionId=
            version_id = int(query_version)
        elif version_match:
            version_id = int(version_match.group(1))
        elif model_match:
            model_id = int(model_match.group(1))
        else:
            raise ValueError("Could not parse CivitAI model or version id from URL.")
    else:
        raise ValueError("Paste a CivitAI model page URL, download URL, or numeric version id.")

    return _resolve_civitai_download(model_id=model_id, version_id=version_id, token=token)


def _parse_direct_url(url: str) -> ParsedRemote:
    text = (url or "").strip()
    if not text.startswith("http"):
        raise ValueError("Direct download URL must start with http:// or https://")
    filename = Path(urllib.parse.urlparse(text).path).name
    if not filename:
        raise ValueError("Could not infer filename from URL — use a link that ends with a file name.")
    return ParsedRemote(source="direct", url=text, filename=filename)


def detect_source(url: str) -> ModelSource:
    parsed = urllib.parse.urlparse(url.strip())
    host = parsed.netloc.removeprefix("www.")
    if host in HF_HOSTS:
        return "huggingface"
    if host in CIVITAI_HOSTS:
        return "civitai"
    return "direct"


def browse_links_html() -> str:
    """Real HTML anchors — Gradio Markdown links are unreliable in some layouts."""
    return """
<div class="aiwf-external-links">
  <a class="aiwf-link-btn" href="https://huggingface.co/models?pipeline_tag=text-to-image"
     target="_blank" rel="noopener noreferrer">Browse Hugging Face</a>
  <a class="aiwf-link-btn" href="https://civitai.com/models"
     target="_blank" rel="noopener noreferrer">Browse CivitAI</a>
</div>
<p class="aiwf-external-links-hint">
  Open a site in a new tab, copy a <strong>model page URL</strong> or <strong>user/model</strong> repo,
  then paste it under <em>Custom download</em> below. Browse links are not direct downloads.
</p>
"""


def inspect_custom_input(
    *,
    source: ModelSource,
    url_or_repo: str,
    filename: str = "",
) -> tuple[ModelSource, str, str, str]:
    """Normalize pasted text and return ``(source, repo_or_url, filename, status_md)``."""
    text = (url_or_repo or "").strip()
    if not text:
        return source, "", filename, ""

    if text.startswith("http"):
        source = detect_source(text)

    try:
        if source == "huggingface":
            if text.startswith("http"):
                repo_id, inferred = split_hf_url(text)
                merged_filename = (filename or inferred).strip()
                status = f"**Hugging Face repo** `{repo_id}`"
                if merged_filename:
                    status += f"  \n**File** `{merged_filename}` — ready to download."
                else:
                    status += (
                        "  \n_Add a filename for a single-file download, or leave it empty "
                        "for a Diffusers folder checkpoint / Wan Diffusers repo._"
                    )
                return source, repo_id, merged_filename, status
            remote = _parse_hf_reference(text, filename)
            return source, text, remote.filename if not filename else filename, (
                f"**Hugging Face repo** `{remote.repo_id}`  \n**File** `{remote.filename}` — ready to download."
            )

        if source == "civitai":
            remote = _parse_civitai_reference(text)
            folder_hint = remote.filename
            return (
                source,
                text,
                filename,
                f"**CivitAI** → `{folder_hint}` — ready to download.",
            )

        remote = _parse_direct_url(text)
        return (
            source,
            text,
            filename,
            f"**Direct file** `{remote.filename}` — ready to download.",
        )
    except ValueError as exc:
        return source, text, filename, f"**Cannot use this link yet** — {exc}"


_UNSAFE_EXTENSIONS = frozenset({".ckpt", ".pt", ".pth"})


def is_unsafe_download_format(filename: str) -> bool:
    """Return True if the file extension can execute arbitrary code on load.

    .ckpt and .pt files are Python pickles that run arbitrary code when
    torch.load() is called.  Prefer .safetensors for all new downloads.
    """
    return Path(filename).suffix.lower() in _UNSAFE_EXTENSIONS


def _safe_filename(filename: str) -> str:
    name = Path(str(filename).replace("\\", "/")).name.strip()
    if not name or name in {".", ".."}:
        raise ValueError("Downloaded filename is invalid.")
    return name


def _safe_repo_dir_name(repo_id: str) -> str:
    name = Path(str(repo_id).replace("\\", "/").rstrip("/").split("/")[-1]).name.strip()
    if not name or name in {".", ".."}:
        raise ValueError("Repository name cannot be used as a folder.")
    return name


def write_download_receipt(dest: Path, *, url: str, source: str) -> None:
    """Write a companion JSON receipt alongside a downloaded model file.

    Records the download URL, source, and UTC timestamp so every file can
    be traced back to its origin.  Silently skips on any I/O error.
    """
    try:
        receipt_path = dest.with_suffix(dest.suffix + ".receipt.json")
        payload = {
            "file": dest.name,
            "url": url,
            "source": source,
            "downloaded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        receipt_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception:
        pass  # receipts are advisory; never fail a download over them


class ModelDownloadService:
    """User-facing model download boundary.

    The catalog and UI route through this service so every download lands in a
    category-safe folder, private-network URLs can be blocked, extensions are
    checked before writing, and receipts preserve provenance for maintainers.
    """

    def __init__(self, flags: RuntimeFlags) -> None:
        self.flags = flags

    def models_root(self) -> Path:
        return self.flags.resolved_models_dir()

    def category_choices(self) -> list[tuple[str, str]]:
        return [(label, key) for key, label in CATEGORY_LABELS.items()]

    def ensure_dirs(self) -> None:
        root = self.models_root()
        root.mkdir(parents=True, exist_ok=True)
        self.flags.resolved_ckpt_dir().mkdir(parents=True, exist_ok=True)
        seen: set[Path] = set()
        for folders in CATEGORY_FOLDERS.values():
            if not folders:
                continue
            path = root.joinpath(*folders).resolve()
            if path not in seen:
                path.mkdir(parents=True, exist_ok=True)
                seen.add(path)

    def destination_dir(self, category: ModelCategory) -> Path:
        root = self.models_root()
        folders = CATEGORY_FOLDERS.get(category, ())
        if folders:
            return root.joinpath(*folders)
        if category == "checkpoint":
            return self.flags.resolved_ckpt_dir()
        return root

    def destination_for(self, category: ModelCategory, filename: str) -> Path:
        return self.destination_dir(category) / _safe_filename(filename)

    def snapshot_destination_for(self, category: ModelCategory, repo_id: str) -> Path:
        name = _safe_repo_dir_name(repo_id)
        if category == "preprocessor" and name.lower() == "annotators":
            return self.destination_dir(category)
        return self.destination_dir(category) / name

    def recover_interrupted_snapshot_replacements(self) -> list[Path]:
        """Restore quarantined snapshots whose final path is absent after a crash.

        Replacement uses two directory renames, so a process or machine stop
        between them can leave the old folder in `.aiwf-recovery`. Restore only
        when the expected destination is absent and the recovery directory is
        confined beneath the configured models root.
        """
        models_root = self.models_root().resolve()
        recovery_root = models_root / ".aiwf-recovery"
        if recovery_root.is_symlink() or not recovery_root.is_dir():
            return []
        try:
            resolved_recovery_root = recovery_root.resolve(strict=True)
            resolved_recovery_root.relative_to(models_root)
        except (OSError, RuntimeError, ValueError):
            logger.warning("Ignoring model recovery directory outside the models root: %s", recovery_root)
            return []

        recovered: list[Path] = []
        recovery_pattern = re.compile(r"^(?P<name>.+)-(?P<stamp>\d{8}T\d{6}Z)-(?P<nonce>[0-9a-f]{8})$")
        for category in CATEGORY_FOLDERS:
            category_root = resolved_recovery_root / category
            if category_root.is_symlink() or not category_root.is_dir():
                continue
            candidates: dict[str, list[tuple[str, Path]]] = {}
            try:
                for item in category_root.iterdir():
                    match = recovery_pattern.match(item.name)
                    if not match or item.is_symlink() or not item.is_dir():
                        continue
                    resolved_item = item.resolve(strict=True)
                    resolved_item.relative_to(resolved_recovery_root)
                    candidates.setdefault(match.group("name"), []).append((match.group("stamp"), resolved_item))
            except (OSError, RuntimeError, ValueError):
                logger.warning("Could not safely inspect model recovery category: %s", category_root, exc_info=True)
                continue

            for name, snapshots in candidates.items():
                category_destination = self.destination_dir(category)
                destination = (
                    category_destination
                    if category == "preprocessor" and name == category_destination.name
                    else category_destination / _safe_repo_dir_name(name)
                )
                if not name or destination.exists() or destination.is_symlink():
                    continue
                try:
                    resolved_parent = destination.parent.resolve(strict=False)
                    resolved_parent.relative_to(models_root)
                    resolved_parent.mkdir(parents=True, exist_ok=True)
                    resolved_destination = resolved_parent / destination.name
                    _stamp, previous = max(snapshots, key=lambda item: item[0])
                    previous.rename(resolved_destination)
                    recovered.append(resolved_destination)
                    logger.warning(
                        "Restored model snapshot after an interrupted replacement: %s",
                        resolved_destination,
                    )
                except (OSError, RuntimeError, ValueError):
                    logger.warning("Could not restore interrupted model snapshot: %s", destination, exc_info=True)
        return recovered

    def _validate_destination_filename(self, category: ModelCategory, filename: str) -> None:
        safe_name = _safe_filename(filename) if filename else ""
        if category == "wan_diffusers" and safe_name:
            raise ValueError(
                "Wan Diffusers downloads must be full Hugging Face repository folders. "
                "Leave the filename empty so AIWF can save the folder under models/wan/Diffusers/."
            )
        if category in {"controlnet", "preprocessor", "wan_diffusers"} and not safe_name:
            return
        allowed = CATEGORY_EXTENSION_RULES.get(category)
        if not allowed or not safe_name:
            return
        suffix = Path(safe_name).suffix.lower()
        if suffix not in allowed:
            pretty = ", ".join(allowed)
            raise ValueError(
                f"{CATEGORY_LABELS.get(category, category)} downloads must use {pretty} files. "
                f"Got `{safe_name}`."
            )

    def list_catalog(self) -> list[CatalogEntry]:
        return [item for item in MODEL_DOWNLOAD_CATALOG if not item.coming_soon]

    def find_catalog(self, key: str) -> CatalogEntry | None:
        for item in MODEL_DOWNLOAD_CATALOG:
            if item.key == key:
                return item
        return None

    def is_catalog_installed(self, entry: CatalogEntry, *, search_misplaced: bool = False) -> bool:
        if entry.snapshot:
            target = self.snapshot_destination_for(entry.category, entry.repo_id)
            if self._catalog_snapshot_ready(entry, target):
                return True
            name = _safe_repo_dir_name(entry.repo_id)
            roots = [self.flags.resolved_models_dir(), *self.flags.resolved_extra_model_dirs()]
            seen_roots: set[str] = set()
            for root in roots:
                root_key = str(root.resolve()).casefold()
                if root_key in seen_roots:
                    continue
                seen_roots.add(root_key)
                folder_candidates = (
                    root.joinpath(*CATEGORY_FOLDERS.get(entry.category, ()), name),
                    root / "Diffusers" / name,
                    root / "diffusers" / name,
                    root / name,
                    root / entry.repo_id.replace("/", "--"),
                )
                if entry.category == "flux2_components":
                    folder_candidates += (
                        root / "flux2" / "Diffusers" / name,
                        root / "Flux2" / "Diffusers" / name,
                    )
                elif entry.category == "z_image_components":
                    folder_candidates += (
                        root / "z-image" / "Diffusers" / name,
                        root / "Z-Image" / "Diffusers" / name,
                    )
                if any(self._catalog_snapshot_ready(entry, path) for path in folder_candidates):
                    return True
                if search_misplaced and root.is_dir():
                    try:
                        for candidate in root.rglob(name):
                            if candidate.is_dir() and self._catalog_snapshot_ready(entry, candidate):
                                return True
                    except OSError:
                        continue
            if entry.category == "flux_tokenizer" and self._flux_tokenizer_in_hub_cache(entry):
                return True
            return False
        filename = self._catalog_local_filename_hint(entry)
        if not filename:
            return False
        if self._catalog_file_ready(entry, self.destination_for(entry.category, filename)):
            return True
        candidates = self._shared_catalog_candidates(entry, filename)
        if search_misplaced:
            misplaced = self._find_compatible_misplaced_catalog_file(entry, filename)
            if misplaced is not None:
                return True
        return any(self._catalog_file_ready(entry, path) for path in candidates)

    def _flux_tokenizer_in_hub_cache(self, entry: CatalogEntry) -> bool:
        """Match the Flux runtime resolver's local-only Hugging Face cache search."""
        roots: list[Path] = []
        model_roots = [self.flags.resolved_models_dir(), *self.flags.resolved_extra_model_dirs()]
        for model_root in model_roots:
            roots.extend((model_root, model_root / "hub", model_root / ".cache" / "huggingface" / "hub"))
        for variable in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
            value = os.environ.get(variable)
            if value:
                roots.append(Path(value).expanduser())
        try:
            from huggingface_hub.constants import HF_HUB_CACHE

            roots.append(Path(HF_HUB_CACHE))
        except Exception:
            pass

        cache_name = "models--" + entry.repo_id.replace("/", "--")
        required: tuple[str, ...] = ()
        if entry.repo_id == "openai/clip-vit-large-patch14":
            required = ("vocab.json", "merges.txt", "tokenizer_config.json")
        elif entry.repo_id == "google/t5-v1_1-xxl":
            required = ("spiece.model", "tokenizer_config.json")
        if not required:
            return False

        seen: set[str] = set()
        for root in roots:
            try:
                cache_root = root.resolve(strict=True)
                key = str(cache_root).casefold()
                if key in seen:
                    continue
                seen.add(key)
                repo_root = cache_root / cache_name
                snapshots_root = repo_root / "snapshots"
                resolved_snapshots = snapshots_root.resolve(strict=True)
                resolved_snapshots.relative_to(cache_root)
                revisions = [path for path in snapshots_root.iterdir() if path.is_dir()]
                ref_main = repo_root / "refs" / "main"
                preferred: list[Path] = []
                if ref_main.is_file():
                    revision = ref_main.read_text(encoding="utf-8").strip()
                    if revision:
                        preferred.append(snapshots_root / revision)
                ordered = preferred + sorted(
                    (path for path in revisions if path not in preferred),
                    key=lambda path: path.name.casefold(),
                    reverse=True,
                )
                for snapshot in ordered:
                    resolved_snapshot = snapshot.resolve(strict=True)
                    resolved_snapshot.relative_to(cache_root)
                    if all(
                        (resolved_file := (snapshot / name).resolve(strict=True)).is_file()
                        and resolved_file.stat().st_size > 0
                        and resolved_file.is_relative_to(cache_root)
                        for name in required
                    ):
                        return True
            except (OSError, RuntimeError, ValueError):
                continue
        return False

    def _catalog_snapshot_ready(self, entry: CatalogEntry, target: Path) -> bool:
        if entry.category == "sd_singlefile_config":
            from aiwf.infrastructure.diffusers.single_file_config import missing_single_file_config_files

            family = {
                "hf-sd15-singlefile-config": "sd15",
                "hf-sd15-inpaint-singlefile-config": "sd15_inpaint",
                "hf-sdxl-singlefile-config": "sdxl",
                "hf-sdxl-inpaint-singlefile-config": "sdxl_inpaint",
                "hf-sdxl-refiner-singlefile-config": "sdxl_refiner",
                "hf-sd35-singlefile-config": "sd35",
            }.get(entry.key)
            return family is not None and not missing_single_file_config_files(target, family)
        if entry.category == "flux2_diffusers":
            try:
                from aiwf.infrastructure.diffusers.checkpoints import flux2_klein_missing_local_files

                return not flux2_klein_missing_local_files(target)
            except Exception:
                logger.debug("Could not inspect full Flux.2 Klein pipeline snapshot completeness", exc_info=True)
                return False
        if entry.category == "z_image_components":
            try:
                from aiwf.infrastructure.diffusers.checkpoints import z_image_components_missing_local_files

                return not z_image_components_missing_local_files(target)
            except Exception:
                logger.debug("Could not inspect Z-Image component snapshot completeness", exc_info=True)
                return False
        if entry.category == "flux2_components":
            try:
                from aiwf.infrastructure.diffusers.checkpoints import flux2_klein_components_missing_local_files

                return not flux2_klein_components_missing_local_files(target)
            except Exception:
                logger.debug("Could not inspect Flux.2 Klein component snapshot completeness", exc_info=True)
                return False
        if entry.category == "ltx_text_encoder":
            try:
                from aiwf.services.ltx import ltx_gemma_hf_assets_ready

                return ltx_gemma_hf_assets_ready(target)
            except Exception:
                logger.debug("Could not inspect LTX Gemma snapshot completeness", exc_info=True)
                return False
        if entry.category == "ltx_tokenizer":
            try:
                from aiwf.services.ltx import ltx_t5_tokenizer_ready

                return ltx_t5_tokenizer_ready(target)
            except Exception:
                logger.debug("Could not inspect LTX T5 tokenizer snapshot completeness", exc_info=True)
                return False
        if entry.category == "wan_diffusers":
            required = (
                target / "model_index.json",
                target / "text_encoder" / "config.json",
                target / "text_encoder" / "model.safetensors.index.json",
                target / "tokenizer" / "tokenizer.json",
                target / "scheduler" / "scheduler_config.json",
            )
            try:
                if any(not path.is_file() or path.stat().st_size <= 0 for path in required):
                    return False
                return indexed_safetensors_shards_ready(target / "text_encoder", required[2])
            except OSError:
                return False
        if entry.category == "sana_video_diffusers":
            # Generic Diffusers completeness is not enough for Sana Video: the
            # route requires specific component classes, configs and weights.
            # Keep install/catalog status aligned with the same validator used
            # by Sana's picker and generation preflight.
            try:
                from aiwf.infrastructure.diffusers.checkpoints import sana_video_dir_has_required_local_files

                return sana_video_dir_has_required_local_files(target)
            except Exception:
                logger.debug("Could not inspect Sana Video snapshot completeness", exc_info=True)
                return False
        if entry.category == "flux_kontext_diffusers":
            try:
                from aiwf.infrastructure.diffusers.checkpoints import flux_kontext_missing_local_files

                return not flux_kontext_missing_local_files(target, limit=8)
            except Exception:
                logger.debug("Could not inspect Flux Kontext snapshot completeness", exc_info=True)
                return False
        if entry.category == "flux_kontext_components":
            try:
                from aiwf.infrastructure.diffusers.checkpoints import flux_kontext_missing_local_files

                return not flux_kontext_missing_local_files(
                    target, limit=8, require_transformer_weights=False
                )
            except Exception:
                logger.debug("Could not inspect Flux Kontext component snapshot completeness", exc_info=True)
                return False
        return self._snapshot_target_ready(entry.category, target, repo_id=entry.repo_id)

    def place_misplaced_catalog_asset(self, entry: CatalogEntry) -> bool:
        """Place one confidently identified catalog asset during explicit setup.

        This is intended for an explicit catalog/bundle install action. Ordinary
        status reads stay read-only and bounded. Only header-identified assets
        under the primary models root may be moved; extra/shared roots are never
        modified here.
        """
        if self.is_catalog_installed(entry):
            return False
        root = self.models_root().resolve()
        if entry.snapshot:
            source = self._find_misplaced_catalog_snapshot(entry)
            if source is None:
                return False
            destination = self.snapshot_destination_for(entry.category, entry.repo_id)
        else:
            filename = self._catalog_local_filename_hint(entry)
            if not filename:
                return False
            source = self._find_compatible_misplaced_catalog_file(entry, filename)
            if source is None:
                return False
            destination = self.destination_for(entry.category, filename)
        try:
            source_resolved = source.resolve(strict=True)
            destination_resolved = destination.resolve(strict=False)
            source_resolved.relative_to(root)
            destination_resolved.relative_to(root)
            if destination.exists():
                return False
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Path.rename fails rather than replacing an existing destination,
            # keeping a concurrent install or user file intact.
            source_resolved.rename(destination)
        except (OSError, ValueError):
            logger.info("Could not safely place catalog asset %s from %s", entry.key, source)
            return False
        self._invalidate_model_inventory()
        if entry.snapshot:
            return self._catalog_snapshot_ready(entry, destination)
        return self._catalog_file_ready(entry, destination)

    def copy_shared_catalog_asset_to_primary(self, entry: CatalogEntry) -> dict[str, str] | None:
        """Copy one verified catalog file from an extra model root into AIWF's root.

        This is an explicit setup action. Shared roots are treated as user-owned:
        the source is never moved or modified. Snapshot directories are excluded
        because copying multi-gigabyte pipelines needs a separately confirmed UI
        action with a disk-space estimate.
        """
        if entry.snapshot:
            return None
        filename = self._catalog_local_filename_hint(entry)
        if not filename:
            return None
        destination = self.destination_for(entry.category, filename)
        if self._catalog_file_ready(entry, destination):
            return None
        source = self._find_compatible_misplaced_catalog_file(
            entry,
            filename,
            roots=self.flags.resolved_extra_model_dirs(),
        )
        if source is None:
            return None

        try:
            source_resolved = source.resolve(strict=True)
            destination_resolved = self._confined_primary_destination(destination)
            if destination_resolved is None or destination_resolved.exists() or not source_resolved.is_file():
                return None
            extra_roots = [root.resolve() for root in self.flags.resolved_extra_model_dirs()]
            if not any(source_resolved.is_relative_to(root) for root in extra_roots):
                return None
            source_size = source_resolved.stat().st_size
            if source_size <= 0:
                return None
            destination_resolved.parent.mkdir(parents=True, exist_ok=True)
            # Re-resolve after creating the parent: a linked parent must not
            # redirect an import outside AIWF's canonical models directory.
            destination_resolved = self._confined_primary_destination(destination)
            if destination_resolved is None or destination_resolved.exists():
                return None
            volume_path = self._nearest_existing_parent(destination_resolved.parent)
            free_bytes = shutil.disk_usage(volume_path).free
            required_bytes = source_size + max(64 * 1024 * 1024, (source_size + 49) // 50)
            if free_bytes < required_bytes:
                raise ValueError(
                    f"Insufficient disk space to copy verified shared asset `{entry.key}`: "
                    f"{required_bytes} bytes required, {free_bytes} bytes available. "
                    "Free space in the AIWF model volume and retry; no network download was started."
                )
            with tempfile.TemporaryDirectory(prefix=".aiwf-import-", dir=destination_resolved.parent) as staging_dir:
                staged = Path(staging_dir) / destination.name
                shutil.copy2(source_resolved, staged)
                if staged.stat().st_size != source_size or not self._catalog_file_ready(entry, staged):
                    return None
                # Check again after staging to guard against a changed or
                # redirected destination hierarchy while the copy was active.
                checked_destination = self._confined_primary_destination(destination)
                if checked_destination is None or checked_destination.exists():
                    return None
                # On Windows os.rename fails if a concurrent setup created the
                # destination, so this does not overwrite a user's model.
                os.rename(staged, checked_destination)
                destination_resolved = checked_destination
        except (OSError, ValueError):
            logger.info("Could not safely copy shared catalog asset %s from %s", entry.key, source)
            return None
        self._invalidate_model_inventory()
        return {"source": str(source_resolved), "target": str(destination_resolved)}

    def _confined_primary_destination(self, destination: Path) -> Path | None:
        """Resolve a destination without entering a configured read-only root."""
        try:
            root = self.models_root().resolve(strict=False)
            resolved = Path(destination).resolve(strict=False)
            resolved.relative_to(root)
        except (OSError, RuntimeError, ValueError):
            return None
        # Settings can point a shared root at the primary root or one of its
        # ancestors. In that ambiguous overlap, the shared-root read-only
        # contract wins for imports: do not write into it.
        try:
            for shared_root in self.flags.resolved_extra_model_dirs():
                try:
                    resolved.relative_to(Path(shared_root).resolve(strict=False))
                except ValueError:
                    continue
                return None
        except (OSError, RuntimeError):
            return None
        return resolved

    @staticmethod
    def _nearest_existing_parent(path: Path) -> Path:
        current = Path(path)
        while not current.exists() and current != current.parent:
            current = current.parent
        return current

    def preview_shared_catalog_snapshot_import(self, entry: CatalogEntry) -> dict[str, Any] | None:
        """Describe one verified shared snapshot and its copy-space requirement."""
        if not entry.snapshot or self._catalog_snapshot_ready(
            entry, self.snapshot_destination_for(entry.category, entry.repo_id)
        ):
            return None
        source = self._find_shared_catalog_snapshot(entry)
        if source is None:
            return None
        try:
            size_bytes = self._tree_size_without_links(source)
            target = self._confined_primary_destination(
                self.snapshot_destination_for(entry.category, entry.repo_id)
            )
            if target is None:
                return None
            volume_path = self._nearest_existing_parent(target.parent)
            free_bytes = shutil.disk_usage(volume_path).free
        except OSError:
            return None
        required_bytes = size_bytes + max(64 * 1024 * 1024, (size_bytes + 49) // 50)
        return {
            "source": str(source),
            "target": str(target),
            "sizeBytes": size_bytes,
            "requiredBytes": required_bytes,
            "freeBytes": free_bytes,
            "enoughSpace": free_bytes >= required_bytes,
        }

    @staticmethod
    def _tree_size_without_links(root: Path) -> int:
        def is_link_or_junction(path: Path) -> bool:
            is_junction = getattr(path, "is_junction", None)
            return path.is_symlink() or (callable(is_junction) and is_junction())

        total = 0
        for current, directories, files in os.walk(root, followlinks=False):
            current_path = Path(current)
            for name in directories:
                if is_link_or_junction(current_path / name):
                    raise OSError("Snapshot contains a linked directory")
            for name in files:
                path = current_path / name
                if is_link_or_junction(path) or not path.is_file():
                    raise OSError("Snapshot contains a linked or non-regular file")
                total += path.stat().st_size
        return total

    def _find_shared_catalog_snapshot(self, entry: CatalogEntry) -> Path | None:
        name = _safe_repo_dir_name(entry.repo_id)
        if not name:
            return None
        matches: list[Path] = []
        for root in self.flags.resolved_extra_model_dirs():
            if not root.is_dir():
                continue
            candidates = (
                root.joinpath(*CATEGORY_FOLDERS.get(entry.category, ()), name),
                root / "Diffusers" / name,
                root / "diffusers" / name,
                root / name,
                root / entry.repo_id.replace("/", "--"),
            )
            for candidate in candidates:
                try:
                    if candidate.is_symlink() or not candidate.is_dir():
                        continue
                    resolved = candidate.resolve(strict=True)
                    if not any(resolved.is_relative_to(extra.resolve()) for extra in self.flags.resolved_extra_model_dirs()):
                        continue
                    if self._catalog_snapshot_ready(entry, resolved) and resolved not in matches:
                        matches.append(resolved)
                except (OSError, ValueError):
                    continue
        return matches[0] if len(matches) == 1 else None

    def copy_shared_catalog_snapshot_to_primary(
        self, entry: CatalogEntry, *, expected_source: str, expected_size_bytes: int
    ) -> dict[str, str] | None:
        """Copy a user-confirmed shared snapshot after rechecking source and disk space."""
        preview = self.preview_shared_catalog_snapshot_import(entry)
        if (
            preview is None
            or preview["source"] != expected_source
            or preview["sizeBytes"] != expected_size_bytes
            or not preview["enoughSpace"]
        ):
            return None
        source = Path(preview["source"])
        target = self._confined_primary_destination(Path(preview["target"]))
        if target is None or target.exists():
            return None
        target.parent.mkdir(parents=True, exist_ok=True)
        target = self._confined_primary_destination(self.snapshot_destination_for(entry.category, entry.repo_id))
        if target is None or target.exists():
            return None
        try:
            with tempfile.TemporaryDirectory(prefix=".aiwf-snapshot-import-", dir=target.parent) as staging:
                staged = Path(staging) / target.name
                # Preserve links during the copy so the staged-tree validator
                # rejects them instead of dereferencing a link introduced after
                # the preflight size check.
                shutil.copytree(source, staged, symlinks=True)
                if self._tree_size_without_links(staged) != expected_size_bytes:
                    return None
                if not self._catalog_snapshot_ready(entry, staged):
                    return None
                os.rename(staged, target)
            if (
                self._tree_size_without_links(target) != expected_size_bytes
                or not self._catalog_snapshot_ready(entry, target)
            ):
                logger.error("Copied snapshot failed destination verification: %s", target)
                return None
        except (OSError, ValueError):
            logger.info("Could not safely copy shared snapshot %s from %s", entry.key, source)
            return None
        self._invalidate_model_inventory()
        return {"source": str(source), "target": str(target)}

    def _find_misplaced_catalog_snapshot(self, entry: CatalogEntry) -> Path | None:
        name = _safe_repo_dir_name(entry.repo_id)
        root = self.models_root().resolve()
        if not name or not root.is_dir():
            return None
        matches: list[Path] = []
        try:
            for candidate in root.rglob(name):
                try:
                    if candidate.is_symlink() or not candidate.is_dir():
                        continue
                    resolved = candidate.resolve(strict=True)
                    resolved.relative_to(root)
                    if resolved == self.snapshot_destination_for(entry.category, entry.repo_id).resolve(strict=False):
                        continue
                    if self._catalog_snapshot_ready(entry, resolved):
                        matches.append(resolved)
                        if len(matches) > 1:
                            return None
                except (OSError, ValueError):
                    continue
        except OSError:
            return None
        return matches[0] if len(matches) == 1 else None

    def _find_compatible_misplaced_catalog_file(
        self,
        entry: CatalogEntry,
        filename: str,
        *,
        roots: list[Path] | None = None,
    ) -> Path | None:
        """Return one unambiguous, header-compatible misplaced asset in models/.

        A filename and size match alone are not sufficient: common names such as
        ``ae.safetensors`` can refer to unrelated files. When the local header
        reader cannot prove the catalog category, leave the file untouched and
        allow the normal download path to continue.
        """
        expected = _CATALOG_HEADER_IDENTITIES.get(entry.category)
        if not expected:
            return None
        search_roots = [root.resolve() for root in (roots if roots is not None else [self.models_root()])]
        search_roots = [root for root in search_roots if root.is_dir()]
        if not search_roots:
            return None
        matches: list[Path] = []
        # Only use filename-independent identity when that identity uniquely
        # proves the catalog asset, not merely its broad model family.
        renamed_flux_identity: tuple[str, str] | None = None
        flux_clip_l_requires_typed_directory = False
        # CLIP headers do not distinguish CLIP-L from CLIP-G, and a T5 header
        # does not prove fp8 versus fp16. Only the Flux VAE is specific enough
        # to permit filename-independent placement without risking mislabeling.
        if entry.key == "flux-ae-vae":
            renamed_flux_identity = ("flux-vae", "vae")
        elif entry.key == "flux-clip-l":
            # CLIP headers do not distinguish CLIP-L from CLIP-G. Permit
            # auto-placement only when the shared root also identifies the
            # variant through ComfyUI's canonical text_encoders/CLIP-L layout.
            # Generic/unsorted CLIP files remain manual because their identity
            # cannot be inferred safely from the tensor header alone.
            renamed_flux_identity = ("clip", "text-encoder")
            flux_clip_l_requires_typed_directory = True
        try:
            candidates: list[tuple[Path, Path]] = []
            visited_entries = 0
            target_name = filename.casefold()
            for root in search_roots:
                stack = [root]
                while stack:
                    directory = stack.pop()
                    with os.scandir(directory) as entries:
                        for item in entries:
                            visited_entries += 1
                            # Bound the full walk across all roots and fail
                            # closed rather than choosing from a partial scan.
                            if visited_entries > 20000:
                                return None
                            if item.is_dir(follow_symlinks=False):
                                stack.append(Path(item.path))
                                continue
                            if not item.is_file(follow_symlinks=False):
                                continue
                            exact_name = item.name.casefold() == target_name
                            renamed_candidate = (
                                renamed_flux_identity is not None
                                and Path(item.name).suffix.casefold() in {".safetensors", ".ckpt", ".pt"}
                            )
                            if exact_name or renamed_candidate:
                                candidates.append((Path(item.path), root))
        except (OSError, PermissionError):
            return None
        for path, root in candidates:
            try:
                if path.is_symlink() or not path.is_file():
                    continue
                resolved = path.resolve(strict=True)
                resolved.relative_to(root)
                if resolved == self.destination_for(entry.category, filename).resolve(strict=False):
                    continue
                if not self._catalog_file_ready(entry, resolved):
                    continue
                from aiwf.infrastructure.model_header import read_model_info

                info = read_model_info(resolved)
                # Filename-only fallback classification has no tensor evidence
                # (for example, any empty file named ae.safetensors is called
                # a Flux VAE). Require a non-empty parsed model header.
                identity = (str(info.arch), str(info.role))
                identity_matches = identity == renamed_flux_identity if renamed_flux_identity else identity in expected
                if flux_clip_l_requires_typed_directory:
                    relative_parts = tuple(part.casefold() for part in resolved.relative_to(root).parts[:-1])
                    comfy_clip_l = "text_encoders" in relative_parts and "clip-l" in relative_parts
                    aiwf_flux_textencoder = (
                        "flux" in relative_parts
                        and any(part in {"textencoder", "text_encoders"} for part in relative_parts)
                    )
                    identity_matches = (
                        identity_matches
                        and (comfy_clip_l or aiwf_flux_textencoder)
                        and path.name.casefold() == filename.casefold()
                    )
                if entry.key == "flux-t5-fp8" and "fp8" not in str(getattr(info, "precision", "")).lower():
                    identity_matches = False
                if int(getattr(info, "tensor_count", 0) or 0) > 0 and identity_matches:
                    matches.append(resolved)
                    if len(matches) > 1:
                        return None
            except (OSError, ValueError):
                continue
        return matches[0] if len(matches) == 1 else None

    def _shared_catalog_candidates(self, entry: CatalogEntry, filename: str) -> list[Path]:
        """Find matching files in common shared-model layouts without scanning whole drives."""
        category_roots: dict[str, tuple[str, ...]] = {
            "checkpoint": ("checkpoints", "Stable-diffusion"),
            "lora": ("loras", "Loras"),
            "vae": ("vae", "VAE"),
            "wan_safetensor": ("diffusion_models", "checkpoints", "wan"),
            "wan_gguf": ("diffusion_models", "unet", "wan"),
            "wan_diffusers": ("diffusers",),
            "wan_lora": ("loras", "wan"),
            "wan_vae": ("vae",),
            "wan_text_encoder": ("text_encoders", "clip"),
            "flux_unet_safetensor": ("diffusion_models", "unet", "flux"),
            "flux_unet_gguf": ("diffusion_models", "unet", "flux"),
            "flux_text_encoder": ("text_encoders", "clip", "flux"),
            "flux_vae": ("vae", "flux"),
            "flux2_unet_safetensor": ("diffusion_models", "unet", "flux2"),
            "flux2_unet_gguf": ("diffusion_models", "unet", "flux2"),
            "flux2_components": ("text_encoders", "vae", "clip"),
            "z_image_unet_safetensor": ("diffusion_models", "unet", "z-image"),
            "z_image_unet_gguf": ("diffusion_models", "unet", "z-image"),
            "z_image_components": ("text_encoders", "vae", "clip"),
            "krea2_unet_safetensor": ("diffusion_models", "unet", "krea2"),
            "krea2_text_encoder": ("text_encoders", "clip"),
            "krea2_vae": ("vae",),
            "anima_unet_safetensor": ("diffusion_models", "unet", "anima"),
            "anima_text_encoder": ("text_encoders", "clip"),
            "anima_vae": ("vae",),
            "ltx_gguf": ("diffusion_models", "unet", "ltx"),
            "ltx_checkpoint": ("ltx/checkpoints", "checkpoints/ltx", "checkpoints", "diffusion_models/unet/ltx"),
            "ltx_vae": ("vae",),
            "ltx_audio_vae": ("vae",),
            "ltx_text_encoder": ("text_encoders", "clip"),
            "rife": ("rife",),
            "sam": ("sam",),
        }
        folders = category_roots.get(entry.category, ())
        candidates: list[Path] = []
        seen: set[str] = set()
        roots = [self.flags.resolved_models_dir(), *self.flags.resolved_extra_model_dirs()]
        seen_roots: set[str] = set()
        for root in roots:
            root_key = str(root.resolve()).casefold()
            if root_key in seen_roots:
                continue
            seen_roots.add(root_key)
            for folder in folders:
                search_root = root / folder
                if not search_root.is_dir():
                    continue
                try:
                    matches = search_root.rglob(filename)
                    for path in matches:
                        key = str(path.resolve()).casefold()
                        if key not in seen:
                            seen.add(key)
                            candidates.append(path)
                except OSError:
                    continue
        return candidates

    def _catalog_min_bytes(self, entry: CatalogEntry) -> int:
        if not entry.size_mb:
            return 0
        # Catalog sizes are rounded and upstream repos can repack files. This
        # threshold only rejects obvious failed downloads like 0-byte files,
        # HTML/XML error bodies, and tiny pointer stubs.
        return max(1024 * 1024, int(entry.size_mb * 1024 * 1024 * 0.35))

    @staticmethod
    def _gguf_file_is_structurally_valid(path: Path) -> bool:
        try:
            import gguf

            reader = gguf.GGUFReader(str(path), mode="r")
            tensors = reader.tensors
            if not tensors:
                return False
            file_size = path.stat().st_size
            return all(
                int(tensor.n_bytes) > 0
                and int(tensor.data_offset) >= int(reader.data_offset)
                and int(tensor.data_offset) + int(tensor.n_bytes) <= file_size
                and int(tensor.data.nbytes) == int(tensor.n_bytes)
                for tensor in tensors
            )
        except Exception:
            return False

    def _catalog_file_ready(self, entry: CatalogEntry, path: Path) -> bool:
        if not path.is_file():
            return False
        try:
            # Catalog entries without an upstream size still must have payload
            # bytes. Treating an empty placeholder as installed prevents the
            # explicit setup flow from repairing interrupted downloads.
            if path.stat().st_size <= 0:
                return False
        except OSError:
            return False
        if path.suffix.lower() == ".safetensors":
            # Size alone can mark a truncated or HTML/error payload as
            # installed. Validate the lightweight safetensors framing and
            # tensor byte ranges before trusting catalog readiness or copies.
            from aiwf.infrastructure.safetensors_metadata import safetensors_file_is_structurally_valid

            if not safetensors_file_is_structurally_valid(path):
                return False
        elif path.suffix.lower() == ".gguf":
            if not self._gguf_file_is_structurally_valid(path):
                return False
        min_bytes = self._catalog_min_bytes(entry)
        if not min_bytes:
            return True
        try:
            return path.stat().st_size >= min_bytes
        except OSError:
            return False

    def _quarantine_incomplete_catalog_file(self, entry: CatalogEntry, path: Path) -> None:
        if not path.is_file() or self._catalog_file_ready(entry, path):
            return
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path.replace(path.with_name(f"{path.name}.incomplete-{stamp}.bad"))

    def _catalog_filename_hint(self, entry: CatalogEntry) -> str:
        if entry.filename:
            return Path(entry.filename.replace("\\", "/")).name
        if entry.url:
            return Path(urllib.parse.urlparse(entry.url).path).name
        return ""

    def _catalog_local_filename_hint(self, entry: CatalogEntry) -> str:
        filename = self._catalog_filename_hint(entry)
        if not filename:
            return ""
        if self._catalog_filename_needs_prefix(entry, filename):
            return f"{entry.key}-{filename}"
        return filename

    def _catalog_filename_needs_prefix(self, entry: CatalogEntry, filename: str) -> bool:
        lowered = filename.lower()
        matches = [
            other
            for other in MODEL_DOWNLOAD_CATALOG
            if not other.snapshot
            and other.category == entry.category
            and self._catalog_filename_hint(other).lower() == lowered
        ]
        return len(matches) > 1

    def _snapshot_target_ready(self, category: ModelCategory, target: Path, *, repo_id: str = "") -> bool:
        if not target.is_dir():
            return False
        if category == "flux_tokenizer":
            # These are local-only runtime prerequisites; don't report a
            # partially downloaded tokenizer snapshot as installed.
            resolved_repo_id = repo_id or {
                "clip-vit-large-patch14": "openai/clip-vit-large-patch14",
                "t5-v1_1-xxl": "google/t5-v1_1-xxl",
                "t5-v1_1-base": "google/t5-v1_1-base",
            }.get(target.name)
            required = {
                "openai/clip-vit-large-patch14": ("vocab.json", "merges.txt", "tokenizer_config.json"),
                "google/t5-v1_1-xxl": ("spiece.model", "tokenizer_config.json"),
                "google/t5-v1_1-base": ("spiece.model", "tokenizer_config.json"),
            }.get(resolved_repo_id)
            return bool(required) and all(
                (target / name).is_file() and (target / name).stat().st_size > 0
                for name in required
            )
        if category == "flux_text_encoder" and repo_id == "LifuWang/DistillT5":
            return all(
                (target / name).is_file() and (target / name).stat().st_size > 0
                for name in ("config.json", "model.safetensors")
            )
        if category == "sana_video_diffusers":
            # Catalog installation status must use the same full component
            # contract as Sana Video preflight and generation. A generic
            # Diffusers transformer-only folder is not a ready video route.
            try:
                from aiwf.infrastructure.diffusers.checkpoints import sana_video_dir_has_required_local_files

                return sana_video_dir_has_required_local_files(target)
            except Exception:
                return False
        if category in {
            "checkpoint",
            "wan_diffusers",
            "flux2_diffusers",
            "krea2_diffusers",
            "qwen_image_diffusers",
            "sana_diffusers",
        }:
            try:
                model_index = json.loads((target / "model_index.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return False
            if not isinstance(model_index, dict):
                return False

            weighted_components = {
                "transformer", "unet", "text_encoder", "text_encoder_2", "vae",
                "image_encoder", "controlnet", "prior",
            }
            found_weighted_component = False
            for component, reference in model_index.items():
                if component not in weighted_components or not isinstance(reference, list) or len(reference) < 2 or not reference[1]:
                    continue
                found_weighted_component = True
                component_dir = target / component
                if not (component_dir / "config.json").is_file():
                    return False
                weight_files = [
                    path for path in component_dir.iterdir()
                    if path.is_file()
                    and path.stat().st_size > 0
                    and path.suffix.casefold() in {".safetensors", ".bin", ".pt", ".onnx"}
                ]
                indexes = list(component_dir.glob("*.index.json"))
                if not weight_files and not indexes:
                    return False
                for index in indexes:
                    if not indexed_weight_shards_ready(component_dir, index):
                        return False
            if not found_weighted_component:
                return False
            try:
                from aiwf.infrastructure.diffusers.checkpoints import diffusers_dir_has_required_local_files

                return diffusers_dir_has_required_local_files(target)
            except Exception:
                logger.debug("Could not inspect Diffusers snapshot completeness", exc_info=True)
                return False
        allowed = CATEGORY_EXTENSION_RULES.get(category, ())
        if allowed:
            try:
                paths = [path for path in target.rglob("*") if path.is_file()]
                has_model_file = any(
                    path.is_file() and path.suffix.lower() in allowed
                    for path in paths
                )
                if category == "controlnet":
                    has_diffusers_weight = any(
                        path.name.startswith("diffusion_pytorch_model.")
                        for path in paths
                    )
                    if has_diffusers_weight and not (target / "config.json").is_file():
                        return False
                return has_model_file
            except OSError:
                return False
        try:
            return any(target.iterdir())
        except OSError:
            return False

    def parse_reference(
        self,
        *,
        source: ModelSource,
        url_or_repo: str,
        filename: str = "",
        category: ModelCategory | None = None,
    ) -> ParsedRemote:
        if source == "huggingface":
            return _parse_hf_reference(
                url_or_repo,
                filename,
                allow_snapshot=category in {
                    "checkpoint",
                    "controlnet",
                    "preprocessor",
                    "wan_diffusers",
                    "ltx_text_encoder",
                    "ltx_tokenizer",
                    "flux_text_encoder",
                    "flux_tokenizer",
                    "flux2_components",
                    "flux2_diffusers",
                    "flux_kontext_diffusers",
                    "z_image_components",
                    "krea2_diffusers",
                    "qwen_image_diffusers",
                    "sana_diffusers",
                    "sana_video_diffusers",
                },
            )
        if source == "civitai":
            return _parse_civitai_reference(url_or_repo)
        return _parse_direct_url(url_or_repo)

    def _catalog_to_remote(self, entry: CatalogEntry) -> ParsedRemote:
        if entry.source == "huggingface":
            remote = _parse_hf_reference(entry.repo_id, entry.filename, allow_snapshot=entry.snapshot)
            if entry.snapshot_allow_patterns:
                remote = replace(remote, snapshot_allow_patterns=entry.snapshot_allow_patterns)
        elif entry.source == "civitai":
            remote = _resolve_civitai_download(
                model_id=entry.civitai_model_id,
                version_id=entry.civitai_version_id,
                token=_civitai_token(),
            )
        elif entry.source == "direct":
            remote = _parse_direct_url(entry.url)
        else:
            raise ValueError(f"Unsupported catalog source: {entry.source}")
        if not remote.snapshot:
            local_filename = self._catalog_local_filename_hint(entry)
            if local_filename and local_filename != remote.filename:
                remote = replace(remote, local_filename=local_filename)
        return remote

    def _invalidate_model_inventory(self) -> None:
        try:
            from aiwf.infrastructure.model_inventory import invalidate_model_inventory_cache, inventory_path

            invalidate_model_inventory_cache()
            cache_path = inventory_path(self.flags)
            if cache_path.is_file():
                cache_path.unlink()
        except Exception:
            logger.debug("Could not invalidate model inventory cache after download", exc_info=True)

    def download_parsed(
        self,
        remote: ParsedRemote,
        *,
        category: ModelCategory,
        on_progress: ProgressCallback | None = None,
        snapshot_validator: Callable[[Path], bool] | None = None,
    ) -> Path:
        self.ensure_dirs()
        # Direct URLs are the riskiest source because they can target local
        # services; keep the SSRF/private-network guard at this boundary.
        if self.flags.block_private_download_urls and remote.source == "direct" and is_private_url(remote.url):
            raise ValueError("Private, loopback, and local-network download URLs are blocked by Settings.")
        if remote.snapshot:
            if category not in {
                "checkpoint",
                "sd_singlefile_config",
                "controlnet",
                "preprocessor",
                "wan_diffusers",
                "ltx_text_encoder",
                "ltx_tokenizer",
                "flux_text_encoder",
                "flux_tokenizer",
                "flux2_components",
                "flux2_diffusers",
                "z_image_components",
                "krea2_diffusers",
                "qwen_image_diffusers",
                "sana_diffusers",
                "sana_video_diffusers",
            }:
                raise ValueError(
                    "Repository downloads are only supported for checkpoint/Diffusers support folders, "
                    "Wan Diffusers folders, LTX text/tokenizer assets, Flux tokenizer assets, "
                    "Flux2/Z-Image/Krea2/Qwen/Sana Diffusers/component folders, ControlNet, "
                    "and preprocessor categories."
                )
            path = self._download_hf_snapshot(
                remote, category, on_progress=on_progress, validator=snapshot_validator
            )
            self._invalidate_model_inventory()
            return path
        target_filename = remote.local_filename or remote.filename
        self._validate_destination_filename(category, target_filename)
        dest = self.destination_for(category, target_filename)
        if dest.is_file():
            return dest

        headers: dict[str, str] = {}
        if remote.source == "civitai":
            headers["User-Agent"] = "AIWF-Studio/1.0"
            token = _civitai_token()
            if token:
                headers["Authorization"] = f"Bearer {token}"

        try:
            result = stream_download(remote.url, dest, on_progress=on_progress, headers=headers)
            write_download_receipt(result, url=remote.url, source=remote.source)
            self._invalidate_model_inventory()
            return result
        except Exception as exc:
            if remote.source == "huggingface" and remote.repo_id and remote.filename:
                path = self._download_hf_hub(remote, dest, on_progress=on_progress)
                write_download_receipt(path, url=remote.url, source=remote.source)
                self._invalidate_model_inventory()
                return path
            raise ValueError(f"Download failed: {exc}") from exc

    def _download_hf_hub(
        self,
        remote: ParsedRemote,
        dest: Path,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> Path:
        from huggingface_hub import hf_hub_download

        token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
        cached = hf_hub_download(
            repo_id=remote.repo_id,
            filename=remote.repo_filename or remote.filename,
            token=token,
        )
        cached_path = Path(cached)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_file():
            return dest
        shutil.copy2(cached_path, dest)
        if on_progress:
            size = dest.stat().st_size
            on_progress(size, size)
        return dest

    def _download_hf_snapshot(
        self,
        remote: ParsedRemote,
        category: ModelCategory,
        *,
        on_progress: ProgressCallback | None = None,
        validator: Callable[[Path], bool] | None = None,
    ) -> Path:
        from huggingface_hub import snapshot_download

        if not remote.repo_id:
            raise ValueError("Hugging Face repository is required for a Diffusers folder download.")
        token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
        target = self.snapshot_destination_for(category, remote.repo_id)
        models_root = self.models_root().resolve()
        if target.is_symlink():
            raise ValueError(f"Snapshot destination is a link and cannot be replaced safely: {target}")
        target = target.resolve(strict=False)
        try:
            target.relative_to(models_root)
        except ValueError as exc:
            raise ValueError(f"Snapshot destination escapes the configured models folder: {target}") from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        empty_destination = False
        replace_incomplete_destination = False
        if target.exists():
            if not target.is_dir():
                raise ValueError(f"Snapshot destination exists and is not a folder: {target}")
            if validator is not None and validator(target):
                return target
            try:
                empty_destination = not any(target.iterdir())
            except OSError:
                empty_destination = False
            if not empty_destination:
                if validator is None:
                    raise ValueError(
                        f"Snapshot destination already exists: {target}. Review or move the existing folder before installing again."
                    )
                replace_incomplete_destination = True
        staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.install-", dir=target.parent))
        snapshot_options: dict[str, Any] = {
            "repo_id": remote.repo_id,
            "local_dir": str(staging),
            "token": token,
        }
        if remote.snapshot_allow_patterns:
            snapshot_options["allow_patterns"] = list(remote.snapshot_allow_patterns)
        try:
            snapshot_download(**snapshot_options)
            if not any(staging.iterdir()):
                raise ValueError(f"Downloaded repository for `{remote.repo_id}` contains no files.")
            if validator is not None and not validator(staging):
                raise ValueError(
                    f"Downloaded repository for `{remote.repo_id}` is incomplete. "
                    "Required component files or indexed weight shards are missing."
                )
            # A complete snapshot becomes visible at its final location in one
            # rename; failed or interrupted downloads remain isolated in staging.
            published = False
            if target.exists():
                try:
                    if empty_destination and target.is_dir() and not any(target.iterdir()):
                        target.rmdir()
                    elif (
                        replace_incomplete_destination
                        and target.is_dir()
                        and validator is not None
                        and not validator(target)
                    ):
                        recovery_root = models_root / ".aiwf-recovery" / category
                        recovery_root.mkdir(parents=True, exist_ok=True)
                        if recovery_root.is_symlink():
                            raise ValueError(f"Snapshot recovery destination is a link: {recovery_root}")
                        resolved_recovery_root = recovery_root.resolve(strict=True)
                        try:
                            resolved_recovery_root.relative_to(models_root)
                        except ValueError as exc:
                            raise ValueError(
                                f"Snapshot recovery destination escapes the configured models folder: {recovery_root}"
                            ) from exc
                        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                        recovery = resolved_recovery_root / f"{target.name}-{stamp}-{uuid4().hex[:8]}"
                        try:
                            target.rename(recovery)
                            os.rename(staging, target)
                            published = True
                        except BaseException:
                            if recovery.exists() and not target.exists():
                                recovery.rename(target)
                            raise
                        logger.warning(
                            "Replaced incomplete model snapshot %s; previous folder preserved at %s",
                            target,
                            recovery,
                        )
                    else:
                        raise ValueError(f"Snapshot destination appeared or changed during download: {target}")
                except OSError as exc:
                    raise ValueError(f"Snapshot destination changed during download: {target}") from exc
            if not published:
                os.rename(staging, target)
        finally:
            # KeyboardInterrupt and other BaseException exits must not leave
            # partial snapshots visible to inventory scans.
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
        if on_progress:
            on_progress(1, 1)
        return target

    def download_custom(
        self,
        *,
        source: ModelSource,
        url_or_repo: str,
        category: ModelCategory,
        filename: str = "",
        on_progress: ProgressCallback | None = None,
    ) -> Path:
        remote = self.parse_reference(source=source, url_or_repo=url_or_repo, filename=filename, category=category)
        return self.download_parsed(remote, category=category, on_progress=on_progress)

    def download_catalog(
        self,
        key: str,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> Path:
        entry = self.find_catalog(key)
        if entry is None:
            raise ValueError(f"Unknown catalog entry '{key}'")
        remote = self._catalog_to_remote(entry)
        if not remote.snapshot:
            target = self.destination_for(entry.category, remote.local_filename or remote.filename)
            self._quarantine_incomplete_catalog_file(entry, target)
        path = self.download_parsed(
            remote,
            category=entry.category,
            on_progress=on_progress,
            snapshot_validator=lambda candidate: self._catalog_snapshot_ready(entry, candidate),
        )
        if remote.snapshot and not self._catalog_snapshot_ready(entry, path):
            raise ValueError(
                f"Downloaded repository for `{entry.key}` is incomplete. "
                "Required component files or indexed weight shards are missing."
            )
        if not remote.snapshot and not self._catalog_file_ready(entry, path):
            self._quarantine_incomplete_catalog_file(entry, path)
            raise ValueError(
                f"Downloaded file for `{entry.key}` is smaller than expected. "
                "The upstream response may have been an error page or incomplete transfer."
            )
        return path

    def folder_paths_help(self) -> str:
        lines = ["**Category folders** — files are saved here based on the selected category."]
        for key, label in CATEGORY_LABELS.items():
            try:
                path = self.destination_dir(key)
                lines.append(f"- **{label}** → `{path}`")
            except Exception:
                pass
        return "  \n".join(lines)
