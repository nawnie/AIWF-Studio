from __future__ import annotations

import datetime
import json
import logging
import os
import re
import shutil
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aiwf.core.config.settings import RuntimeFlags
from aiwf.core.domain.model_download import CatalogEntry, ModelCategory, ModelSource
from aiwf.infrastructure.download.stream import stream_download
from aiwf.api.security import is_private_url
from aiwf.services.model_download_catalog import MODEL_DOWNLOAD_CATALOG

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]

HF_HOSTS = ("huggingface.co", "hf.co")
CIVITAI_HOSTS = ("civitai.com", "civitai.green")

CATEGORY_LABELS: dict[ModelCategory, str] = {
    "checkpoint": "Checkpoint",
    "lora": "LoRA",
    "vae": "VAE",
    "controlnet": "ControlNet",
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
    "rife": "RIFE (frame interpolation)",
    "sam": "SAM (segmentation)",
    "other": "Other (models root)",
}

CATEGORY_FOLDERS: dict[ModelCategory, tuple[str, ...]] = {
    "checkpoint": ("Stable-diffusion",),
    "lora": ("Lora",),
    "vae": ("VAE",),
    "controlnet": ("ControlNet",),
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
    "rife": ("rife",),
    "sam": ("sam",),
    "other": (),
}

CATEGORY_EXTENSION_RULES: dict[ModelCategory, tuple[str, ...]] = {
    "checkpoint": (".safetensors", ".ckpt", ".pt"),
    "lora": (".safetensors", ".ckpt", ".pt"),
    "vae": (".safetensors", ".ckpt", ".pt"),
    "controlnet": (".safetensors", ".bin", ".pt", ".pth"),
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
    "rife": (".pth",),
    "sam": (".pth",),
}


@dataclass(frozen=True)
class ParsedRemote:
    source: ModelSource
    url: str
    filename: str
    repo_id: str = ""
    civitai_model_id: int | None = None
    civitai_version_id: int | None = None
    snapshot: bool = False


def _civitai_token() -> str | None:
    return os.environ.get("CIVITAI_API_TOKEN") or os.environ.get("CIVITAI_TOKEN")


def split_hf_url(text: str) -> tuple[str, str]:
    """Parse a Hugging Face URL into ``(repo_id, inferred_filename)``."""
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
    return repo_id, Path(inferred).name if inferred else ""


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
            resolved_name = inferred or filename.strip()
            if not resolved_name:
                raise ValueError("Hugging Face file URL must include a filename after /resolve/<revision>/.")
            return ParsedRemote(
                source="huggingface",
                url=text,
                filename=Path(resolved_name).name,
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
                repo_id=repo_id,
                snapshot=True,
            )
        raise ValueError("Enter a filename or subpath for the Hugging Face repo.")
    url = f"https://huggingface.co/{repo_id}/resolve/main/{file_path}"
    return ParsedRemote(source="huggingface", url=url, filename=Path(file_path).name, repo_id=repo_id)


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
                        "  \n_Add a filename from the model's Files tab "
                        "(e.g. `model.safetensors`)._"
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
    """Download checkpoints, LoRAs, and other assets from Hugging Face, CivitAI, or direct URLs."""

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
            for name in folders:
                path = (root / name).resolve()
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
        return self.destination_dir(category) / filename

    def _validate_destination_filename(self, category: ModelCategory, filename: str) -> None:
        if category == "wan_diffusers":
            return
        allowed = CATEGORY_EXTENSION_RULES.get(category)
        if not allowed or not filename:
            return
        suffix = Path(filename).suffix.lower()
        if suffix not in allowed:
            pretty = ", ".join(allowed)
            raise ValueError(
                f"{CATEGORY_LABELS.get(category, category)} downloads must use {pretty} files. "
                f"Got `{filename}`."
            )

    def list_catalog(self) -> list[CatalogEntry]:
        return list(MODEL_DOWNLOAD_CATALOG)

    def find_catalog(self, key: str) -> CatalogEntry | None:
        for item in MODEL_DOWNLOAD_CATALOG:
            if item.key == key:
                return item
        return None

    def is_catalog_installed(self, entry: CatalogEntry) -> bool:
        filename = entry.filename or self._catalog_filename_hint(entry)
        if not filename:
            return entry.snapshot and self.destination_for(entry.category, entry.repo_id.split("/")[-1]).is_dir()
        return self.destination_for(entry.category, filename).is_file()

    def _catalog_filename_hint(self, entry: CatalogEntry) -> str:
        if entry.filename:
            return entry.filename
        if entry.url:
            return Path(urllib.parse.urlparse(entry.url).path).name
        return ""

    def parse_reference(
        self,
        *,
        source: ModelSource,
        url_or_repo: str,
        filename: str = "",
        category: ModelCategory | None = None,
    ) -> ParsedRemote:
        if source == "huggingface":
            return _parse_hf_reference(url_or_repo, filename, allow_snapshot=category == "wan_diffusers")
        if source == "civitai":
            return _parse_civitai_reference(url_or_repo)
        return _parse_direct_url(url_or_repo)

    def _catalog_to_remote(self, entry: CatalogEntry) -> ParsedRemote:
        if entry.source == "huggingface":
            return _parse_hf_reference(entry.repo_id, entry.filename, allow_snapshot=entry.snapshot)
        if entry.source == "civitai":
            return _resolve_civitai_download(
                model_id=entry.civitai_model_id,
                version_id=entry.civitai_version_id,
                token=_civitai_token(),
            )
        if entry.source == "direct":
            return _parse_direct_url(entry.url)
        raise ValueError(f"Unsupported catalog source: {entry.source}")

    def download_parsed(
        self,
        remote: ParsedRemote,
        *,
        category: ModelCategory,
        on_progress: ProgressCallback | None = None,
    ) -> Path:
        self.ensure_dirs()
        if self.flags.block_private_download_urls and remote.source == "direct" and is_private_url(remote.url):
            raise ValueError("Private, loopback, and local-network download URLs are blocked by Settings.")
        if remote.snapshot:
            if category != "wan_diffusers":
                raise ValueError("Full repository downloads are only supported for Diffusers folder categories.")
            return self._download_hf_snapshot(remote, category, on_progress=on_progress)
        self._validate_destination_filename(category, remote.filename)
        dest = self.destination_for(category, remote.filename)
        if dest.is_file():
            return dest

        headers: dict[str, str] = {}
        if remote.source == "civitai":
            token = _civitai_token()
            if token:
                headers["Authorization"] = f"Bearer {token}"

        try:
            result = stream_download(remote.url, dest, on_progress=on_progress, headers=headers)
            write_download_receipt(result, url=remote.url, source=remote.source)
            return result
        except Exception as exc:
            if remote.source == "huggingface" and remote.repo_id and remote.filename:
                path = self._download_hf_hub(remote, dest, on_progress=on_progress)
                write_download_receipt(path, url=remote.url, source=remote.source)
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
            filename=remote.filename,
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
    ) -> Path:
        from huggingface_hub import snapshot_download

        if not remote.repo_id:
            raise ValueError("Hugging Face repository is required for a Diffusers folder download.")
        token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
        target = self.destination_dir(category) / remote.repo_id.split("/")[-1]
        target.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=remote.repo_id, local_dir=str(target), token=token)
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
        return self.download_parsed(remote, category=entry.category, on_progress=on_progress)

    def folder_paths_help(self) -> str:
        lines = ["**Category folders** — files are saved here based on the selected category."]
        for key, label in CATEGORY_LABELS.items():
            try:
                path = self.destination_dir(key)
                lines.append(f"- **{label}** → `{path}`")
            except Exception:
                pass
        return "  \n".join(lines)
