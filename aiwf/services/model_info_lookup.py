"""
aiwf/services/model_info_lookup.py

Fetch human-readable metadata about a model from remote sources.

Supported sources
-----------------
* **HuggingFace** — model card, tags, downloads, library_name, license.
  Uses the HF Hub REST API (no huggingface_hub package required).
* **CivitAI** — model name, description, tags, trigger words, base model,
  download count. Uses the CivitAI v1 API.
* **Ollama** — model details from the local Ollama server's /api/show.

Design rules
------------
* Zero imports at module level beyond stdlib.  Every HTTP call goes through
  ``_http_get()`` which uses ``urllib.request`` (always available).
* Returns None rather than raising on network errors, 404s, or missing tokens.
* Callers must treat the result as advisory — never block the user on a
  lookup failure.
* All token resolution reads environment variables or the passed argument;
  it never imports ``launch`` or any engine module.
"""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class RemoteModelInfo:
    """Normalised metadata fetched from a remote source."""

    source: str                          # "huggingface" | "civitai" | "ollama"
    name: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    trigger_words: list[str] = field(default_factory=list)
    base_model: str = ""
    license: str = ""
    url: str = ""
    downloads: int = 0
    raw: dict = field(default_factory=dict)

    def summary_markdown(self) -> str:
        """Return a short Markdown summary for display in the UI."""
        lines: list[str] = [f"**{self.name}** — *{self.source}*"]
        if self.base_model:
            lines.append(f"Base: {self.base_model}")
        if self.license:
            lines.append(f"License: {self.license}")
        if self.downloads:
            lines.append(f"Downloads: {self.downloads:,}")
        if self.tags:
            lines.append(f"Tags: {', '.join(self.tags[:10])}")
        if self.trigger_words:
            lines.append(f"**Trigger words:** `{'`, `'.join(self.trigger_words[:8])}`")
        if self.description:
            desc = self.description[:300].rstrip()
            if len(self.description) > 300:
                desc += "…"
            lines.append("")
            lines.append(desc)
        if self.url:
            lines.append(f"\n[View on {self.source}]({self.url})")
        return "  \n".join(lines)


# ---------------------------------------------------------------------------
# Lookup service
# ---------------------------------------------------------------------------

class ModelInfoLookup:
    """Fetch metadata from HuggingFace, CivitAI, or Ollama.

    All network I/O is done with ``urllib.request`` (stdlib only).
    """

    # ------------------------------------------------------------------ HF --

    def lookup_hf(
        self,
        repo_id: str,
        *,
        token: str = "",
    ) -> RemoteModelInfo | None:
        """Fetch HuggingFace model-card metadata for *repo_id* (e.g. ``org/model``).

        Returns None on any network or API error.
        """
        repo_id = repo_id.strip().strip("/")
        if not repo_id or "/" not in repo_id:
            return None

        hf_token = (token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
        url = f"https://huggingface.co/api/models/{urllib.parse.quote(repo_id, safe='/')}"
        try:
            data = _http_get_json(url, token=hf_token)
        except Exception as exc:
            logger.debug("[ModelInfoLookup] HF lookup failed for %s: %s", repo_id, exc)
            return None

        if not isinstance(data, dict):
            return None

        # Parse tags — separate out pipeline/library tags from concept tags
        all_tags: list[str] = data.get("tags", [])
        skip_prefixes = ("license:", "arxiv:", "base_model:", "language:")
        concept_tags = [t for t in all_tags if not any(t.startswith(p) for p in skip_prefixes)]

        license_tag = next((t.removeprefix("license:") for t in all_tags if t.startswith("license:")), "")

        # Downloads
        downloads = int(data.get("downloads", 0) or 0)

        # Description from card_data
        card_data: dict = data.get("cardData", {}) or {}
        description = str(card_data.get("description", "") or "").strip()
        if not description:
            description = str(data.get("description", "") or "").strip()

        # Trigger words (common card_data field)
        trigger_words: list[str] = _coerce_list(card_data.get("trigger_words") or [])

        # Base model
        base_model_refs: list = _coerce_list(card_data.get("base_model") or [])
        base_model = ", ".join(str(b) for b in base_model_refs[:3]) if base_model_refs else ""

        return RemoteModelInfo(
            source="huggingface",
            name=str(data.get("modelId", repo_id)),
            description=description,
            tags=concept_tags[:20],
            trigger_words=trigger_words,
            base_model=base_model,
            license=license_tag,
            url=f"https://huggingface.co/{repo_id}",
            downloads=downloads,
            raw=data,
        )

    # --------------------------------------------------------------- CivitAI --

    def lookup_civitai(
        self,
        model_id_or_url: str,
        *,
        token: str = "",
    ) -> RemoteModelInfo | None:
        """Fetch CivitAI model info by model ID, version ID, or page URL.

        Returns None on any error or if the model is not found.
        """
        model_id_or_url = model_id_or_url.strip()
        if not model_id_or_url:
            return None

        civitai_token = (token or os.environ.get("CIVITAI_API_TOKEN") or "").strip()
        model_id, version_id = _parse_civitai_ref(model_id_or_url)
        if model_id is None:
            return None

        # Fetch model
        model_url = f"https://civitai.com/api/v1/models/{model_id}"
        try:
            model_data = _http_get_json(model_url, token=civitai_token, token_param="token")
        except Exception as exc:
            logger.debug("[ModelInfoLookup] CivitAI model lookup failed for %s: %s", model_id, exc)
            return None

        if not isinstance(model_data, dict):
            return None

        name = str(model_data.get("name", f"Model {model_id}"))
        description_html = str(model_data.get("description", "") or "")
        description = _strip_html(description_html)[:500]
        tags: list[str] = [str(t) for t in model_data.get("tags", [])]
        license_str = str(model_data.get("allowCommercialUse", "") or "")
        downloads = int(model_data.get("stats", {}).get("downloadCount", 0) or 0)

        # Find the version (latest or specific)
        versions: list = model_data.get("modelVersions", [])
        version: dict = {}
        if version_id is not None:
            version = next((v for v in versions if v.get("id") == version_id), {})
        if not version and versions:
            version = versions[0]  # latest

        base_model = str(version.get("baseModel", "") or "")
        trigger_words: list[str] = _coerce_list(version.get("trainedWords") or [])

        # Page URL
        page_url = f"https://civitai.com/models/{model_id}"
        if version.get("id"):
            page_url += f"?modelVersionId={version['id']}"

        return RemoteModelInfo(
            source="civitai",
            name=name,
            description=description,
            tags=tags[:20],
            trigger_words=trigger_words[:20],
            base_model=base_model,
            license=license_str,
            url=page_url,
            downloads=downloads,
            raw=model_data,
        )

    # --------------------------------------------------------------- Ollama --

    def lookup_ollama(
        self,
        model_name: str,
        *,
        base_url: str = "http://localhost:11434",
    ) -> RemoteModelInfo | None:
        """Fetch model details from a local Ollama server via ``/api/show``.

        Returns None if Ollama is not running or the model is not found.
        """
        model_name = model_name.strip()
        if not model_name:
            return None

        url = f"{base_url.rstrip('/')}/api/show"
        payload = json.dumps({"model": model_name}).encode()
        try:
            data = _http_post_json(url, payload)
        except Exception as exc:
            logger.debug("[ModelInfoLookup] Ollama lookup failed for %s: %s", model_name, exc)
            return None

        if not isinstance(data, dict):
            return None

        details: dict = data.get("details", {}) or {}
        model_info_block: dict = data.get("model_info", {}) or {}

        # Build description from parameters block (architecture summary)
        parameters_str = str(data.get("parameters", "") or "").strip()
        modelfile_str = str(data.get("modelfile", "") or "")
        description = ""
        if parameters_str:
            description = f"Parameters:\n{parameters_str[:300]}"

        family = str(details.get("family", "") or details.get("families", [""])[0] if details.get("families") else "")
        param_size = str(details.get("parameter_size", "") or "")
        quant = str(details.get("quantization_level", "") or "")

        tags: list[str] = []
        if family:
            tags.append(family)
        if param_size:
            tags.append(param_size)
        if quant:
            tags.append(quant)

        # Extract template / system for context
        system_prompt = str(data.get("system", "") or "").strip()
        if system_prompt and not description:
            description = system_prompt[:300]

        display_name = model_name
        if param_size:
            display_name = f"{model_name} ({param_size})"

        return RemoteModelInfo(
            source="ollama",
            name=display_name,
            description=description,
            tags=tags,
            base_model=family,
            raw=data,
        )

    # ------------------------------------------------------------ Auto-detect --

    def lookup_auto(
        self,
        query: str,
        *,
        hf_token: str = "",
        civitai_token: str = "",
        ollama_url: str = "http://localhost:11434",
    ) -> RemoteModelInfo | None:
        """Detect the source from *query* and dispatch to the right lookup method.

        Detection rules (in order):
        1. ``civitai.com`` in string → CivitAI lookup
        2. Numeric string (or ``models/<int>``) → CivitAI model ID
        3. ``org/repo`` (one slash, no ``.com``) → HuggingFace
        4. Otherwise → try Ollama, fall back to HuggingFace search by name
        """
        q = query.strip()
        if not q:
            return None

        # CivitAI URL
        if "civitai.com" in q.lower():
            return self.lookup_civitai(q, token=civitai_token)

        # Bare numeric ID → CivitAI
        if re.fullmatch(r"\d+", q):
            return self.lookup_civitai(q, token=civitai_token)

        # HF repo ID: exactly one slash, no domain
        if re.fullmatch(r"[^/\s]+/[^/\s]+", q) and ".com" not in q:
            return self.lookup_hf(q, token=hf_token)

        # Try Ollama first (fast, local)
        result = self.lookup_ollama(q, base_url=ollama_url)
        if result is not None:
            return result

        return None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_lookup: ModelInfoLookup | None = None


def get_model_info_lookup() -> ModelInfoLookup:
    """Return the process-global ModelInfoLookup instance."""
    global _lookup
    if _lookup is None:
        _lookup = ModelInfoLookup()
    return _lookup


# ---------------------------------------------------------------------------
# Internal HTTP helpers — stdlib only
# ---------------------------------------------------------------------------

def _http_get_json(
    url: str,
    *,
    token: str = "",
    token_param: str = "",      # if set, add token as query param instead of Bearer header
    timeout: int = 15,
) -> object:
    """GET *url*, return parsed JSON.  Raises on HTTP/network error."""
    if token and token_param:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}{token_param}={urllib.parse.quote(token)}"
        req = urllib.request.Request(url)
    elif token:
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    else:
        req = urllib.request.Request(url)

    req.add_header("User-Agent", "AIWF-Studio/1.0 (model-info-lookup)")
    req.add_header("Accept", "application/json")

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _http_post_json(url: str, body: bytes, *, timeout: int = 15) -> object:
    """POST *body* to *url*, return parsed JSON."""
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "AIWF-Studio/1.0 (model-info-lookup)",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


# ---------------------------------------------------------------------------
# CivitAI URL/ID parsing
# ---------------------------------------------------------------------------

def _parse_civitai_ref(ref: str) -> tuple[int | None, int | None]:
    """Return (model_id, version_id) from a CivitAI URL, model ID, or version ID.

    Returns (None, None) if the ref cannot be parsed.
    """
    ref = ref.strip()

    # Bare integer → model ID
    if re.fullmatch(r"\d+", ref):
        return int(ref), None

    # URL
    try:
        parsed = urllib.parse.urlparse(ref)
        if not any(h in parsed.netloc for h in ("civitai.com", "civitai.green")):
            return None, None

        # /models/<id> or /models/<id>/…
        m = re.search(r"/models/(\d+)", parsed.path)
        model_id = int(m.group(1)) if m else None

        # ?modelVersionId=<id>
        qs = urllib.parse.parse_qs(parsed.query)
        version_id_list = qs.get("modelVersionId", [])
        version_id = int(version_id_list[0]) if version_id_list else None

        return model_id, version_id
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def _coerce_list(value) -> list[str]:
    """Coerce a list, comma-string, or single value to list[str]."""
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and "," in value:
        return [v.strip() for v in value.split(",") if v.strip()]
    if value:
        return [str(value).strip()]
    return []


def _strip_html(text: str) -> str:
    """Very light HTML stripper — removes tags and decodes common entities."""
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
