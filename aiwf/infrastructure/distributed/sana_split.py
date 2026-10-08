from __future__ import annotations

import hashlib
import http.client
import ipaddress
import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

SANA_SPLIT_PROTOCOL_VERSION = "1"
SANA_SPLIT_MEDIA_TYPE = "application/vnd.aiwf.sana-embeddings+safetensors"
SANA_SPLIT_MAX_RESPONSE_BYTES = 64 * 1024 * 1024


class SanaSplitError(RuntimeError):
    pass


def _is_private_url(url: str) -> bool:
    parsed = urllib.parse.urlparse((url or "").strip())
    if parsed.scheme not in {"http", "https"}:
        return True
    host = parsed.hostname
    if not host:
        return True
    try:
        addresses = [ipaddress.ip_address(host)]
    except ValueError:
        try:
            infos = socket.getaddrinfo(host, None)
        except OSError:
            return False
        addresses = []
        for info in infos:
            sockaddr = info[4]
            if not sockaddr:
                continue
            try:
                addresses.append(ipaddress.ip_address(sockaddr[0]))
            except ValueError:
                continue
    return any(
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
        for address in addresses
    )


@dataclass(frozen=True)
class SanaEncoding:
    prompt_embeds: torch.Tensor
    prompt_attention_mask: torch.Tensor
    model_fingerprint: str


def _required_json(path: Path) -> bytes:
    if not path.is_file():
        raise SanaSplitError(f"Required Sana component metadata is missing: {path}")
    return path.read_bytes()


def sana_model_fingerprint(model_root: str | Path) -> str:
    root = Path(model_root).expanduser().resolve()
    model_index = json.loads(_required_json(root / "model_index.json"))
    if model_index.get("_class_name") != "SanaSprintPipeline":
        raise SanaSplitError(f"Expected a SanaSprintPipeline snapshot at {root}.")

    digest = hashlib.sha256()
    metadata_paths = (
        root / "model_index.json",
        root / "text_encoder" / "config.json",
        root / "tokenizer" / "tokenizer_config.json",
    )
    for path in metadata_paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = _required_json(path)
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)

    weight_files = sorted((root / "text_encoder").glob("*.safetensors"))
    if not weight_files:
        raise SanaSplitError(f"No text-encoder safetensors were found under {root}.")
    for path in weight_files:
        stat = path.stat()
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(int(stat.st_size).to_bytes(8, "little"))
    return digest.hexdigest()


def pack_sana_encoding(
    prompt_embeds: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    *,
    model_fingerprint: str,
) -> bytes:
    if not model_fingerprint:
        raise SanaSplitError("The encoder response needs a model fingerprint.")
    tensors = {
        "prompt_embeds": prompt_embeds.detach().to("cpu").contiguous(),
        "prompt_attention_mask": prompt_attention_mask.detach().to("cpu").contiguous(),
    }
    return save_safetensors(
        tensors,
        metadata={
            "protocol_version": SANA_SPLIT_PROTOCOL_VERSION,
            "model_fingerprint": model_fingerprint,
        },
    )


def unpack_sana_encoding(payload: bytes, *, model_fingerprint: str) -> SanaEncoding:
    if not payload or len(payload) > SANA_SPLIT_MAX_RESPONSE_BYTES:
        raise SanaSplitError("The encoder response size is invalid.")
    try:
        tensors = load_safetensors(payload)
    except Exception as exc:
        raise SanaSplitError(f"The encoder returned invalid safetensors: {exc}") from exc
    if set(tensors) != {"prompt_embeds", "prompt_attention_mask"}:
        raise SanaSplitError("The encoder response has an unexpected tensor contract.")

    prompt_embeds = tensors["prompt_embeds"]
    prompt_attention_mask = tensors["prompt_attention_mask"]
    if prompt_embeds.ndim != 3 or prompt_attention_mask.ndim != 2:
        raise SanaSplitError("The encoder response has invalid tensor ranks.")
    if prompt_embeds.shape[:2] != prompt_attention_mask.shape:
        raise SanaSplitError("The prompt embedding and attention-mask shapes do not match.")
    if not prompt_embeds.is_floating_point() or not torch.isfinite(prompt_embeds).all().item():
        raise SanaSplitError("The prompt embeddings must be finite floating-point values.")
    if prompt_attention_mask.dtype not in {
        torch.bool,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise SanaSplitError("The prompt attention mask must use a boolean or integer dtype.")
    return SanaEncoding(
        prompt_embeds=prompt_embeds,
        prompt_attention_mask=prompt_attention_mask,
        model_fingerprint=model_fingerprint,
    )


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001, ANN201
        return None


class SanaEncoderClient:
    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        timeout: float = 120.0,
        opener: Any | None = None,
    ) -> None:
        normalized = (base_url or "").strip().rstrip("/")
        if not normalized or not _is_private_url(normalized):
            raise SanaSplitError("The Sana encoder endpoint must resolve to loopback or a private address.")
        if len((token or "").strip()) < 24:
            raise SanaSplitError("The Sana encoder token must contain at least 24 characters.")
        self.base_url = normalized
        self.token = token.strip()
        self.timeout = max(1.0, float(timeout))
        self._opener = opener or urllib.request.build_opener(_NoRedirectHandler())

    def _open(self, request: urllib.request.Request):
        try:
            return self._opener.open(request, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read(4096).decode("utf-8", errors="replace")
            raise SanaSplitError(f"Sana encoder HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise SanaSplitError(f"Sana encoder connection failed: {exc.reason}") from exc
        except (http.client.HTTPException, OSError) as exc:
            raise SanaSplitError(f"Sana encoder connection failed: {exc}") from exc

    def health(self) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/healthz",
            headers={"Authorization": f"Bearer {self.token}"},
        )
        with self._open(request) as response:
            payload = response.read(256 * 1024)
        try:
            result = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise SanaSplitError("The Sana encoder health response was not valid JSON.") from exc
        if result.get("protocol_version") != SANA_SPLIT_PROTOCOL_VERSION:
            raise SanaSplitError("The Sana encoder protocol version does not match this client.")
        return result

    def encode(
        self,
        prompt: str,
        *,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 300,
    ) -> SanaEncoding:
        body = json.dumps(
            {
                "prompt": prompt,
                "num_images_per_prompt": int(num_images_per_prompt),
                "max_sequence_length": int(max_sequence_length),
            },
            separators=(",", ":"),
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/v1/encode/sana-sprint",
            data=body,
            method="POST",
            headers={
                "Accept": SANA_SPLIT_MEDIA_TYPE,
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
        )
        with self._open(request) as response:
            if response.headers.get_content_type() != SANA_SPLIT_MEDIA_TYPE:
                raise SanaSplitError("The Sana encoder returned an unexpected media type.")
            protocol = response.headers.get("X-AIWF-Protocol-Version", "")
            fingerprint = response.headers.get("X-AIWF-Model-Fingerprint", "")
            if protocol != SANA_SPLIT_PROTOCOL_VERSION or not fingerprint:
                raise SanaSplitError("The Sana encoder response is missing compatibility headers.")
            payload = response.read(SANA_SPLIT_MAX_RESPONSE_BYTES + 1)
        return unpack_sana_encoding(payload, model_fingerprint=fingerprint)
