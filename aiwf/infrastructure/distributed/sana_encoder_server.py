from __future__ import annotations

import gc
import secrets
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import psutil
import torch
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from aiwf.api.security import ApiRateLimitMiddleware
from aiwf.infrastructure.distributed.sana_split import (
    SANA_SPLIT_MEDIA_TYPE,
    SANA_SPLIT_PROTOCOL_VERSION,
    SanaEncoding,
    pack_sana_encoding,
    sana_model_fingerprint,
)


class EncoderRuntime(Protocol):
    model_fingerprint: str

    def encode(
        self,
        prompt: str,
        *,
        num_images_per_prompt: int,
        max_sequence_length: int,
    ) -> SanaEncoding: ...

    def health(self) -> dict[str, object]: ...


@dataclass
class SanaSprintEncoderRuntime:
    pipe: object
    device: torch.device
    dtype: torch.dtype
    model_fingerprint: str
    _lock: threading.Lock

    @classmethod
    def load(
        cls,
        model_root: str | Path,
        *,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        min_free_ram_gib: float = 10.0,
    ) -> "SanaSprintEncoderRuntime":
        root = Path(model_root).expanduser().resolve()
        free_ram_gib = psutil.virtual_memory().available / (1024.0**3)
        if free_ram_gib < float(min_free_ram_gib):
            raise RuntimeError(
                f"Refusing the encoder load with only {free_ram_gib:.2f} GiB free RAM; "
                f"{float(min_free_ram_gib):.2f} GiB is required."
            )

        target = torch.device(device)
        if target.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("The first Sana split test requires a CUDA laptop GPU.")
        if target.index is not None and target.index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device {target} is not available.")
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(dtype)
        if torch_dtype is None:
            raise ValueError("dtype must be bfloat16 or float16")

        from diffusers import SCMScheduler, SanaSprintPipeline
        from transformers import Gemma2Model, GemmaTokenizerFast

        tokenizer = GemmaTokenizerFast.from_pretrained(
            str(root / "tokenizer"),
            local_files_only=True,
        )
        load_kwargs = {
            "local_files_only": True,
            "low_cpu_mem_usage": True,
            "dtype": torch_dtype,
        }
        try:
            text_encoder = Gemma2Model.from_pretrained(str(root / "text_encoder"), **load_kwargs)
        except TypeError:
            load_kwargs["torch_dtype"] = load_kwargs.pop("dtype")
            text_encoder = Gemma2Model.from_pretrained(str(root / "text_encoder"), **load_kwargs)
        scheduler = SCMScheduler.from_pretrained(str(root / "scheduler"), local_files_only=True)
        text_encoder.to(target)
        text_encoder.eval()
        pipe = SanaSprintPipeline(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=None,
            transformer=None,
            scheduler=scheduler,
        )
        pipe.set_progress_bar_config(disable=True)
        gc.collect()
        torch.cuda.synchronize(target)
        return cls(
            pipe=pipe,
            device=target,
            dtype=torch_dtype,
            model_fingerprint=sana_model_fingerprint(root),
            _lock=threading.Lock(),
        )

    def encode(
        self,
        prompt: str,
        *,
        num_images_per_prompt: int,
        max_sequence_length: int,
    ) -> SanaEncoding:
        with self._lock, torch.inference_mode():
            prompt_embeds, prompt_attention_mask = self.pipe.encode_prompt(
                prompt=prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=self.device,
                clean_caption=False,
                max_sequence_length=max_sequence_length,
            )
        return SanaEncoding(
            prompt_embeds=prompt_embeds.detach().to("cpu").contiguous(),
            prompt_attention_mask=prompt_attention_mask.detach().to("cpu").contiguous(),
            model_fingerprint=self.model_fingerprint,
        )

    def health(self) -> dict[str, object]:
        index = self.device.index or 0
        return {
            "ready": True,
            "protocol_version": SANA_SPLIT_PROTOCOL_VERSION,
            "model_fingerprint": self.model_fingerprint,
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
            "gpu_name": torch.cuda.get_device_name(index),
            "vram_allocated_mib": round(torch.cuda.memory_allocated(index) / (1024.0**2), 1),
            "vram_reserved_mib": round(torch.cuda.memory_reserved(index) / (1024.0**2), 1),
            "available_ram_gib": round(psutil.virtual_memory().available / (1024.0**3), 2),
            "disk_offload": False,
        }


class EncodeRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=8000)
    num_images_per_prompt: int = Field(default=1, ge=1, le=4)
    max_sequence_length: int = Field(default=300, ge=1, le=300)


def create_sana_encoder_app(
    runtime: EncoderRuntime,
    *,
    token: str,
    on_encode: Callable[[], None] | None = None,
) -> FastAPI:
    expected_token = (token or "").strip()
    if len(expected_token) < 24:
        raise ValueError("The Sana encoder token must contain at least 24 characters.")

    def authorize(authorization: str | None = Header(default=None)) -> None:
        expected = f"Bearer {expected_token}"
        if not authorization or not secrets.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="Missing or invalid Sana encoder token.")

    app = FastAPI(
        title="AIWF Sana Sprint Encoder",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.add_middleware(ApiRateLimitMiddleware, requests_per_minute=60)

    @app.get("/healthz", dependencies=[Depends(authorize)])
    def health() -> dict[str, object]:
        return runtime.health()

    @app.post("/api/v1/encode/sana-sprint", dependencies=[Depends(authorize)])
    def encode(request: EncodeRequest) -> Response:
        result = runtime.encode(
            request.prompt,
            num_images_per_prompt=request.num_images_per_prompt,
            max_sequence_length=request.max_sequence_length,
        )
        payload = pack_sana_encoding(
            result.prompt_embeds,
            result.prompt_attention_mask,
            model_fingerprint=result.model_fingerprint,
        )
        if on_encode is not None:
            on_encode()
        return Response(
            content=payload,
            media_type=SANA_SPLIT_MEDIA_TYPE,
            headers={
                "X-AIWF-Protocol-Version": SANA_SPLIT_PROTOCOL_VERSION,
                "X-AIWF-Model-Fingerprint": result.model_fingerprint,
                "Cache-Control": "no-store",
            },
        )

    return app
