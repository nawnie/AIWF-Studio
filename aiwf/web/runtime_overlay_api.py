from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from aiwf.core.domain.runtime_overlay import RuntimeOverlayReceipt, RuntimeOverlayValidateRequest
from aiwf.services.runtime_overlay import RuntimeOverlayService


def build_runtime_overlay_router(ctx: Any) -> APIRouter:
    router = APIRouter(prefix="/api/pro/runtime-overlays", tags=["runtime-overlays"])
    service = RuntimeOverlayService(ctx)

    @router.get("/registry")
    def registry() -> dict:
        return service.registry().model_dump(mode="json", by_alias=True)

    @router.post("/validate")
    def validate(payload: RuntimeOverlayValidateRequest) -> dict:
        return service.validate(payload).model_dump(mode="json", by_alias=True)

    @router.get("/receipts")
    def receipts(limit: int = Query(default=80, ge=1, le=500)) -> dict:
        return {
            "receipts": [item.model_dump(mode="json", by_alias=True) for item in service.receipts(limit=limit)]
        }

    @router.post("/receipts")
    def write_receipt(payload: RuntimeOverlayReceipt) -> dict:
        receipt = service.write_receipt(payload)
        return {
            "receipt": receipt.model_dump(mode="json", by_alias=True),
            "receipts": [item.model_dump(mode="json", by_alias=True) for item in service.receipts()],
        }

    return router
