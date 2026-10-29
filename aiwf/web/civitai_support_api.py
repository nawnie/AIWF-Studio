"""Read-only API for the local Civitai resource capability map."""
from __future__ import annotations

from fastapi import APIRouter

from aiwf.services.civitai_support import civitai_support_catalog


def build_civitai_support_router() -> APIRouter:
    router = APIRouter()

    @router.get("/api/pro/civitai/support")
    def civitai_support() -> dict:
        return civitai_support_catalog()

    return router
