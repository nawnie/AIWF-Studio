from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from aiwf.bootstrap import AppContext

logger = logging.getLogger(__name__)


class ClientErrorPayload(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    stack: str | None = Field(default=None, max_length=12000)
    source: str | None = Field(default=None, max_length=512)
    url: str | None = Field(default=None, max_length=2048)
    kind: str = Field(default="error", max_length=64)
    user_agent: str | None = Field(default=None, max_length=512)
    session_id: str | None = Field(default=None, max_length=64)
    context: dict[str, Any] | None = None


class ClientEventPayload(BaseModel):
    """Dev-only telemetry from the browser (clicks, queue, lifecycle)."""

    action: str = Field(min_length=1, max_length=128)
    detail: str | None = Field(default=None, max_length=4000)
    url: str | None = Field(default=None, max_length=2048)
    session_id: str | None = Field(default=None, max_length=64)
    context: dict[str, Any] | None = None


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _append_log(log_path: Path, lines: list[str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line)
        handle.write("\n")


def _write_json_log(log_path: Path, payload: dict[str, Any]) -> None:
    payload["logged_at"] = _utc_stamp()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_client_log_router(ctx: AppContext) -> APIRouter:
    router = APIRouter()

    @router.post("/client-errors")
    def report_client_error(payload: ClientErrorPayload) -> dict[str, str]:
        stamp = _utc_stamp()
        summary = payload.message.replace("\n", " ")[:240]
        logger.warning(
            "Browser %s: %s%s",
            payload.kind,
            summary,
            f" ({payload.source})" if payload.source else "",
        )
        if payload.session_id:
            logger.warning("  session: %s", payload.session_id)
        if payload.context:
            logger.warning("  context: %s", json.dumps(payload.context, ensure_ascii=False)[:500])
        if payload.stack:
            for line in payload.stack.splitlines()[:12]:
                logger.warning("  %s", line[:500])

        log_dir = ctx.flags.resolved_output_dir()
        human_path = log_dir / "client-errors.log"
        _append_log(
            human_path,
            [
                f"[{stamp}] {payload.kind}: {payload.message}",
                *( [f"  session: {payload.session_id}"] if payload.session_id else [] ),
                *( [f"  source: {payload.source}"] if payload.source else [] ),
                *( [f"  url: {payload.url}"] if payload.url else [] ),
                *( [payload.stack[:12000]] if payload.stack else [] ),
                *( [f"  context: {json.dumps(payload.context, ensure_ascii=False)}"] if payload.context else [] ),
            ],
        )
        _write_json_log(
            log_dir / "client-errors.jsonl",
            payload.model_dump(exclude_none=True),
        )

        return {"status": "logged"}

    @router.post("/client-events")
    def report_client_event(payload: ClientEventPayload) -> dict[str, str]:
        """Dev-only browser telemetry — actions, queue, lifecycle (not user-facing errors)."""
        detail = (payload.detail or "").replace("\n", " ")[:240]
        logger.info(
            "Browser event %s: %s%s",
            payload.action,
            detail or payload.action,
            f" (session {payload.session_id})" if payload.session_id else "",
        )

        log_dir = ctx.flags.resolved_output_dir()
        _write_json_log(
            log_dir / "client-events.jsonl",
            payload.model_dump(exclude_none=True),
        )
        return {"status": "logged"}

    return router