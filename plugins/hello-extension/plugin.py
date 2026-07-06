"""Template AIWF Studio extension.

Copy this folder, rename it (the folder name becomes the extension id), and
edit freely. AIWF loads every ``plugins/<folder>/plugin.py`` at startup and
calls ``plugin.on_load(ctx)`` with the full application context.

What ``ctx`` gives you (see aiwf/bootstrap.py AppContext for the full list):

- ``ctx.generation``  — image generation service (checkpoints, txt2img, ...)
- ``ctx.models``      — model catalog (checkpoints, LoRAs, VAEs)
- ``ctx.segment``     — SAM/DINO segmentation
- ``ctx.events``      — application event bus (subscribe to app events)
- ``ctx.settings``    — persisted user settings
- ``ctx.plugins``     — this registry: register API routes and Gradio tabs

Extension points:

1. REST API   — ``ctx.plugins.register_api(<id>, router)`` serves a FastAPI
   router under ``/api/ext/<id>/`` in AIWF Studio Pro.
2. Gradio tab — ``ctx.plugins.register_tab(name, factory)`` for the Lab UI.
3. Events     — ``ctx.events.subscribe(EventType, handler)``.

Disable/enable extensions from Settings -> System in the Pro app (or the
``disabled_extensions`` list in config.json). Changes apply on restart.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class HelloExtension:
    name = "Hello Extension"
    version = "1.0.0"
    description = "Template extension demonstrating the AIWF extension points."

    def on_load(self, ctx) -> None:
        # --- 1. REST API under /api/ext/hello-extension/ -------------------
        try:
            from fastapi import APIRouter

            router = APIRouter()

            @router.get("/hello")
            def hello():
                checkpoints = 0
                try:
                    checkpoints = len(ctx.generation.list_checkpoints())
                except Exception:
                    pass
                return {
                    "message": "Hello from a user extension!",
                    "checkpointsVisible": checkpoints,
                }

            ctx.plugins.register_api("hello-extension", router)
        except Exception:
            logger.exception("hello-extension: API registration failed")

        # --- 2. App events --------------------------------------------------
        try:
            from aiwf.core.events.types import AppStarted

            ctx.events.subscribe(AppStarted, lambda event: logger.info("hello-extension: app started"))
        except Exception:
            logger.debug("hello-extension: event hook skipped", exc_info=True)

        logger.info("hello-extension loaded (try GET /api/ext/hello-extension/hello)")


plugin = HelloExtension()
