# AIWF Studio Extensions

AIWF Studio loads user extensions from the `plugins/` folder at startup. An
extension is a plain folder — no packaging, no pip install:

```
plugins/
  my-extension/
    plugin.py      # required: the entry point
    plugin.json    # optional: name/version/description metadata
```

Copy `plugins/hello-extension/` to start; the folder name becomes the
extension id.

## Entry point

`plugin.py` must expose either a `plugin` object with `on_load(ctx)` or a
module-level `setup(ctx)` function. It runs once at app startup with the full
application context:

```python
class MyExtension:
    name = "My Extension"
    version = "1.0.0"

    def on_load(self, ctx) -> None:
        ...

plugin = MyExtension()
```

## Extension points

### 1. REST API routes (AIWF Studio Pro)

Register a FastAPI router; it is served under `/api/ext/<extension-id>/`:

```python
from fastapi import APIRouter

router = APIRouter()

@router.get("/status")
def status():
    return {"ok": True}

ctx.plugins.register_api("my-extension", router)
```

### 2. Application events

```python
from aiwf.core.events.types import AppStarted

ctx.events.subscribe(AppStarted, lambda event: print("app is up"))
```

### 3. Gradio Lab tabs

```python
ctx.plugins.register_tab("My Tab", build_my_tab)  # factory receives ctx
```

### 4. Services on ctx

The context exposes the same services the built-in UI uses — a few examples:

| Attribute        | What it does                                   |
| ---------------- | ---------------------------------------------- |
| `ctx.generation` | image generation, checkpoint list/load         |
| `ctx.models`     | model catalog (checkpoints, LoRAs, VAEs)       |
| `ctx.segment`    | SAM + GroundingDINO masking                    |
| `ctx.faceswap`   | ReActor face swap                              |
| `ctx.settings`   | persisted user settings                        |
| `ctx.flags`      | runtime flags and resolved data/model paths    |

See `AppContext` in `aiwf/bootstrap.py` for the full list.

## Managing extensions

- **Settings → System → Extensions** in AIWF Studio Pro lists every detected
  extension with its load status, lets you enable/disable each one, and shows
  load errors inline. Changes apply on the next restart.
- Disabled extensions are stored in `config.json` under
  `disabled_extensions` (by folder name) and are never imported.
- A crashing extension never blocks startup: it is listed with its error and
  the rest of the app loads normally.
- Folders starting with `_` or `.` are ignored.

## Safety notes

Extensions are Python code running inside the app process with your
permissions — only install extensions you trust, exactly like A1111/Comfy
custom nodes.
