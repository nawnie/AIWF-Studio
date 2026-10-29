from __future__ import annotations

from importlib import metadata


STARTUP_MESSAGES = {
    "starting": "Starting AIWF Studio...",
    "checking_runtime": "Checking your hardware and local AI tools...",
    "allocator_cuda": "GPU memory manager is enabled.",
    "genlog_enabled": "Generation data logging is enabled: {path}",
    "wan_flags": "Video speed options enabled: {flags}.",
    "using_device": "Using {device}.",
    "loading_library": "Loading your model library...",
    "building_workspace": "Building the workspace...",
    "ready": "AIWF Studio is ready.",
    "open_browser": "Open in your browser: {url}",
    "phone_access": "Phone and tablet access: {url}",
    "phone_access_off": "Phone and tablet access is off for now. Turn it on later in Settings.",
    "share_link": "Public share link: {url}",
}

VIDEO_MESSAGES = {
    "complete": "Created {frames} frames at {width}x{height} in {seconds:.1f}s.",
}


def sageattention_version() -> str | None:
    try:
        return metadata.version("sageattention")
    except Exception:
        return None


def sageattention_2_available() -> bool:
    version = sageattention_version()
    if not version:
        return False
    try:
        major = int(version.split(".", 1)[0])
    except Exception:
        return False
    if major < 2:
        return False
    try:
        import sageattention  # noqa: F401

        return True
    except Exception:
        return False


def attention_display_label(flags) -> str:
    backend = str(getattr(flags, "attention_backend", "") or "").strip().lower().replace("-", "_")
    if getattr(flags, "xformers", False) or backend == "xformers":
        return "xFormers"
    if backend in {"sdpa", "sdp"} or getattr(flags, "opt_sdp_attention", False) or getattr(flags, "opt_split_attention", False):
        return "SDPA"
    if backend in {"sage", "sageattention", "sage_sdpa"}:
        return "Sage" if sageattention_2_available() else "SDPA"
    if backend == "none":
        return "Default"
    return "SDPA"
