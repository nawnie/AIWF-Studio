"""Data-only map from generation routes to their setup actions.

This registry describes how setup is offered; it is not a readiness check.
Each route's existing family-specific preflight remains authoritative.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSetupRoute:
    route_key: str
    modality: str
    architectures: tuple[str, ...]
    preflight_key: str
    bundle_key: str | None = None
    setup_action: str | None = None
    identity_contains: tuple[str, ...] = ()
    identity_requires: tuple[str, ...] = ()
    identity_excludes: tuple[str, ...] = ()
    requires_diffusers_snapshot: bool = False
    requires_gguf: bool = False
    requires_safetensors: bool = False
    identity_default: bool = False
    support_state: str = "supported"
    limitation: str = ""


# Ordered from specific variants to family defaults. These entries only
# recommend the setup action; they never mark assets or a runtime as ready.
MODEL_SETUP_ROUTES: tuple[ModelSetupRoute, ...] = (
    ModelSetupRoute("pro.image.flux2.9b", "image", ("flux2_klein",), "flux2_diffusers", "flux2-9b-components", identity_contains=("9b",)),
    ModelSetupRoute(
        "pro.image.flux2.4b-base-full-snapshot",
        "image",
        ("flux2_klein",),
        "flux2_diffusers",
        "flux2-4b-base",
        identity_requires=("base", "4b"),
        requires_diffusers_snapshot=True,
    ),
    ModelSetupRoute(
        "pro.image.flux2.4b-full-snapshot",
        "image",
        ("flux2_klein",),
        "flux2_diffusers",
        "flux2",
        identity_contains=("4b",),
        requires_diffusers_snapshot=True,
    ),
    ModelSetupRoute("pro.image.flux2.4b", "image", ("flux2_klein",), "flux2_diffusers", "flux2-4b-components", identity_contains=("4b",)),
    ModelSetupRoute("pro.image.flux-kontext", "image", ("flux_kontext",), "flux_kontext_diffusers", "flux-kontext", identity_default=True, limitation="Txt2img is routed; the current Pro checkpoint path does not expose image editing for this family."),
    ModelSetupRoute("pro.video.sana-video.720p", "video", ("sana_video",), "sana_video", "sana-video-720p", identity_contains=("sana-video_2b_720p", "sana video 720p")),
    ModelSetupRoute(
        "pro.image.flux-distillt5-control",
        "image",
        ("flux", "flux_fill"),
        "flux_conditioning",
        "flux-distillt5-control-components",
        identity_contains=("distillt5-control",),
        limitation="Installs DistillT5 and its T5-Base tokenizer; the standard Flux CLIP-L and VAE support assets are still required.",
    ),
    ModelSetupRoute(
        "pro.image.flux-universal-conditioning",
        "image",
        ("flux", "flux_fill"),
        "flux_conditioning",
        support_state="runtime-dependent",
        identity_contains=("universal",),
        limitation=(
            "Manual setup required: set AIWF_FLUX_UNIVERSAL_ADAPTER_PATH to a folder containing "
            "adapter_config.json and adapter.safetensors, set AIWF_FLUX_UNIVERSAL_CORE_MODEL to a "
            "complete local UMT5-base snapshot, and provide the `ute.integrations` runtime package. "
            "AIWF does not currently provide a verified installation source for that package; do not "
            "install it by package name alone. No catalog bundle can supply user-specific adapter weights."
        ),
    ),
    ModelSetupRoute("pro.image.flux-teacher", "image", ("flux", "flux_fill"), "flux_conditioning", "flux-components", identity_default=True),
    ModelSetupRoute("pro.video.wan.ti2v-diffusers", "video", ("wan",), "wan_diffusers", "wan-ti2v-diffusers", identity_contains=("ti2v",), identity_requires=("5b",), requires_diffusers_snapshot=True),
    ModelSetupRoute(
        "pro.video.wan.ti2v-standalone-safetensors",
        "video",
        ("wan",),
        "preflight_wan_pipeline",
        "wan-ti2v-support",
        identity_contains=("ti2v",),
        identity_requires=("5b",),
        requires_safetensors=True,
        limitation=(
            "The selected standalone TI2V 5B transformer remains a supported runtime format when its shared "
            "text_encoder/tokenizer/scheduler and VAE assets are installed. The catalog has no standalone "
            "checkpoint installer; this bundle installs the shared UMT5/text stack and the required Wan 2.2 48-channel VAE "
            "for the selected transformer and does not download or replace it."
        ),
    ),
    ModelSetupRoute("pro.video.wan.fp8-pair", "video", ("wan",), "preflight_wan_pipeline", "wan-14b-components", identity_contains=("high_noise", "highnoise", "low_noise", "lownoise"), identity_requires=("2.2", "14b", "fp8"), requires_safetensors=True),
    ModelSetupRoute("pro.video.wan.gguf-pair", "video", ("wan",), "preflight_wan_pipeline", "wan-14b-components", identity_contains=("high_noise", "highnoise", "low_noise", "lownoise"), identity_requires=("wan2.2", "i2v"), identity_excludes=("ti2v",), requires_gguf=True),
    ModelSetupRoute(
        "pro.image.krea2-raw-diffusers",
        "image",
        ("krea2", "krea_2"),
        "krea2",
        "krea2-raw",
        identity_contains=("raw",),
        support_state="runtime-dependent",
        limitation="Krea 2 Raw requires the full Diffusers snapshot; Comfy split-file assets do not provide this route.",
    ),
    ModelSetupRoute(
        "pro.image.krea2-diffusers",
        "image",
        ("krea2", "krea_2"),
        "krea2",
        "krea2",
        identity_default=True,
        support_state="runtime-dependent",
        limitation="The complete Diffusers folder route is implemented only when the installed runtime exposes Krea2Pipeline; Comfy split-file Krea 2 assets remain unsupported.",
    ),
    ModelSetupRoute("pro.image.zimage", "image", ("z_image", "zimage"), "zimage", "zimage", identity_default=True),
    ModelSetupRoute(
        "pro.image.qwen-2.1",
        "image",
        ("qwen_image",),
        "qwen_image_21_runtime",
        identity_contains=("qwenimage21pipeline", "qwen image 2.1", "qwen-image-2.1"),
        support_state="runtime-dependent",
        limitation="Readiness depends on whether the installed Diffusers runtime exposes QwenImage21Pipeline; the Pro readiness check reports the current capability.",
    ),
    ModelSetupRoute(
        "pro.image.qwen-nunchaku",
        "image",
        ("qwen_image_nunchaku",),
        "preflight_qwen_nunchaku_pipeline",
        "qwen-nunchaku",
        setup_action="POST /api/pro/engines/qwen_nunchaku/install",
        support_state="setup-available",
        limitation="Model setup is available. Image generation remains blocked until the installed runtime passes a real route smoke test.",
    ),
    ModelSetupRoute(
        "pro.image.qwen-original",
        "image",
        ("qwen_image", "qwen"),
        "qwen_image",
        "qwen-image-original",
        identity_contains=("qwen-image", "qwen image"),
        identity_excludes=("2512", "2.1"),
    ),
    ModelSetupRoute(
        "pro.image.qwen-2512",
        "image",
        ("qwen_image", "qwen"),
        "qwen_image",
        "qwen-image",
        identity_contains=("2512",),
    ),
    ModelSetupRoute("pro.image.qwen-diffusers", "image", ("qwen_image", "qwen"), "qwen_image", "qwen-image", identity_default=True),
    ModelSetupRoute("pro.image.anima", "image", ("anima",), "preflight_anima", support_state="unsupported", limitation="The current Pro runtime has no native Anima loader; catalog assets do not make this route runnable."),
    ModelSetupRoute("pro.image.onnx", "image", ("onnx",), "preflight_onnx_pipeline", support_state="settings-configured", setup_action="Configure Settings.onnx_model_dir to a root containing complete ONNX model folders."),
    ModelSetupRoute(
        "pro.image.sana-sprint-1.6b",
        "image",
        ("sana",),
        "sana",
        "sana-sprint-16b",
        identity_contains=("sana sprint 1.6b", "sana_sprint_1.6b", "sana sprint 1.6b"),
    ),
    ModelSetupRoute(
        "pro.image.sana-1.6b",
        "image",
        ("sana",),
        "sana",
        "sana-16b",
        identity_contains=("sana 1.6b", "sana_1600m", "sana 1600m"),
    ),
    ModelSetupRoute("pro.image.sana", "image", ("sana",), "sana", "sana", identity_default=True),
    ModelSetupRoute("pro.video.sana-video.480p", "video", ("sana_video",), "sana_video", "sana-video", identity_excludes=("sana-video_2b_720p", "sana video 720p"), identity_default=True),
    ModelSetupRoute("pro.image.sd15", "image", ("sd15", "inpaint"), "diffusers", "sd-components", identity_default=True),
    ModelSetupRoute("pro.image.sdxl", "image", ("sdxl", "sdxl_inpaint", "sdxl_refiner"), "diffusers", "sdxl-components", identity_default=True),
    ModelSetupRoute("pro.image.sd35", "image", ("sd35",), "diffusers", "sd35-components", identity_default=True),
    ModelSetupRoute("pro.audio.minimum", "audio", (), "audio_minimum", setup_action="POST /api/pro/audio/setup/minimum"),
    ModelSetupRoute("pro.audio.acestep.1-5-turbo", "audio", (), "acestep", setup_action="POST /api/pro/audio/setup/engine/acestep"),
    ModelSetupRoute("pro.audio.moss-sfx.v2", "audio", (), "moss-sfx", setup_action="POST /api/pro/audio/setup/engine/moss-sfx"),
    ModelSetupRoute("pro.audio.musicgen.small", "audio", (), "musicgen", setup_action="POST /api/pro/audio/setup/musicgen/small"),
    ModelSetupRoute("pro.audio.musicgen.medium", "audio", (), "musicgen", setup_action="POST /api/pro/audio/setup/musicgen/medium"),
    ModelSetupRoute("pro.audio.musicgen.melody", "audio", (), "musicgen", setup_action="POST /api/pro/audio/setup/musicgen/melody"),
    ModelSetupRoute("pro.audio.musicgen.stereo-small", "audio", (), "musicgen", setup_action="POST /api/pro/audio/setup/musicgen/stereo-small"),
    ModelSetupRoute("pro.audio.mmaudio.small-16k", "audio", (), "mmaudio", setup_action="POST /api/pro/audio/setup/mmaudio/small_16k"),
    ModelSetupRoute("pro.audio.mmaudio.large-44k-v2", "audio", (), "mmaudio", setup_action="POST /api/pro/audio/setup/mmaudio/large_44k_v2"),
    ModelSetupRoute("pro.audio.mmaudio.large-44k", "audio", (), "mmaudio", setup_action="POST /api/pro/audio/setup/mmaudio/large_44k"),
    ModelSetupRoute("pro.audio.mmaudio.medium-44k", "audio", (), "mmaudio", setup_action="POST /api/pro/audio/setup/mmaudio/medium_44k"),
    ModelSetupRoute("pro.audio.mmaudio.small-44k", "audio", (), "mmaudio", setup_action="POST /api/pro/audio/setup/mmaudio/small_44k"),
    ModelSetupRoute("pro.video.audio.mmaudio", "video", (), "mmaudio", setup_action="POST /api/pro/audio/setup/mmaudio/{variant}"),
    ModelSetupRoute(
        "pro.video.ltx.diffusers-2b",
        "video",
        ("ltx",),
        "preflight_ltx_pipeline",
        "ltx-2b",
        identity_contains=("diffusers_2b", "diffusers 2b"),
        limitation="This LTX 2B route supports text-to-video only; image-conditioned video is not implemented for this pipeline.",
    ),
    ModelSetupRoute("pro.video.ltx.distilled", "video", ("ltx",), "preflight_ltx_pipeline", "ltx23", setup_action="POST /api/pro/engines/ltx/install", identity_contains=("distilled",)),
    ModelSetupRoute(
        "pro.video.ltx.one-stage",
        "video",
        ("ltx",),
        "preflight_ltx_pipeline",
        "ltx23-one-stage",
        setup_action="POST /api/pro/engines/ltx/install",
        identity_contains=("one_stage", "one stage"),
        support_state="blocked-runtime",
        limitation=(
            "The native LTX 2.3 22B BF16 one-stage worker is blocked on Windows after a bounded smoke exited "
            "with access violation 3221225477. The bundle can stage files but does not make this route runnable; "
            "the working LTX 2B Diffusers route is the supported alternative."
        ),
    ),
    ModelSetupRoute(
        "pro.video.ltx",
        "video",
        ("ltx",),
        "preflight_ltx_pipeline",
        support_state="supported-when-folder-installed",
        limitation=(
            "LTX support depends on the exact selected pipeline and its local model/runtime assets. "
            "Use the supported Diffusers 2B or distilled route setup; this family entry has no single install bundle."
        ),
    ),
)


# ControlNet is an optional conditioner selected alongside an SD checkpoint,
# so it is not a base checkpoint route above. Keep its single-model setup
# choices here and resolve them against the normal download catalog.
CONTROLNET_SETUP_CATALOG_KEYS: dict[str, tuple[str, ...]] = {
    "sd15": (
        "cn15-canny", "cn15-depth", "cn15-openpose", "cn15-tile", "cn15-lineart", "cn15-softedge",
    ),
    "sdxl": ("cnxl-canny", "cnxl-openpose", "cnxl-depth", "cnxl-softedge"),
}


def resolve_setup_route(data: dict[str, object]) -> ModelSetupRoute | None:
    """Resolve the declared route for an identity, without asserting readiness."""
    explicit_route = str(data.get("route_key") or data.get("routeKey") or "").strip().lower()
    if explicit_route:
        return next((route for route in MODEL_SETUP_ROUTES if route.route_key == explicit_route), None)
    architecture = str(data.get("architecture") or "").strip().lower().replace("-", "_")
    if architecture in {"flux", "flux_fill"} and "flux_conditioning_mode" in data:
        if str(data.get("flux_conditioning_mode") or "").strip().lower() not in {
            "teacher", "distillt5_control", "distillt5-control", "universal",
        }:
            return None
    # Variant identity may use a checkpoint's own directory name, but not
    # arbitrary parent folders (for example, a generic checkpoint under /4b/).
    raw_path = str(data.get("path") or "")
    if architecture in {"krea2", "krea_2"} and Path(raw_path).suffix.casefold() in {".safetensors", ".gguf"}:
        # Comfy split files do not satisfy the Diffusers folder route. Avoid
        # recommending a different full snapshot while the split file remains selected.
        return None
    pipeline_class = ""
    snapshot_path = Path(raw_path).expanduser()
    if snapshot_path.is_dir():
        try:
            model_index = json.loads((snapshot_path / "model_index.json").read_text(encoding="utf-8"))
            if isinstance(model_index, dict):
                pipeline_class = str(model_index.get("_class_name") or "")
        except (OSError, ValueError):
            pass
    identity = " ".join(
        str(data.get(key) or "")
        for key in ("title", "filename", "pipeline", "runtime_mode")
    ) + " " + snapshot_path.name + " " + pipeline_class + " " + str(data.get("flux_conditioning_mode") or "")
    identity = identity.lower()
    for route in MODEL_SETUP_ROUTES:
        if architecture not in route.architectures:
            continue
        if route.identity_contains and not any(value in identity for value in route.identity_contains):
            continue
        if route.identity_requires and not all(value in identity for value in route.identity_requires):
            continue
        if route.identity_excludes and any(value in identity for value in route.identity_excludes):
            continue
        if route.requires_diffusers_snapshot:
            path = snapshot_path
            if not path.is_dir() or not (path / "model_index.json").is_file():
                continue
        if route.requires_gguf and Path(raw_path).suffix.casefold() != ".gguf":
            continue
        if route.requires_safetensors and Path(raw_path).suffix.casefold() != ".safetensors":
            continue
        if route.identity_default or route.identity_contains or not route.identity_excludes:
            return route
    return None


def setup_route_descriptor(data: dict[str, object]) -> dict[str, str | None] | None:
    """Return the safe setup/support contract for UI/API serialization.

    This is declarative metadata only: it does not establish readiness or
    backend residency. File paths and support asset details are intentionally
    omitted; the route's preflight remains authoritative for those facts.
    """
    route = resolve_setup_route(data)
    if route is None:
        return None
    support_state = route.support_state
    limitation = route.limitation or None
    if route.route_key == "pro.image.qwen-2.1":
        runtime_available = data.get("qwen_image_21_pipeline_available")
        if runtime_available is True:
            support_state = "supported-when-folder-installed"
            limitation = "Requires a complete local Qwen Image 2.1 Diffusers snapshot; this runtime exposes QwenImage21Pipeline."
        elif runtime_available is False:
            support_state = "blocked-runtime"
            limitation = "This route is blocked because the installed Diffusers runtime does not expose QwenImage21Pipeline."
        else:
            support_state = "runtime-dependent"
            limitation = "Readiness depends on whether the installed Diffusers runtime exposes QwenImage21Pipeline; the Pro readiness check reports the current capability."
    checkpoint_path = str(data.get("path") or data.get("filename") or "")
    setup_bundle_key = route.bundle_key
    if (
        route.route_key == "pro.image.flux-kontext"
        and Path(checkpoint_path).suffix.casefold() == ".gguf"
    ):
        setup_bundle_key = "flux-kontext-gguf-components"
        limitation = (
            "Installs the required Kontext text encoders, tokenizers, scheduler, VAE, and transformer config; "
            "the selected GGUF supplies transformer weights."
        )
    if (
        route.route_key == "pro.video.ltx.one-stage"
        and Path(checkpoint_path).name.casefold() == "ltx-2.3-22b-dev-fp8.safetensors"
    ):
        # The Windows access violation is specific to the BF16 checkpoint;
        # LtxService explicitly exempts the FP8 checkpoint from that blocker.
        support_state = "supported-when-folder-installed"
        setup_bundle_key = "ltx23-one-stage-fp8"
        limitation = (
            "The selected FP8 one-stage checkpoint is supported when its required local assets pass preflight. "
            "The 22B BF16 one-stage checkpoint remains blocked on Windows."
        )
    return {
        "routeKey": route.route_key,
        "modality": route.modality,
        "supportState": support_state,
        "preflightKey": route.preflight_key,
        "setupBundleKey": setup_bundle_key,
        "setupAction": route.setup_action,
        "limitation": limitation,
    }


def recommended_setup_bundle_key(data: dict[str, object]) -> str | None:
    """Return a known bundle for a checkpoint, without asserting readiness."""
    descriptor = setup_route_descriptor(data)
    return descriptor["setupBundleKey"] if descriptor is not None else None
