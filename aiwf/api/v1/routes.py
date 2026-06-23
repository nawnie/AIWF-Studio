from __future__ import annotations

import base64
import io
import json
from typing import TYPE_CHECKING, Any
from uuid import UUID

from fastapi import APIRouter, HTTPException
from PIL import Image
from pydantic import ValidationError

from aiwf.api.schemas import (
    BatchImg2ImgPayload,
    ControlNetDetectPayload,
    EnhancePayload,
    Img2ImgPayload,
    InpaintPayload,
    PlotPayload,
    Txt2ImgPayload,
)
from aiwf.core.domain.controlnet import ControlNetUnit
from aiwf.core.domain.enhance import RestoreOptions, UpscaleOptions
from aiwf.core.domain.generation import JobRecord, JobState
from aiwf.core.domain.generation import GenerationMode, GenerationRequest
from aiwf.core.infotext import normalize_sampler

if TYPE_CHECKING:
    from aiwf.bootstrap import AppContext


def _b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _decode(data: str) -> Image.Image:
    if data.startswith("data:image"):
        data = data.split(",", 1)[1]
    try:
        return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, "Invalid image payload") from exc


def _job_status(job: JobRecord) -> dict[str, Any]:
    progress = job.progress
    return {
        "id": str(job.id),
        "state": job.state.value,
        "progress": progress.percent if progress else (100 if job.state == JobState.COMPLETED else 0),
        "step": progress.step if progress else 0,
        "total_steps": progress.total_steps if progress else 0,
        "message": progress.message if progress else job.error or "",
        "has_result": job.result is not None,
        "error": job.error,
    }


def _a1111_alwayson_args(data: dict[str, Any], script_name: str) -> list[Any]:
    scripts = data.get("alwayson_scripts") or {}
    if not isinstance(scripts, dict):
        return []
    for name, script in scripts.items():
        if str(name).lower() != script_name.lower() or not isinstance(script, dict):
            continue
        args = script.get("args", [])
        return args if isinstance(args, list) else []
    return []


def _a1111_inpaint_mask_content(value: Any) -> str:
    mapping = {
        0: "fill",
        1: "original",
        2: "latent noise",
        3: "latent nothing",
        "0": "fill",
        "1": "original",
        "2": "latent noise",
        "3": "latent nothing",
        "fill": "fill",
        "original": "original",
        "latent noise": "latent noise",
        "latent_noise": "latent noise",
        "latent nothing": "latent nothing",
        "latent_nothing": "latent nothing",
    }
    return mapping.get(value, str(value or "original"))


def _a1111_controlnet_units(data: dict[str, Any]) -> list[dict[str, Any]]:
    raw_units = data.get("controlnet_units")
    if raw_units is None:
        raw_units = _a1111_alwayson_args(data, "controlnet")
    if not raw_units:
        return []
    if not isinstance(raw_units, list):
        raise HTTPException(422, "controlnet_units must be a list")

    units: list[dict[str, Any]] = []
    for raw in raw_units:
        if not isinstance(raw, dict):
            raise HTTPException(422, "ControlNet units must be objects")
        unit = dict(raw)
        image = unit.get("image")
        if isinstance(image, dict):
            if "image" in image:
                unit["image"] = image["image"]
            if "mask" in image and "mask" not in unit:
                unit["mask"] = image["mask"]
        if "input_image" in unit and "image" not in unit:
            unit["image"] = unit["input_image"]
        if "mask_image" in unit and "mask" not in unit:
            unit["mask"] = unit["mask_image"]
        if "detect_resolution" in unit and "processor_res" not in unit:
            unit["processor_res"] = unit["detect_resolution"]
        try:
            units.append(ControlNetUnit.model_validate(unit).model_dump())
        except ValidationError as exc:
            raise HTTPException(422, f"Invalid ControlNet unit: {exc}") from exc
    return units


def _a1111_generation_request(data: dict[str, Any], mode: GenerationMode) -> GenerationRequest:
    override_settings = data.get("override_settings") or {}
    sampler = normalize_sampler(data.get("sampler_name") or data.get("sampler")) or "euler_a"
    hr_denoising_strength = data.get("hr_denoising_strength")
    if hr_denoising_strength is None and mode == GenerationMode.TXT2IMG and "denoising_strength" in data:
        hr_denoising_strength = data["denoising_strength"]
    mapped = {
        "mode": mode,
        "prompt": data.get("prompt", ""),
        "negative_prompt": data.get("negative_prompt", ""),
        "steps": data.get("steps", 20),
        "cfg_scale": data.get("cfg_scale", 7.0),
        "width": data.get("width", 512),
        "height": data.get("height", 512),
        "seed": data.get("seed", -1),
        "sampler": sampler,
        "scheduler": str(data.get("scheduler") or "automatic").lower(),
        "batch_size": data.get("batch_size", 1),
        "batch_count": data.get("n_iter", data.get("batch_count", 1)),
        "denoising_strength": data.get("denoising_strength", 0.75),
        "mask_blur": data.get("mask_blur", data.get("mask_blur_x", 4)),
        "inpaint_only_masked": data.get("inpaint_full_res", data.get("inpaint_only_masked", False)),
        "inpaint_masked_padding": data.get("inpaint_full_res_padding", data.get("inpaint_masked_padding", 32)),
        "inpaint_mask_content": _a1111_inpaint_mask_content(
            data.get("inpainting_fill", data.get("inpaint_mask_content", "original"))
        ),
        "clip_skip": data.get(
            "clip_skip",
            data.get("CLIP_stop_at_last_layers", override_settings.get("CLIP_stop_at_last_layers", 1)),
        ),
        "enable_hr": data.get("enable_hr", False),
        "hr_scale": data.get("hr_scale", 2.0),
        "hr_steps": data.get("hr_second_pass_steps", data.get("hr_steps", 20)),
        "hr_denoising_strength": 0.35 if hr_denoising_strength is None else hr_denoising_strength,
        "hr_upscaler": data.get("hr_upscaler", "lanczos"),
        "save_before_hires": data.get("save_before_hires", False),
        "save_interrupted": data.get("save_interrupted", False),
        "sdxl_refiner_enabled": data.get("sdxl_refiner_enabled", False),
        "sdxl_refiner_checkpoint_id": data.get("sdxl_refiner_checkpoint_id"),
        "sdxl_refiner_steps": data.get("sdxl_refiner_steps", 10),
        "sdxl_refiner_strength": data.get("sdxl_refiner_strength", 0.25),
        "checkpoint_id": override_settings.get("sd_model_checkpoint") or data.get("checkpoint_id"),
        "vae_id": override_settings.get("sd_vae") or data.get("vae_id"),
        "save_images": not data.get("do_not_save_samples", False),
        "controlnet_units": _a1111_controlnet_units(data),
    }
    try:
        return GenerationRequest.model_validate(mapped)
    except ValidationError as exc:
        raise HTTPException(422, f"Invalid generation request: {exc}") from exc


def _a1111_response(job: JobRecord, parameters: dict[str, Any]) -> dict[str, Any]:
    if job.result is None:
        raise HTTPException(500, job.error or "Generation failed")
    return {
        "images": [_b64(img) for img in job.result.images],
        "parameters": parameters,
        "info": json.dumps(
            {
                "job_id": str(job.id),
                "infotexts": job.result.infotexts,
                "seeds": job.result.seeds,
            }
        ),
    }


def build_router(ctx: AppContext) -> APIRouter:
    router = APIRouter()
    native = APIRouter(prefix="/api/v1")
    sdapi = APIRouter(prefix="/sdapi/v1")

    @native.get("/health")
    def health():
        return {"status": "ok", "version": "0.1.0"}

    @native.get("/image/maturity")
    def image_maturity():
        from aiwf.services.image_lab import image_maturity_matrix

        return image_maturity_matrix().model_dump(mode="json")

    @native.get("/models")
    def models():
        return [c.model_dump() for c in ctx.generation.list_checkpoints()]

    @native.get("/samplers")
    def samplers():
        return [s.model_dump() for s in ctx.generation.list_samplers()]

    @native.get("/loras")
    def loras():
        return [item.model_dump() for item in ctx.generation.list_loras()]

    @native.get("/vae")
    def vaes():
        return [item.model_dump() for item in ctx.generation.list_vaes()]

    @native.get("/controlnet/models")
    def controlnet_models():
        return [item.model_dump() for item in ctx.controlnet.list_models()]

    @native.get("/controlnet/modules")
    def controlnet_modules():
        return ctx.controlnet.list_modules()

    @native.get("/plugins")
    def plugins():
        return [plugin.model_dump() for plugin in ctx.plugins.list_plugins()]

    @native.get("/optimization/status")
    def optimization_status():
        diagnostics = getattr(ctx, "optimization_diagnostics", None)
        if diagnostics is not None and callable(getattr(diagnostics, "status", None)):
            return diagnostics.status()
        profile_id = getattr(getattr(ctx, "settings", None), "optimization_profile_id", "balanced_sdpa_fp16")
        return {
            "profile_id": profile_id,
            "requested_profile_id": profile_id,
            "runtime_flags": {},
            "capability_report": {},
            "planner_decisions": [],
            "active_runtime_paths": [],
            "runtime_mismatches": ["Optimization diagnostics service is not configured."],
            "known_failures": [],
            "latest_receipts": [],
            "promotion_gates": {
                "status": "unavailable",
                "reason": "Optimization diagnostics service is not configured.",
                "candidate": False,
            },
        }

    @native.get("/jobs")
    def jobs(limit: int = 20):
        return [_job_status(job) for job in ctx.generation.recent_jobs(limit)]

    @native.get("/jobs/{job_id}")
    def job_status(job_id: UUID):
        job = ctx.generation.get_job(job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        return _job_status(job)

    @native.get("/progress")
    def progress():
        job = ctx.generation.active_job()
        if job is None:
            recent = ctx.generation.recent_jobs(1)
            return _job_status(recent[0]) if recent else {"state": "idle", "progress": 0}
        return _job_status(job)

    @native.post("/txt2img")
    def txt2img(payload: Txt2ImgPayload):
        request = GenerationRequest(
            mode=GenerationMode.TXT2IMG,
            **payload.model_dump(),
        )
        job = ctx.generation.submit(request)
        if job.result is None:
            raise HTTPException(500, job.error or "Generation failed")
        return {
            "job_id": str(job.id),
            "images": [_b64(img) for img in job.result.images],
            "infotexts": job.result.infotexts,
            "seeds": job.result.seeds,
        }

    @native.post("/img2img")
    def img2img(payload: Img2ImgPayload):
        if not payload.init_images:
            raise HTTPException(422, "init_images required")
        data = payload.model_dump()
        images = data.pop("init_images")
        request = GenerationRequest(mode=GenerationMode.IMG2IMG, **data)
        job = ctx.generation.submit(request, init_images=[_decode(item) for item in images])
        if job.result is None:
            raise HTTPException(500, job.error or "Generation failed")
        return {
            "job_id": str(job.id),
            "images": [_b64(img) for img in job.result.images],
            "infotexts": job.result.infotexts,
            "seeds": job.result.seeds,
        }

    @native.post("/img2img/batch")
    def img2img_batch(payload: BatchImg2ImgPayload):
        data = payload.model_dump()
        images = data.pop("init_images")
        results = []
        for image_data in images:
            request = GenerationRequest(mode=GenerationMode.IMG2IMG, **data)
            job = ctx.generation.submit(request, init_images=[_decode(image_data)])
            if job.result is None:
                raise HTTPException(500, job.error or "Generation failed")
            results.extend(_b64(img) for img in job.result.images)
        return {"images": results}

    @native.post("/xyz-plot")
    def xyz_plot(payload: PlotPayload):
        from aiwf.services.plot import PlotRequest

        request = PlotRequest(
            base=GenerationRequest(mode=GenerationMode.TXT2IMG, **payload.base.model_dump()),
            axes=[axis.model_dump() for axis in payload.axes],
        )
        result = ctx.plots.run(request)
        return {
            "labels": result.labels,
            "images": [_b64(img) for img in result.images],
            "grid": _b64(result.grid) if result.grid is not None else None,
            "infotexts": result.infotexts,
        }

    @native.post("/inpaint")
    def inpaint(payload: InpaintPayload):
        if not payload.init_images:
            raise HTTPException(422, "init_images required")
        if not payload.mask_image:
            raise HTTPException(422, "mask_image required")

        data = payload.model_dump()
        images = data.pop("init_images")
        mask_b64 = data.pop("mask_image")
        request = GenerationRequest(mode=GenerationMode.INPAINT, **data)

        init_image = _decode(images[0])
        from aiwf.infrastructure.diffusers.mask import prepare_inpaint_mask

        mask = prepare_inpaint_mask(_decode(mask_b64), size=init_image.size)
        if mask is None or mask.getbbox() is None:
            raise HTTPException(422, "mask_image must contain painted regions")

        job = ctx.generation.submit(request, init_images=[init_image], mask_images=[mask])
        if job.result is None:
            raise HTTPException(500, job.error or "Generation failed")
        return {
            "job_id": str(job.id),
            "images": [_b64(img) for img in job.result.images],
            "infotexts": job.result.infotexts,
            "seeds": job.result.seeds,
        }

    @native.get("/upscalers")
    def upscalers():
        return [m.model_dump() for m in ctx.enhance.list_upscalers()]

    @native.get("/restorers")
    def restorers():
        return [m.model_dump() for m in ctx.enhance.list_restorers()]

    @native.post("/enhance")
    def enhance(payload: EnhancePayload):
        image = _decode(payload.image)
        upscale = (
            UpscaleOptions(
                model_id=payload.upscaler_id,
                scale=payload.scale,
                tile_size=payload.tile_size,
                tile_overlap=payload.tile_overlap,
            )
            if payload.upscaler_id
            else None
        )
        restore = (
            RestoreOptions(
                model_id=payload.restorer_id,
                visibility=payload.restore_visibility,
                codeformer_weight=payload.codeformer_weight,
            )
            if payload.restorer_id
            else None
        )
        if upscale is None and restore is None:
            raise HTTPException(422, "Provide an upscaler_id and/or restorer_id")
        result_image, infotext = ctx.enhance.run_pipeline(image, upscale=upscale, restore=restore)
        return {"image": _b64(result_image), "infotext": infotext}

    @native.post("/controlnet/detect")
    def controlnet_detect(payload: ControlNetDetectPayload):
        control = ctx.controlnet.preprocess(
            _decode(payload.image),
            payload.module,
            processor_res=payload.processor_res,
            threshold_a=payload.threshold_a,
            threshold_b=payload.threshold_b,
        )
        return {"images": [_b64(control)]}

    @native.get("/controlnet/downloadable")
    def controlnet_downloadable():
        return [
            {
                "key": item.key,
                "title": item.title,
                "filename": item.filename,
                "preprocessor": item.preprocessor,
                "size_mb": item.size_mb,
                "installed": ctx.controlnet.is_installed(item),
            }
            for item in ctx.controlnet.list_downloadable()
        ]

    @native.get("/models/download/categories")
    def model_download_categories():
        return [
            {"key": key, "label": label, "folder": str(ctx.model_download.destination_dir(key))}
            for label, key in ctx.model_download.category_choices()
        ]

    @native.get("/models/download/catalog")
    def model_download_catalog():
        return [
            {
                "key": item.key,
                "title": item.title,
                "category": item.category,
                "source": item.source,
                "size_mb": item.size_mb,
                "notes": item.notes,
                "installed": ctx.model_download.is_catalog_installed(item),
            }
            for item in ctx.model_download.list_catalog()
        ]

    @native.post("/interrupt")
    def interrupt():
        ctx.generation.interrupt()
        return {"status": "interrupt_requested"}

    @sdapi.post("/txt2img")
    def sdapi_txt2img(payload: dict[str, Any]):
        request = _a1111_generation_request(payload, GenerationMode.TXT2IMG)
        job = ctx.generation.submit(request)
        return _a1111_response(job, payload)

    @sdapi.post("/img2img")
    def sdapi_img2img(payload: dict[str, Any]):
        request = _a1111_generation_request(payload, GenerationMode.INPAINT if payload.get("mask") else GenerationMode.IMG2IMG)
        images = payload.get("init_images") or []
        if not images:
            raise HTTPException(422, "init_images required")
        init_image = _decode(images[0])
        mask_images = None
        if payload.get("mask"):
            from aiwf.infrastructure.diffusers.mask import prepare_inpaint_mask

            mask = prepare_inpaint_mask(_decode(payload["mask"]), size=init_image.size)
            if mask is None or mask.getbbox() is None:
                raise HTTPException(422, "mask must contain painted regions")
            mask_images = [mask]
        job = ctx.generation.submit(request, init_images=[init_image], mask_images=mask_images)
        return _a1111_response(job, payload)

    @sdapi.get("/progress")
    def sdapi_progress():
        job = ctx.generation.active_job()
        if job is None:
            recent = ctx.generation.recent_jobs(1)
            job = recent[0] if recent else None
        if job is None or job.progress is None:
            return {"progress": 0, "eta_relative": 0, "state": {"job": None}, "current_image": None}
        return {
            "progress": job.progress.percent / 100,
            "eta_relative": 0,
            "state": _job_status(job),
            "current_image": _b64(job.progress.current_image) if job.progress.current_image else None,
        }

    @sdapi.post("/interrupt")
    def sdapi_interrupt():
        ctx.generation.interrupt()
        return {}

    @sdapi.get("/options")
    def sdapi_options():
        checkpoint = ctx.generation.resolve_checkpoint(None)
        return {
            "sd_model_checkpoint": checkpoint.title,
            "sd_vae": None,
            "CLIP_stop_at_last_layers": 1,
            "samples_save": ctx.settings.save_images,
        }

    @sdapi.post("/options")
    def sdapi_set_options(_payload: dict[str, Any]):
        return {}

    @sdapi.get("/sd-models")
    def sdapi_models():
        return [
            {
                "title": model.title,
                "model_name": model.title,
                "hash": model.hash or "",
                "sha256": model.hash or "",
                "filename": model.path,
                "config": None,
            }
            for model in ctx.generation.list_checkpoints()
        ]

    @sdapi.get("/sd-vae")
    def sdapi_vaes():
        return [{"model_name": vae.title, "filename": vae.path} for vae in ctx.generation.list_vaes()]

    @sdapi.get("/loras")
    def sdapi_loras():
        return [{"name": lora.title, "alias": lora.title, "path": lora.path} for lora in ctx.generation.list_loras()]

    @sdapi.get("/samplers")
    def sdapi_samplers():
        return [{"name": sampler.label, "aliases": [sampler.id], "options": {}} for sampler in ctx.generation.list_samplers()]

    @sdapi.get("/memory")
    def sdapi_memory():
        try:
            import psutil

            vm = psutil.virtual_memory()
            return {"ram": {"free": vm.available, "total": vm.total}, "cuda": {"system": {"free": 0, "total": 0}}}
        except Exception:
            return {"ram": {"free": 0, "total": 0}, "cuda": {"system": {"free": 0, "total": 0}}}

    @sdapi.post("/controlnet/detect")
    def sdapi_controlnet_detect(payload: dict[str, Any]):
        images = payload.get("controlnet_input_images") or []
        if not images:
            raise HTTPException(422, "controlnet_input_images required")
        module = payload.get("controlnet_module", "canny")
        control = ctx.controlnet.preprocess(
            _decode(images[0]),
            module,
            processor_res=int(payload.get("controlnet_processor_res", 512)),
            threshold_a=float(payload.get("controlnet_threshold_a", 100.0)),
            threshold_b=float(payload.get("controlnet_threshold_b", 200.0)),
        )
        return {"images": [_b64(control)], "info": "Success"}

    @sdapi.get("/controlnet/model_list")
    def sdapi_controlnet_model_list():
        return {"model_list": ctx.controlnet.model_ids()}

    @sdapi.get("/controlnet/module_list")
    def sdapi_controlnet_module_list():
        return {"module_list": ctx.controlnet.list_modules()}

    @sdapi.get("/extensions")
    def sdapi_extensions():
        return [
            {
                "name": plugin.name,
                "remote": None,
                "branch": None,
                "commit_hash": None,
                "version": plugin.version,
                "enabled": plugin.enabled,
            }
            for plugin in ctx.plugins.list_plugins()
        ]

    router.include_router(native)
    router.include_router(sdapi)
    return router
