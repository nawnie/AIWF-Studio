"""Build and validate ComfyUI API-format prompts for Qwen-Image 2.1.

The builder mirrors the three official Comfy-Org templates (text to image,
image edit with references, background removal) and adds the optional
pieces the model supports natively in ComfyUI core:

* LoRA stacking through ``LoraLoaderModelOnly``;
* few-step accelerator LoRAs (plain KSampler or ``ManualSigmas`` +
  ``SamplerCustom`` for the distills that ship an exact sigma schedule);
* the experimental ``QwenImage21Cache`` KV-cache node for memory-starved edits;
* the Fun ControlNet Union model patch (control map and/or inpaint + mask);
* the APG + FreSca guidance stack used by the community "Fix" recipe;
* the Qwen3.5-9B prompt enhancers through ``TextGenerate``.

The output is a plain ``dict`` ready for ``POST /prompt``. ``validate_prompt``
checks link integrity and node signatures offline, and against a live
``/object_info`` when one is supplied.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from . import presets as P

Prompt = dict[str, dict[str, Any]]
Link = list  # ["node_id", slot]

MAX_REFERENCES = 16  # TextEncodeQwenImage21 exposes image_1 .. image_16
RECOMMENDED_MAX_REFERENCES = 10  # model card: up to 10 reference images


@dataclass
class LoraEntry:
    name: str  # filename as listed under models/loras
    strength: float = 1.0


@dataclass
class ControlNetSpec:
    patch: str  # filename under models/model_patches
    control_image: str | None = None  # uploaded input filename (pre-processed map)
    inpaint_image: str | None = None  # uploaded input filename to keep outside the mask
    mask_image: str | None = None  # uploaded grayscale PNG; white = regenerate
    strength: float = 1.0
    start_percent: float = 0.0
    end_percent: float = 1.0


@dataclass
class GenerationSpec:
    mode: str = "t2i"  # "t2i" | "edit"
    prompt: str = ""
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    batch_size: int = 1
    canvas_mode: str = "match_ref"  # edit only: "match_ref" | "custom"
    seed: int = 0
    steps: int = 25
    cfg: float = 1.0
    sampler: str = "euler"
    scheduler: str = "simple"
    denoise: float = 1.0
    dit: str = P.RECOMMENDED_16GB["dit"]
    text_encoder: str = P.RECOMMENDED_16GB["text_encoder"]
    text_encoder_device: str = "default"  # "default" | "cpu"
    vae: str = P.RECOMMENDED_16GB["vae"]
    loras: list[LoraEntry] = field(default_factory=list)
    accelerator: str = "none"
    accelerator_lora_name: str | None = None  # filename under models/loras; defaults to preset filename
    kv_cache_device: str = "auto"  # auto | gpu | cpu | off
    kv_cache_dtype: str = "default"  # default | int8 | int4
    references: list[str] = field(default_factory=list)  # uploaded filenames, image_1 first
    ref_resolution: int = 1024  # 0 keeps each reference at its own size
    prompt_enhancer: bool = False
    pe_model: str | None = None
    pe_thinking: bool = False
    pe_system_prompt: str | None = None
    pe_max_length: int = 4096
    controlnet: ControlNetSpec | None = None
    fix_guidance: bool = False
    output_prefix: str = "Qwen_image_2.1"
    use_save_image_advanced: bool = True

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "GenerationSpec":
        data = dict(raw)
        data["loras"] = [LoraEntry(**l) if isinstance(l, dict) else l for l in data.get("loras", [])]
        cn = data.get("controlnet")
        if isinstance(cn, dict):
            data["controlnet"] = ControlNetSpec(**cn)
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    def apply_accelerator_defaults(self) -> None:
        """Copy steps/cfg/sampler/scheduler from the selected accelerator preset."""
        acc = P.ACCELERATORS.get(self.accelerator)
        if acc is None or acc.key == "none":
            return
        self.steps = acc.steps
        self.cfg = acc.cfg
        self.sampler = acc.sampler
        self.scheduler = acc.scheduler

    def apply_sampler_preset(self, key: str) -> None:
        preset = P.SAMPLER_PRESETS[key]
        self.steps, self.cfg = preset.steps, preset.cfg
        self.sampler, self.scheduler = preset.sampler, preset.scheduler
        self.negative_prompt = preset.negative
        self.fix_guidance = preset.fix_guidance


class SpecError(ValueError):
    """Raised when a GenerationSpec cannot be turned into a prompt."""


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class _Graph:
    def __init__(self) -> None:
        self.nodes: Prompt = {}
        self._counter = 0

    def add(self, node_id: str, class_type: str, inputs: dict[str, Any], title: str | None = None) -> str:
        if node_id in self.nodes:
            self._counter += 1
            node_id = f"{node_id}_{self._counter}"
        node: dict[str, Any] = {"class_type": class_type, "inputs": inputs}
        node["_meta"] = {"title": title or class_type}
        self.nodes[node_id] = node
        return node_id


def _check_spec(spec: GenerationSpec) -> None:
    if spec.mode not in ("t2i", "edit"):
        raise SpecError(f"mode must be 't2i' or 'edit', got {spec.mode!r}")
    if spec.mode == "edit" and not spec.references:
        raise SpecError("edit mode needs at least one reference image (image_1 is the image being edited)")
    if len(spec.references) > MAX_REFERENCES:
        raise SpecError(f"at most {MAX_REFERENCES} reference images are wired by TextEncodeQwenImage21")
    if spec.mode == "t2i" and spec.references:
        raise SpecError("text-to-image does not take reference images; switch mode to 'edit'")
    if spec.steps < 1:
        raise SpecError("steps must be >= 1")
    if spec.cfg < 1.0:
        raise SpecError("cfg below 1.0 is not supported by Qwen-Image 2.1 (official guidance: keep cfg >= 1)")
    if spec.width % P.RESOLUTION_STEP or spec.height % P.RESOLUTION_STEP:
        raise SpecError(f"width/height must be multiples of {P.RESOLUTION_STEP} (got {spec.width}x{spec.height})")
    if spec.ref_resolution % 32:
        raise SpecError("ref_resolution must be 0 or a multiple of 32")
    if spec.kv_cache_device not in ("auto", "gpu", "cpu", "off"):
        raise SpecError("kv_cache_device must be auto|gpu|cpu|off")
    if spec.kv_cache_dtype not in ("default", "int8", "int4"):
        raise SpecError("kv_cache_dtype must be default|int8|int4")
    if spec.accelerator not in P.ACCELERATORS:
        raise SpecError(f"unknown accelerator preset {spec.accelerator!r}")
    if spec.prompt_enhancer and not spec.pe_model:
        raise SpecError("prompt_enhancer is on but no pe_model file was selected")
    if spec.controlnet is not None:
        cn = spec.controlnet
        if not cn.patch:
            raise SpecError("controlnet.patch (model_patches file) is required")
        if not (cn.control_image or cn.inpaint_image or cn.mask_image):
            raise SpecError("controlnet needs a control_image and/or inpaint_image + mask_image")
        if cn.mask_image and not cn.inpaint_image:
            raise SpecError("controlnet mask_image needs inpaint_image (the picture to keep outside the mask)")
    for lora in spec.loras:
        if not lora.name:
            raise SpecError("every LoRA entry needs a filename")


def build_prompt(spec: GenerationSpec) -> Prompt:
    """Turn a GenerationSpec into a ComfyUI API prompt."""
    _check_spec(spec)
    g = _Graph()
    acc = P.ACCELERATORS[spec.accelerator]

    # --- loaders -----------------------------------------------------------
    model = [g.add("unet", "UNETLoader", {"unet_name": spec.dit, "weight_dtype": "default"}, "Load Diffusion Model"), 0]
    clip_inputs: dict[str, Any] = {"clip_name": spec.text_encoder, "type": "qwen_image"}
    if spec.text_encoder_device != "default":
        clip_inputs["device"] = spec.text_encoder_device
    clip = [g.add("clip", "CLIPLoader", clip_inputs, "Load Text Encoder (Qwen3-VL 8B)"), 0]
    vae = [g.add("vae", "VAELoader", {"vae_name": spec.vae}, "Load RGBA VAE"), 0]

    # --- model patch chain ---------------------------------------------------
    if acc.key != "none":
        lora_file = spec.accelerator_lora_name or acc.filename
        model = [g.add("accel_lora", "LoraLoaderModelOnly",
                       {"model": model, "lora_name": lora_file, "strength_model": acc.strength},
                       f"Accelerator LoRA: {acc.label}"), 0]
        if acc.model_sampling_flux is not None:
            max_shift, base_shift = acc.model_sampling_flux
            model = [g.add("model_sampling", "ModelSamplingFlux",
                           {"model": model, "max_shift": max_shift, "base_shift": base_shift,
                            "width": spec.width, "height": spec.height}, "ModelSamplingFlux (accelerator shift)"), 0]
    for index, lora in enumerate(spec.loras, start=1):
        model = [g.add(f"lora_{index}", "LoraLoaderModelOnly",
                       {"model": model, "lora_name": lora.name, "strength_model": float(lora.strength)},
                       f"LoRA {index}: {lora.name}"), 0]
    if spec.kv_cache_device != "auto" or spec.kv_cache_dtype != "default":
        model = [g.add("kv_cache", "QwenImage21Cache",
                       {"model": model, "device": spec.kv_cache_device, "dtype": spec.kv_cache_dtype},
                       "Qwen Image 2.1 Cache"), 0]
    if spec.fix_guidance:
        apg = P.FIX_GUIDANCE["apg"]
        fresca = P.FIX_GUIDANCE["fresca"]
        model = [g.add("apg", "APG", {"model": model, **apg}, "APG (adaptive projected guidance)"), 0]
        model = [g.add("fresca", "FreSca", {"model": model, **fresca}, "FreSca"), 0]

    # --- references --------------------------------------------------------
    ref_images: list[Link] = []
    for index, filename in enumerate(spec.references, start=1):
        node_id = g.add(f"ref_{index}", "LoadImage", {"image": filename}, f"Reference image_{index}")
        ref_images.append([node_id, 0])

    # --- ControlNet (Fun union patch) -----------------------------------------
    if spec.controlnet is not None:
        cn = spec.controlnet
        patch = [g.add("cn_patch", "ModelPatchLoader", {"name": cn.patch}, "Load Fun ControlNet Union"), 0]
        cn_inputs: dict[str, Any] = {"model": model, "model_patch": patch, "vae": vae,
                                     "strength": float(cn.strength),
                                     "start_percent": float(cn.start_percent),
                                     "end_percent": float(cn.end_percent)}
        if cn.control_image:
            cn_inputs["image"] = [g.add("cn_image", "LoadImage", {"image": cn.control_image}, "Control map"), 0]
        if cn.inpaint_image:
            cn_inputs["inpaint_image"] = [g.add("cn_inpaint", "LoadImage", {"image": cn.inpaint_image},
                                                "Inpaint source"), 0]
        if cn.mask_image:
            mask_img = [g.add("cn_mask_img", "LoadImage", {"image": cn.mask_image}, "Inpaint mask (white=redo)"), 0]
            cn_inputs["mask"] = [g.add("cn_mask", "ImageToMask", {"image": mask_img, "channel": "red"},
                                       "Mask from red channel"), 0]
        model = [g.add("controlnet", "ZImageFunControlnet", cn_inputs, "Apply Fun ControlNet (Qwen 2.1)"), 0]

    # --- prompt text (optionally rewritten by the prompt enhancer) ------------
    prompt_value: Any = spec.prompt
    if spec.prompt_enhancer:
        pe_clip = [g.add("pe_clip", "CLIPLoader", {"clip_name": spec.pe_model, "type": "qwen_image"},
                         "Load prompt enhancer"), 0]
        tg_inputs: dict[str, Any] = {
            "clip": pe_clip,
            "prompt": spec.prompt,
            "max_length": int(spec.pe_max_length),
            "sampling_mode": "on",
            "sampling_mode.temperature": 1.0,
            "sampling_mode.top_k": 20,
            "sampling_mode.top_p": 0.95,
            "sampling_mode.min_p": 0.0,
            "sampling_mode.repetition_penalty": 1.0,
            "sampling_mode.seed": int(spec.seed),
            "sampling_mode.presence_penalty": 1.5 if spec.mode == "t2i" else 0.0,
            "thinking": bool(spec.pe_thinking),
            "use_default_template": True,
            "mtp": "auto",
        }
        if spec.pe_system_prompt:
            tg_inputs["system_prompt"] = spec.pe_system_prompt
        if ref_images:
            batched = ref_images[0]
            for index, extra in enumerate(ref_images[1:], start=2):
                batched = [g.add(f"pe_batch_{index}", "ImageBatch", {"image1": batched, "image2": extra},
                                 "Batch references for enhancer"), 0]
            tg_inputs["image"] = batched
        prompt_value = [g.add("prompt_enhancer", "TextGenerate", tg_inputs, "Prompt enhancer (TextGenerate)"), 0]

    # --- conditioning ------------------------------------------------------
    enc_inputs: dict[str, Any] = {"clip": clip, "prompt": prompt_value, "negative_prompt": spec.negative_prompt}
    if spec.mode == "edit":
        enc_inputs["vae"] = vae
        enc_inputs["resolution"] = int(spec.ref_resolution)
        for index, link in enumerate(ref_images, start=1):
            enc_inputs[f"images.image_{index}"] = link
    enc = g.add("encode", "TextEncodeQwenImage21", enc_inputs, "Text Encode Qwen Image 2.1")
    positive, negative = [enc, 0], [enc, 1]

    # --- latent -----------------------------------------------------------
    if spec.mode == "edit" and spec.canvas_mode == "match_ref":
        latent = [enc, 2]
    else:
        latent = [g.add("latent", "EmptyLatentImage",
                        {"width": spec.width, "height": spec.height, "batch_size": int(spec.batch_size)},
                        "Empty latent (canvas)"), 0]

    # --- sampler ----------------------------------------------------------
    if acc.mode == "sigmas":
        sampler_sel = [g.add("sampler_select", "KSamplerSelect", {"sampler_name": acc.sampler}, "Sampler"), 0]
        sigmas = [g.add("sigmas", "ManualSigmas", {"sigmas": acc.sigmas_string()}, f"Sigmas: {acc.label}"), 0]
        samples = [g.add("sample", "SamplerCustom",
                         {"model": model, "add_noise": True, "noise_seed": int(spec.seed), "cfg": float(spec.cfg),
                          "positive": positive, "negative": negative, "sampler": sampler_sel, "sigmas": sigmas,
                          "latent_image": latent}, "SamplerCustom (distilled schedule)"), 0]
    else:
        samples = [g.add("sample", "KSampler",
                         {"model": model, "seed": int(spec.seed), "steps": int(spec.steps), "cfg": float(spec.cfg),
                          "sampler_name": spec.sampler, "scheduler": spec.scheduler,
                          "positive": positive, "negative": negative, "latent_image": latent,
                          "denoise": float(spec.denoise)}, "KSampler"), 0]

    # --- decode + save ------------------------------------------------------
    image = [g.add("decode", "VAEDecode", {"samples": samples, "vae": vae}, "VAE Decode (RGBA aware)"), 0]
    if spec.use_save_image_advanced:
        g.add("save", "SaveImageAdvanced",
              {"images": image, "filename_prefix": spec.output_prefix, "format": "png",
               "format.bit_depth": "8-bit", "format.input_color_space": "sRGB"}, "Save PNG (keeps alpha)")
    else:
        g.add("save", "SaveImage", {"images": image, "filename_prefix": spec.output_prefix}, "Save Image")
    return g.nodes


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@dataclass
class Issue:
    level: str  # "error" | "warning"
    node_id: str
    message: str

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return f"[{self.level}] {self.node_id}: {self.message}"


def _is_link(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and isinstance(value[1], int)


def _matches_signature(key: str, sig: dict[str, Any]) -> bool:
    if key in sig.get("required", ()) or key in sig.get("optional", ()):
        return True
    autogrow = sig.get("autogrow")
    if autogrow and key.startswith(autogrow):
        return True
    # DynamicCombo sub-fields like "format.bit_depth" / "sampling_mode.seed"
    head = key.split(".", 1)[0]
    return "." in key and (head in sig.get("required", ()) or head in sig.get("optional", ()))


def validate_prompt(prompt: Prompt, object_info: dict[str, Any] | None = None) -> list[Issue]:
    """Validate link integrity, offline node signatures and (optionally) a live ``/object_info``."""
    issues: list[Issue] = []
    if not isinstance(prompt, dict) or not prompt:
        return [Issue("error", "-", "prompt is empty")]
    outputs_needed: dict[str, int] = {}
    for node_id, node in prompt.items():
        if not isinstance(node, dict) or "class_type" not in node or not isinstance(node.get("inputs"), dict):
            issues.append(Issue("error", node_id, "node needs 'class_type' and an 'inputs' dict"))
            continue
        for key, value in node["inputs"].items():
            if _is_link(value):
                target, slot = value
                if target not in prompt:
                    issues.append(Issue("error", node_id, f"input {key!r} links to missing node {target!r}"))
                elif target == node_id:
                    issues.append(Issue("error", node_id, f"input {key!r} links to itself"))
                else:
                    outputs_needed[target] = max(outputs_needed.get(target, 0), slot)
    # cycle check
    state: dict[str, int] = {}

    def visit(nid: str) -> bool:
        if state.get(nid) == 1:
            return True
        if state.get(nid) == 2:
            return False
        state[nid] = 1
        for value in prompt[nid]["inputs"].values():
            if _is_link(value) and value[0] in prompt and visit(value[0]):
                return True
        state[nid] = 2
        return False

    for nid in prompt:
        if nid in prompt and isinstance(prompt[nid], dict) and isinstance(prompt[nid].get("inputs"), dict):
            if visit(nid):
                issues.append(Issue("error", nid, "cycle detected in graph"))
                break
    # offline signatures
    for node_id, node in prompt.items():
        if not isinstance(node, dict) or "class_type" not in node:
            continue
        sig = P.NODE_SIGNATURES.get(node["class_type"])
        if sig is None:
            issues.append(Issue("warning", node_id, f"{node['class_type']} is not in the offline signature table"))
            continue
        for req in sig.get("required", ()):
            if req not in node["inputs"]:
                issues.append(Issue("error", node_id, f"{node['class_type']} is missing required input {req!r}"))
        for key in node["inputs"]:
            if not _matches_signature(key, sig):
                issues.append(Issue("warning", node_id, f"{node['class_type']} has unexpected input {key!r}"))
    if object_info:
        issues.extend(_validate_against_object_info(prompt, object_info))
    # has an output node?
    if not any(isinstance(n, dict) and n.get("class_type") in ("SaveImage", "SaveImageAdvanced", "PreviewImage")
               for n in prompt.values()):
        issues.append(Issue("error", "-", "prompt has no SaveImage/SaveImageAdvanced output node"))
    return issues


def _validate_against_object_info(prompt: Prompt, object_info: dict[str, Any]) -> list[Issue]:
    issues: list[Issue] = []
    for node_id, node in prompt.items():
        if not isinstance(node, dict) or "class_type" not in node:
            continue
        class_type = node["class_type"]
        info = object_info.get(class_type)
        if info is None:
            hint = ""
            if class_type in ("TextEncodeQwenImage21", "QwenImage21Cache"):
                hint = f" (needs ComfyUI >= {P.MIN_COMFYUI_VERSION})"
            elif class_type in ("ZImageFunControlnet", "ModelPatchLoader"):
                hint = " (Fun ControlNet support landed 2026-09-28; update ComfyUI)"
            issues.append(Issue("error", node_id, f"server has no node {class_type!r}{hint}"))
            continue
        inputs_info = info.get("input", {}) or {}
        required = inputs_info.get("required", {}) or {}
        optional = inputs_info.get("optional", {}) or {}
        for req_name, req_def in required.items():
            if req_name in node["inputs"]:
                continue
            # autogrow / dynamic templates are reported under their group name; accept dotted children
            if any(k.startswith(req_name + ".") for k in node["inputs"]):
                continue
            # inputs with defaults are technically optional for the server
            if isinstance(req_def, (list, tuple)) and len(req_def) > 1 and isinstance(req_def[1], dict) \
                    and "default" in req_def[1]:
                continue
            issues.append(Issue("error", node_id, f"{class_type} requires input {req_name!r}"))
        # combo values (model files) must exist on the server
        for name, value in node["inputs"].items():
            definition = required.get(name) or optional.get(name)
            if definition is None or _is_link(value):
                continue
            if isinstance(definition, (list, tuple)) and definition and isinstance(definition[0], list):
                options = definition[0]
                if options and all(isinstance(o, str) for o in options) and value not in options:
                    if name in ("unet_name", "clip_name", "vae_name", "lora_name", "name", "image"):
                        issues.append(Issue("error", node_id,
                                            f"{class_type}.{name}: {value!r} is not present on the server"))
                    else:
                        issues.append(Issue("warning", node_id,
                                            f"{class_type}.{name}: {value!r} not in server options"))
    return issues


def has_errors(issues: Iterable[Issue]) -> bool:
    return any(issue.level == "error" for issue in issues)


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


def export_prompt(spec: GenerationSpec, path: str | Path) -> Path:
    """Write the API prompt to ``path`` and the spec next to it as ``*.spec.json``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prompt = build_prompt(spec)
    path.write_text(json.dumps(prompt, indent=2, ensure_ascii=False), encoding="utf-8")
    spec_path = path.with_suffix(".spec.json")
    spec_path.write_text(json.dumps(spec.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def describe_prompt(prompt: Prompt) -> str:
    """One line per node, for the UI's "workflow" pane."""
    lines = []
    for node_id, node in prompt.items():
        inputs = ", ".join(
            f"{k}={'<' + v[0] + '>' if _is_link(v) else (repr(v) if not isinstance(v, str) or len(v) < 48 else repr(v[:45] + '...'))}"
            for k, v in node["inputs"].items()
        )
        lines.append(f"{node_id:<16} {node['class_type']:<26} {inputs}")
    return "\n".join(lines)
