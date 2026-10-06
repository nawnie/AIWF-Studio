"""Regenerate the reference ComfyUI API workflows under engines/qwen_image_2_1/workflows/.

    python scripts/export_qwen21_workflows.py

Each JSON is a ready-to-POST ComfyUI API prompt (also loadable by drag-drop in the
ComfyUI frontend). A *.spec.json with the GenerationSpec sits next to each file.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "engines" / "qwen_image_2_1"
sys.path.insert(0, str(ENGINE))

from qwen21 import presets as P  # noqa: E402
from qwen21.workflow_builder import ControlNetSpec, GenerationSpec, LoraEntry, export_prompt, validate_prompt  # noqa: E402

OUT = ENGINE / "workflows"

PROMPT_T2I = ("A small corner bookstore at dusk, warm window light, a chalkboard sign reading \"New Arrivals\" in neat "
              "hand lettering, wet cobblestones reflecting the glow, 35mm photo, shallow depth of field")


def specs() -> dict[str, GenerationSpec]:
    turbo = GenerationSpec(prompt=PROMPT_T2I, seed=42, accelerator="turbo8")
    turbo.apply_accelerator_defaults()
    pruna = GenerationSpec(prompt=PROMPT_T2I, seed=42, accelerator="pruna8")
    pruna.apply_accelerator_defaults()
    return {
        "qwen21_t2i_int8_25step": GenerationSpec(prompt=PROMPT_T2I, seed=42),
        "qwen21_t2i_2k_16x9": GenerationSpec(prompt=PROMPT_T2I, seed=42, width=2752, height=1536),
        "qwen21_t2i_rgba_transparent": GenerationSpec(
            prompt="A single red maple leaf with fine veins, studio lighting. " + P.RGBA_PROMPT_HINT, seed=7),
        "qwen21_t2i_turbo8_8step": turbo,
        "qwen21_t2i_pruna8_sigmas": pruna,
        "qwen21_t2i_with_loras": GenerationSpec(
            prompt="ohwx_subject portrait, soft window light, 85mm", seed=42,
            loras=[LoraEntry("my_qwen21_character_lora.safetensors", 0.9),
                   LoraEntry("qwen-image-2.1-fix-1.0-comfy.safetensors", 1.0)]),
        "qwen21_t2i_prompt_enhancer": GenerationSpec(
            prompt="bookstore at dusk with a New Arrivals sign", seed=42, prompt_enhancer=True,
            pe_model=P.RECOMMENDED_16GB["pe_t2i"], pe_system_prompt="(see engines/qwen_image_2_1/prompts/pe_t2i_system.txt)"),
        "qwen21_edit_single_ref": GenerationSpec(
            mode="edit", prompt="Change the jacket in <image1> to a light blue denim jacket, keep everything else",
            references=["qwen21/portrait.png"], seed=42),
        "qwen21_edit_multiref_outfit_swap": GenerationSpec(
            mode="edit",
            prompt="Keep the character and pose in <image1> unchanged, put the light blue denim shirt from <image2> on "
                   "the character, keep the lighting and background of <image1>",
            references=["qwen21/portrait.png", "qwen21/denim_shirt.png"], seed=42),
        "qwen21_edit_10_references": GenerationSpec(
            mode="edit",
            prompt="Compose a group photo: the person from <image1> in the center, the people from <image2> to <image6> "
                   "around them, wearing the outfits from <image7> and <image8>, in the location from <image9>, "
                   "lit like <image10>",
            references=[f"qwen21/ref_{i:02d}.png" for i in range(1, 11)], seed=42, kv_cache_device="cpu",
            kv_cache_dtype="int8", text_encoder=P.TEXT_ENCODERS[1].filename),
        "qwen21_remove_background_rgba": GenerationSpec(
            mode="edit", prompt="Remove the background, and output a PNG image", references=["qwen21/product.png"],
            seed=42),
        "qwen21_edit_marked_region": GenerationSpec(
            mode="edit", prompt="Change only the area marked in red in <image1>: replace the mug with a glass of water",
            references=["qwen21/desk_marked.png"], seed=42),
        "qwen21_edit_turbo8": GenerationSpec(
            mode="edit", prompt="Remove the background, and output a PNG image", references=["qwen21/product.png"],
            seed=42, accelerator="turbo8", steps=8, cfg=1.0),
        "qwen21_controlnet_depth_t2i": GenerationSpec(
            prompt=PROMPT_T2I, seed=42,
            controlnet=ControlNetSpec(patch=P.RECOMMENDED_16GB["controlnet"], control_image="qwen21/depth_map.png",
                                      strength=0.8, end_percent=0.8)),
        "qwen21_controlnet_inpaint_edit": GenerationSpec(
            mode="edit", prompt="Replace the marked region with a bouquet of white tulips", seed=42,
            references=["qwen21/desk_marked.png"],
            controlnet=ControlNetSpec(patch=P.RECOMMENDED_16GB["controlnet"], inpaint_image="qwen21/desk.png",
                                      mask_image="qwen21/desk_mask.png")),
        "qwen21_t2i_fix_lora_recipe": _fix_recipe(),
    }


def _fix_recipe() -> GenerationSpec:
    spec = GenerationSpec(prompt=PROMPT_T2I, seed=42, width=1184, height=1600,
                          loras=[LoraEntry("qwen-image-2.1-fix-1.0-comfy.safetensors", 1.0)])
    spec.apply_sampler_preset("fix_lora")
    return spec


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    bad = 0
    for name, spec in specs().items():
        path = export_prompt(spec, OUT / f"{name}.json")
        import json

        issues = validate_prompt(json.loads(path.read_text(encoding="utf-8")))
        errors = [i for i in issues if i.level == "error"]
        bad += bool(errors)
        print(f"{name:<40} {'OK ' if not errors else 'ERR'} {len(issues)} issue(s)")
        for issue in errors:
            print("   ", issue)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
