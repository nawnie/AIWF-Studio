from pathlib import Path

from PIL import Image
import gradio as gr
import pytest

from aiwf.web.studio import tab as studio_tab
from aiwf.web.studio.catalogs import StudioCatalogs
from aiwf.web.studio.helpers import paste_control_values, segment_source_image
from aiwf.web.studio.request_builder import build_generation_request
from aiwf.web.studio.session import StudioSession


def test_paste_control_values_syncs_img2img_and_inpaint_denoise():
    values = paste_control_values(
        {
            "prompt": "portrait",
            "negative_prompt": "blur",
            "sampler": "euler_a",
            "denoising_strength": 0.42,
            "mask_blur": 12,
        },
        sampler_id_to_label={"euler_a": "Euler a"},
        default_sampler_label="Euler a",
    )

    assert values["img2img_denoise"] == 0.42
    assert values["inpaint_denoise"] == 0.42
    assert values["mask_blur"] == 12
    assert values["sampler"] == "Euler a"


def test_segment_source_prefers_inpaint_editor_background():
    workspace = Image.new("RGB", (4, 4), "red")
    editor_background = Image.new("RGB", (8, 8), "blue")

    assert segment_source_image(workspace, {"background": editor_background}) is editor_background


def test_segment_source_falls_back_to_workspace_image():
    workspace = Image.new("RGB", (4, 4), "red")

    assert segment_source_image(workspace, None) is workspace


def _catalogs():
    return StudioCatalogs(
        sampler_map={
            "Euler a": "euler_a",
            "DPM++ 2M": "dpmpp_2m",
            "DPM++ 2M Karras": "dpmpp_2m_karras",
        },
        sampler_id_to_label={
            "euler_a": "Euler a",
            "dpmpp_2m": "DPM++ 2M",
            "dpmpp_2m_karras": "DPM++ 2M Karras",
        },
        default_sampler_label="Euler a",
        schedule_map={"Automatic": "automatic", "Beta": "beta"},
        schedule_id_to_label={"automatic": "Automatic", "beta": "Beta"},
        default_schedule_label="Automatic",
    )


def _request_kwargs(**overrides):
    values = {
        "catalogs": _catalogs(),
        "session": StudioSession(),
        "mode_label": "Text",
        "editing_mask": False,
        "prompt_text": "cat",
        "negative_text": "",
        "ckpt_title": "Model A",
        "sampler_label": "Euler a",
        "scheduler_label": "Automatic",
        "step_count": 4,
        "cfg_scale": 7.0,
        "clip_skip_value": 1,
        "w": 512,
        "h": 512,
        "bs": 1,
        "bc": 1,
        "seed_value": -1,
        "vae_id": None,
        "hires_enabled": False,
        "hires_scale": 2.0,
        "hires_steps": 10,
        "hires_denoise": 0.35,
        "hires_upscaler": "lanczos",
        "img2img_denoise": 0.75,
        "inpaint_denoise_value": 0.75,
        "mask_blur_value": 4,
        "seam_erode_value": 0,
        "inpaint_area_value": "Only masked",
        "inpaint_padding_value": 32,
        "masked_content_value": "latent noise",
        "source_image": None,
        "editor_value": None,
        "ckpt_map": {"Model A": "model-a"},
        "tags_text": "",
        "use_file": False,
        "prompt_file_path": None,
        "dynamic_seed": None,
        "style_name": None,
        "style_template_prompt": None,
        "style_template_negative": None,
        "cn_enable": False,
        "cn_model_id": None,
        "cn_module": None,
        "cn_image": None,
        "cn_weight": 1.0,
        "cn_guidance_start": 0.0,
        "cn_guidance_end": 1.0,
        "cn_threshold_a": 100.0,
        "cn_threshold_b": 200.0,
        "inpaint_source_choice": "original",
    }
    values.update(overrides)
    return values


def test_request_builder_normalizes_incompatible_sampler_schedule_pair():
    request, *_ = build_generation_request(
        **_request_kwargs(sampler_label="DPM++ 2M Karras", scheduler_label="Beta")
    )

    assert request.sampler == "dpmpp_2m_karras"
    assert request.scheduler == "automatic"


def test_request_builder_rejects_stale_checkpoint_selection():
    with pytest.raises(gr.Error, match="Selected checkpoint"):
        build_generation_request(**_request_kwargs(ckpt_title="Missing"))


def test_main_studio_stop_bypasses_gradio_queue():
    source = Path(studio_tab.__file__).read_text(encoding="utf-8")
    stop_registration = source.split("interrupt.click(", 1)[1].split(")", 1)[0]

    assert "queue=False" in stop_registration
    assert "show_progress=False" in stop_registration
