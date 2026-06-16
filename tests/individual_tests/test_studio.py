from PIL import Image

from aiwf.web.studio.helpers import paste_control_values, segment_source_image


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
