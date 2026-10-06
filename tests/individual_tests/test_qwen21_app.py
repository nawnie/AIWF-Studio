"""Contract tests for the Qwen Image 2.1 Studio engine (engines/qwen_image_2_1).

No ComfyUI, no GPU, no PySide6 required: these cover the pure-Python layers the
GUI and CLI sit on (workflow builder/validator, presets, trainer config
generation, dataset scanning, progress parsing, settings, status server).
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import pytest


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "launch.py").is_file():
            return parent
    raise RuntimeError("repo root not found")


ENGINE = _repo_root() / "engines" / "qwen_image_2_1"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from qwen21 import presets as P  # noqa: E402
from qwen21.prompts import load_pe_system_prompt  # noqa: E402
from qwen21.settings import Settings  # noqa: E402
from qwen21.training import aitoolkit, dataset as ds, diffsynth  # noqa: E402
from qwen21.training.runner import parse_progress  # noqa: E402
from qwen21.web_status import StatusState, WebStatusServer  # noqa: E402
from qwen21.workflow_builder import (ControlNetSpec, GenerationSpec, LoraEntry, SpecError, build_prompt,  # noqa: E402
                                     has_errors, validate_prompt)


# ---------------------------------------------------------------- presets

def test_recommended_16gb_files_exist_in_catalog():
    names = {f.filename for f in P.ALL_MODEL_FILES}
    for key in ("dit", "text_encoder", "vae", "controlnet", "pe_t2i", "pe_i2i"):
        assert P.RECOMMENDED_16GB[key] in names
    assert P.RECOMMENDED_16GB["dit"] == "qwen_image_2.1_int8_convrot.safetensors"
    assert P.RECOMMENDED_16GB["text_encoder"] == "qwen3vl_8b_int8_convrot.safetensors"


def test_native_resolutions_are_on_the_32px_grid():
    for table in (P.NATIVE_2K, P.ONE_MP):
        for w, h in table.values():
            assert w % 32 == 0 and h % 32 == 0
    assert P.snap_resolution(1000, 777) == (992, 768)


def test_sigma_accelerators_end_on_zero_and_are_descending():
    for acc in P.ACCELERATORS.values():
        if acc.mode == "sigmas":
            assert acc.sigmas[0] == 1.0 and acc.sigmas[-1] == 0.0
            assert list(acc.sigmas) == sorted(acc.sigmas, reverse=True)
            assert len(acc.sigmas) == acc.steps + 1
    assert P.ACCELERATORS["turbo8"].model_sampling_flux == (0.6935, 0.5)


def test_pe_system_prompts_load_without_attribution_header():
    for kind in ("t2i", "i2i"):
        text = load_pe_system_prompt(kind)
        assert text.startswith("# ") and "#!" not in text[:200] and len(text) > 5000


# ---------------------------------------------------------------- builder

def _types(prompt):
    return [n["class_type"] for n in prompt.values()]


def test_t2i_prompt_mirrors_official_template():
    spec = GenerationSpec(prompt="a fox", seed=3)
    prompt = build_prompt(spec)
    assert not validate_prompt(prompt)
    types = _types(prompt)
    for expected in ("UNETLoader", "CLIPLoader", "VAELoader", "TextEncodeQwenImage21", "EmptyLatentImage",
                     "KSampler", "VAEDecode", "SaveImageAdvanced"):
        assert expected in types
    ks = prompt["sample"]["inputs"]
    assert (ks["steps"], ks["cfg"], ks["sampler_name"], ks["scheduler"], ks["denoise"]) == (25, 1.0, "euler", "simple", 1.0)
    assert prompt["clip"]["inputs"]["type"] == "qwen_image"
    assert prompt["save"]["inputs"]["format"] == "png"
    # no reference wiring, no vae on the encoder, no KV cache node when defaults are kept
    assert "vae" not in prompt["encode"]["inputs"]
    assert "QwenImage21Cache" not in types


def test_edit_prompt_wires_references_in_order_and_uses_reference_latent():
    refs = [f"r{i}.png" for i in range(1, 11)]
    spec = GenerationSpec(mode="edit", prompt="x", references=refs, ref_resolution=1024)
    prompt = build_prompt(spec)
    assert not validate_prompt(prompt)
    enc = prompt["encode"]["inputs"]
    for i in range(1, 11):
        assert enc[f"images.image_{i}"] == [f"ref_{i}", 0]
        assert prompt[f"ref_{i}"]["inputs"]["image"] == f"r{i}.png"
    assert enc["vae"] == ["vae", 0] and enc["resolution"] == 1024
    assert prompt["sample"]["inputs"]["latent_image"] == ["encode", 2]
    assert "EmptyLatentImage" not in _types(prompt)


def test_edit_custom_canvas_uses_empty_latent():
    spec = GenerationSpec(mode="edit", prompt="x", references=["a.png"], canvas_mode="custom", width=1344, height=768)
    prompt = build_prompt(spec)
    assert prompt["sample"]["inputs"]["latent_image"] == ["latent", 0]
    assert prompt["latent"]["inputs"]["width"] == 1344


def test_lora_stack_chains_in_order_and_feeds_sampler():
    spec = GenerationSpec(prompt="x", loras=[LoraEntry("a.safetensors", 0.8), LoraEntry("b.safetensors", 0.5)])
    prompt = build_prompt(spec)
    assert prompt["lora_1"]["inputs"]["model"] == ["unet", 0]
    assert prompt["lora_2"]["inputs"]["model"] == ["lora_1", 0]
    assert prompt["lora_2"]["inputs"]["strength_model"] == 0.5
    assert prompt["sample"]["inputs"]["model"] == ["lora_2", 0]


def test_turbo8_uses_ksampler_with_model_sampling_flux():
    spec = GenerationSpec(prompt="x", accelerator="turbo8")
    spec.apply_accelerator_defaults()
    prompt = build_prompt(spec)
    assert spec.steps == 8 and spec.cfg == 1.0
    assert prompt["accel_lora"]["inputs"]["lora_name"] == "turbo8_lora_step2500.safetensors"
    ms = prompt["model_sampling"]["inputs"]
    assert (ms["max_shift"], ms["base_shift"]) == (0.6935, 0.5)
    assert "SamplerCustom" not in _types(prompt)


def test_sigma_accelerator_uses_manual_sigmas_and_sampler_custom():
    spec = GenerationSpec(prompt="x", accelerator="fun_acc4")
    spec.apply_accelerator_defaults()
    prompt = build_prompt(spec)
    assert not has_errors(validate_prompt(prompt))
    assert prompt["sigmas"]["inputs"]["sigmas"] == "1, 0.9169867, 0.7861579, 0.549491, 0"
    sc = prompt["sample"]["inputs"]
    assert sc["sampler"] == ["sampler_select", 0] and sc["sigmas"] == ["sigmas", 0] and sc["cfg"] == 1.0
    assert "KSampler" not in _types(prompt)


def test_kv_cache_fix_guidance_and_controlnet_nodes():
    spec = GenerationSpec(mode="edit", prompt="x", references=["a.png"], kv_cache_device="cpu", kv_cache_dtype="int8",
                          fix_guidance=True,
                          controlnet=ControlNetSpec(patch="cn.safetensors", inpaint_image="a.png", mask_image="m.png",
                                                    control_image="d.png", strength=0.7, end_percent=0.6))
    prompt = build_prompt(spec)
    assert not has_errors(validate_prompt(prompt))
    assert prompt["kv_cache"]["inputs"] == {"model": ["unet", 0], "device": "cpu", "dtype": "int8"}
    assert prompt["apg"]["inputs"]["norm_threshold"] == 10.0 and prompt["fresca"]["inputs"]["freq_cutoff"] == 8
    cn = prompt["controlnet"]["inputs"]
    assert cn["model_patch"] == ["cn_patch", 0] and cn["vae"] == ["vae", 0]
    assert cn["mask"] == ["cn_mask", 0] and prompt["cn_mask"]["inputs"]["channel"] == "red"
    assert cn["image"] == ["cn_image", 0] and cn["strength"] == 0.7 and cn["end_percent"] == 0.6
    assert prompt["sample"]["inputs"]["model"] == ["controlnet", 0]


def test_prompt_enhancer_chain_batches_references_and_feeds_prompt():
    spec = GenerationSpec(mode="edit", prompt="make it blue", references=["a.png", "b.png", "c.png"],
                          prompt_enhancer=True, pe_model="pe.safetensors", pe_system_prompt="SYS")
    prompt = build_prompt(spec)
    assert not has_errors(validate_prompt(prompt))
    tg = prompt["prompt_enhancer"]["inputs"]
    assert tg["clip"] == ["pe_clip", 0] and tg["system_prompt"] == "SYS" and tg["sampling_mode"] == "on"
    assert tg["image"] == ["pe_batch_3", 0]
    assert prompt["pe_batch_3"]["inputs"]["image1"] == ["pe_batch_2", 0]
    assert prompt["encode"]["inputs"]["prompt"] == ["prompt_enhancer", 0]


def test_spec_rejections():
    with pytest.raises(SpecError):
        build_prompt(GenerationSpec(mode="edit", prompt="x"))  # no references
    with pytest.raises(SpecError):
        build_prompt(GenerationSpec(prompt="x", references=["a.png"]))  # refs in t2i
    with pytest.raises(SpecError):
        build_prompt(GenerationSpec(prompt="x", cfg=0.5))
    with pytest.raises(SpecError):
        build_prompt(GenerationSpec(prompt="x", width=1000))
    with pytest.raises(SpecError):
        build_prompt(GenerationSpec(mode="edit", prompt="x", references=[f"{i}.png" for i in range(17)]))
    with pytest.raises(SpecError):
        build_prompt(GenerationSpec(prompt="x", prompt_enhancer=True))


def test_validator_flags_broken_links_and_live_object_info():
    prompt = build_prompt(GenerationSpec(prompt="x"))
    prompt["sample"]["inputs"]["model"] = ["nope", 0]
    issues = validate_prompt(prompt)
    assert any("missing node" in i.message for i in issues)
    good = build_prompt(GenerationSpec(prompt="x"))
    object_info = {ct: {"input": {"required": {}}} for ct in set(_types(good)) - {"TextEncodeQwenImage21"}}
    object_info["UNETLoader"] = {"input": {"required": {"unet_name": [["other.safetensors"]], "weight_dtype": [["default"]]}}}
    issues = validate_prompt(good, object_info)
    messages = " ".join(i.message for i in issues if i.level == "error")
    assert "TextEncodeQwenImage21" in messages and "0.37.0" in messages
    assert "not present on the server" in messages


def test_spec_roundtrip():
    spec = GenerationSpec(mode="edit", prompt="p", references=["a.png"], loras=[LoraEntry("l", 0.3)],
                          controlnet=ControlNetSpec(patch="c", control_image="i"))
    again = GenerationSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert again == spec


def test_reference_workflows_are_valid_and_regenerable():
    workflows = sorted((ENGINE / "workflows").glob("*.json"))
    workflows = [w for w in workflows if not w.name.endswith(".spec.json")]
    assert len(workflows) >= 10
    for path in workflows:
        prompt = json.loads(path.read_text(encoding="utf-8"))
        issues = validate_prompt(prompt)
        assert not has_errors(issues), (path.name, [str(i) for i in issues])
        spec = GenerationSpec.from_dict(json.loads(path.with_suffix(".spec.json").read_text(encoding="utf-8")))
        assert build_prompt(spec) == prompt, path.name


# ---------------------------------------------------------------- training

def _make_dataset(tmp_path: Path, n: int = 3, captions: bool = True, control: bool = False):
    folder = tmp_path / "ds"
    folder.mkdir()
    ctrl = tmp_path / "refs"
    if control:
        ctrl.mkdir()
    for i in range(n):
        (folder / f"img_{i}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        if captions:
            (folder / f"img_{i}.txt").write_text(f"caption {i}", encoding="utf-8")
        if control:
            (ctrl / f"img_{i}.jpg").write_bytes(b"\xff\xd8\xff")
    return folder, (ctrl if control else None)


def test_dataset_scan_reports_captions_and_control_mismatch(tmp_path):
    folder, ctrl = _make_dataset(tmp_path, control=True)
    (folder / "img_2.txt").unlink()
    (ctrl / "img_1.jpg").unlink()
    report = ds.scan_dataset(folder, [ctrl], check_alpha=False)
    assert len(report.images) == 3 and [p.name for p in report.captions_missing] == ["img_2.png"]
    assert not report.ok and "img_1" in report.errors[0]
    assert ds.ensure_captions(folder, "a photo", trigger="ohwx") == 1
    assert (folder / "img_2.txt").read_text(encoding="utf-8") == "ohwx a photo"
    assert ds.prepend_trigger(folder, "ohwx") == 2  # img_2 already starts with the trigger


def test_aitoolkit_config_16gb_shape(tmp_path):
    folder, ctrl = _make_dataset(tmp_path, control=True)
    spec = aitoolkit.TrainSpec(name="t", dataset_dir=str(folder), control_dirs=[str(ctrl)], rgba=True)
    aitoolkit.apply_preset(spec, aitoolkit.PRESET_16GB)
    assert not aitoolkit.validate_spec(spec)
    cfg = aitoolkit.build_config(spec)
    proc = cfg["config"]["process"][0]
    assert cfg["job"] == "extension" and proc["type"] == "diffusion_trainer"
    model = proc["model"]
    assert model["arch"] == "qwen_image_2" and model["qtype"] == "convrot8" and model["qtype_te"] == "convrot8"
    assert model["low_vram"] is True and model["layer_offloading"] is True
    assert model["model_kwargs"]["rgba"] is True
    assert proc["datasets"][0]["control_path"] == str(ctrl)
    assert proc["train"]["cache_text_embeddings"] is True and proc["train"]["noise_scheduler"] == "flowmatch"
    assert proc["network"] == {"type": "lora", "linear": 16, "linear_alpha": 16}
    assert proc["train"]["disable_sampling"] is True
    path = aitoolkit.write_config(spec, tmp_path / "cfg.yaml")
    import yaml

    assert yaml.safe_load(path.read_text(encoding="utf-8")) == cfg
    cmd = aitoolkit.build_command("python.exe", tmp_path / "ai-toolkit", path)
    assert cmd[1].endswith("run.py") and cmd[2] == str(path)


def test_aitoolkit_trigger_with_cached_embeddings_is_rejected(tmp_path):
    folder, _ = _make_dataset(tmp_path)
    spec = aitoolkit.TrainSpec(name="t", dataset_dir=str(folder), trigger_word="ohwx")
    assert any("trigger_word" in p for p in aitoolkit.validate_spec(spec))


def test_diffsynth_metadata_and_command(tmp_path):
    folder, ctrl = _make_dataset(tmp_path, control=True)
    meta = ds.write_diffsynth_metadata(folder, [ctrl])
    rows = json.loads(meta.read_text(encoding="utf-8"))
    assert rows[0]["image"] == "img_0.png" and rows[0]["prompt"] == "caption 0" and "img_0.jpg" in rows[0]["edit_image"]
    csv_meta = ds.write_diffsynth_metadata(folder)
    assert csv_meta.name == "metadata.csv" and "image,prompt" in csv_meta.read_text(encoding="utf-8")
    spec = diffsynth.DiffSynthSpec(name="t", dataset_dir=str(folder), metadata_path=str(meta), edit_mode=True)
    cmd = diffsynth.build_command(spec, "python", tmp_path / "DiffSynth-Studio")
    joined = " ".join(cmd)
    assert "model_training/train.py" in joined.replace("\\", "/")
    assert "--extra_inputs edit_image" in joined and "--lora_base_model dit" in joined
    assert "--fp8_models Qwen/Qwen-Image-2.1:text_encoder/model*.safetensors" in joined
    assert "--use_gradient_checkpointing_offload" in joined


def test_progress_parser_reads_tqdm_lines():
    prog = parse_progress("my_lora:  12%|█▏        | 240/2000 [10:03<1:13:00,  2.49s/it, lr: 1.0e-04 loss: 4.123e-01]")
    assert prog is not None and (prog.step, prog.total) == (240, 2000) and abs(prog.loss - 0.4123) < 1e-6
    assert parse_progress("Loading transformer") is None
    nxt = parse_progress("Epoch 2 | loss=0.25", prog)
    assert nxt.epoch == 2 and nxt.step == 240


# ---------------------------------------------------------------- settings / status

def test_settings_roundtrip_and_utf8_env(tmp_path):
    s = Settings(comfy_url="http://127.0.0.1:8188", comfy_launch_command=r"F:\ComfyUI\venv\Scripts\python.exe main.py")
    s.save(tmp_path / "s.json")
    again = Settings.load(tmp_path / "s.json")
    assert again == s
    env = s.comfy_launch_env()
    assert env["PYTHONUTF8"] == "1" and env["PYTHONIOENCODING"] == "utf-8"
    assert Settings.load(tmp_path / "missing.json") == Settings()


def test_web_status_health_endpoint(tmp_path):
    (tmp_path / "out.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    server = WebStatusServer(StatusState(tmp_path, lambda: {"comfy_connected": False}), port=0)
    assert server.start()
    try:
        port = server._server.server_address[1]
        health = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=5).read())
        assert health["status"] == "ok" and health["comfy_connected"] is False
        page = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5).read().decode("utf-8")
        assert "out.png" in page
        assert urllib.request.urlopen(f"http://127.0.0.1:{port}/outputs/out.png", timeout=5).read().startswith(b"\x89PNG")
    finally:
        server.stop()
