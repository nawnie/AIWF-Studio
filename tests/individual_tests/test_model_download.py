from __future__ import annotations

import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

import aiwf.services.model_download as model_download_module
from aiwf.core.config.settings import RuntimeFlags
from aiwf.core.domain.model_download import CatalogEntry
from aiwf.services.model_download import (
    ModelDownloadService,
    ParsedRemote,
    _parse_civitai_reference,
    _parse_hf_reference,
    detect_source,
    inspect_custom_input,
    split_hf_url,
)
from aiwf.services.model_download_catalog import QUICK_START_BUNDLES
from scripts.ensure_default_sd15 import DEFAULT_SD15_CATALOG_KEY


def _write_component_safetensors(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = json.dumps({"weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0")


def _write_large_safetensors(path: Path, size_bytes: int) -> None:
    """Create a structurally valid sparse fixture of the requested logical size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload_size = max(0, size_bytes - 128)
    while True:
        header = json.dumps({
            "weight": {"dtype": "U8", "shape": [payload_size], "data_offsets": [0, payload_size]}
        }, separators=(",", ":")).encode("utf-8")
        total_size = 8 + len(header) + payload_size
        if total_size >= size_bytes:
            break
        payload_size += size_bytes - total_size
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(header)))
        stream.write(header)
        stream.truncate(total_size)


def _write_minimal_gguf(path: Path, *, tensor_elements: int = 1, pad_to_bytes: int = 0) -> None:
    gguf = pytest.importorskip("gguf")
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(path), "llama")
    try:
        writer.add_tensor("weight", np.zeros(tensor_elements, dtype=np.float32))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()
    if path.stat().st_size < pad_to_bytes:
        with path.open("ab") as stream:
            stream.truncate(pad_to_bytes)


def _write_support_components(path: Path, *, family: str) -> None:
    if family == "z_image":
        declarations = {
            "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
            "text_encoder": ["transformers", "Qwen3Model"],
            "tokenizer": ["transformers", "Qwen2Tokenizer"],
            "transformer": ["diffusers", "ZImageTransformer2DModel"],
            "vae": ["diffusers", "AutoencoderKL"],
        }
        class_name = "ZImagePipeline"
        configs = {
            "scheduler/scheduler_config.json": {"_class_name": "FlowMatchEulerDiscreteScheduler", "num_train_timesteps": 1000},
            "text_encoder/config.json": {"model_type": "qwen3", "hidden_size": 8, "num_hidden_layers": 1, "vocab_size": 16},
            "vae/config.json": {"_class_name": "AutoencoderKL", "in_channels": 3, "out_channels": 3, "latent_channels": 4},
            "tokenizer/tokenizer_config.json": {"tokenizer_class": "Qwen2Tokenizer"},
            "tokenizer/tokenizer.json": {"model": {"type": "BPE", "vocab": {"x": 0}}},
        }
        weight_paths = ("text_encoder/model.safetensors", "vae/diffusion_pytorch_model.safetensors")
    else:
        declarations = {
            "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
            "text_encoder": ["transformers", "Qwen3ForCausalLM"],
            "tokenizer": ["transformers", "Qwen2TokenizerFast"],
            "transformer": ["diffusers", "Flux2Transformer2DModel"],
            "vae": ["diffusers", "AutoencoderKLFlux2"],
        }
        class_name = "Flux2KleinPipeline"
        configs = {
            "scheduler/scheduler_config.json": {"_class_name": "FlowMatchEulerDiscreteScheduler", "num_train_timesteps": 1000},
            "text_encoder/config.json": {"model_type": "qwen3", "hidden_size": 8, "num_hidden_layers": 1, "vocab_size": 16},
            "vae/config.json": {"_class_name": "AutoencoderKLFlux2", "in_channels": 3, "out_channels": 3, "latent_channels": 32},
            "tokenizer/tokenizer_config.json": {"tokenizer_class": "Qwen2TokenizerFast"},
            "tokenizer/tokenizer.json": {"model": {"type": "BPE", "vocab": {"x": 0}}},
        }
        weight_paths = ("text_encoder/model.safetensors", "vae/diffusion_pytorch_model.safetensors")

    path.mkdir(parents=True, exist_ok=True)
    (path / "model_index.json").write_text(
        json.dumps({"_class_name": class_name, **declarations}), encoding="utf-8"
    )
    for relative, payload in configs.items():
        destination = path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload), encoding="utf-8")
    for relative in weight_paths:
        _write_component_safetensors(path / relative)


def test_detect_source():
    assert detect_source("https://huggingface.co/runwayml/stable-diffusion-v1-5") == "huggingface"
    assert detect_source("https://civitai.com/models/4384") == "civitai"
    assert detect_source("https://example.com/file.safetensors") == "direct"


def test_parse_hf_repo_and_filename():
    remote = _parse_hf_reference("runwayml/stable-diffusion-v1-5", "v1-5-pruned-emaonly.safetensors")
    assert remote.repo_id == "runwayml/stable-diffusion-v1-5"
    assert remote.filename == "v1-5-pruned-emaonly.safetensors"
    assert "resolve/main" in remote.url


def test_parse_hf_resolve_url():
    url = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/diffusion_pytorch_model.safetensors"
    remote = _parse_hf_reference(url)
    assert remote.filename == "diffusion_pytorch_model.safetensors"
    assert remote.repo_filename == "diffusion_pytorch_model.safetensors"
    assert remote.repo_id == "stabilityai/sd-vae-ft-mse-original"


def test_parse_hf_subpath_preserves_remote_repo_filename():
    remote = _parse_hf_reference(
        "Comfy-Org/Lumina_Image_2.0_Repackaged",
        "split_files/vae/ae.safetensors",
    )
    assert remote.filename == "ae.safetensors"
    assert remote.repo_filename == "split_files/vae/ae.safetensors"
    assert remote.url.endswith("/resolve/main/split_files/vae/ae.safetensors")


def test_parse_hf_repo_requires_filename():
    with pytest.raises(ValueError):
        _parse_hf_reference("runwayml/stable-diffusion-v1-5", "")


def test_split_hf_tree_url():
    url = "https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main/v1-5-pruned-emaonly.safetensors"
    repo, filename = split_hf_url(url)
    assert repo == "runwayml/stable-diffusion-v1-5"
    assert filename == "v1-5-pruned-emaonly.safetensors"


def test_split_hf_url_keeps_nested_file_path():
    url = "https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged/blob/main/split_files/vae/ae.safetensors"
    repo, filename = split_hf_url(url)
    assert repo == "Comfy-Org/Lumina_Image_2.0_Repackaged"
    assert filename == "split_files/vae/ae.safetensors"


def test_sana_video_download_setup_rejects_generic_diffusers_snapshot(tmp_path: Path):
    target = tmp_path / "models" / "sana-video" / "Diffusers" / "incomplete"
    target.mkdir(parents=True)
    (target / "model_index.json").write_text(
        '{"_class_name":"WrongPipeline","text_encoder":["x","WrongTextEncoder"],'
        '"transformer":["x","WrongTransformer"],"vae":["x","WrongVAE"]}',
        encoding="utf-8",
    )
    for component in ("text_encoder", "transformer", "vae"):
        component_dir = target / component
        component_dir.mkdir()
        (component_dir / "config.json").write_text("{}", encoding="utf-8")
        (component_dir / "weights.safetensors").write_bytes(b"weights")

    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path))

    assert not service._snapshot_target_ready("sana_video_diffusers", target)


def test_split_hf_browse_page_rejected():
    with pytest.raises(ValueError, match="browse page"):
        split_hf_url("https://huggingface.co/models?pipeline_tag=text-to-image")


def test_inspect_custom_input_normalizes_hf_page():
    _, repo, filename, status = inspect_custom_input(
        source="huggingface",
        url_or_repo="https://huggingface.co/runwayml/stable-diffusion-v1-5",
        filename="",
    )
    assert repo == "runwayml/stable-diffusion-v1-5"
    assert filename == ""
    assert "filename" in status.lower()


def test_destination_dirs(tmp_path: Path):
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models")
    service = ModelDownloadService(flags)
    assert service.destination_dir("checkpoint").name == "Stable-diffusion"
    assert service.destination_dir("lora").name == "Loras"
    assert service.destination_dir("controlnet").name == "ControlNet"
    assert service.destination_dir("preprocessor") == tmp_path / "models" / "ControlNet" / "Annotators"
    assert service.destination_dir("upscaler").name == "RealESRGAN"
    assert service.destination_dir("esrgan").name == "ESRGAN"
    assert service.destination_dir("gfpgan").name == "GFPGAN"
    assert service.destination_dir("codeformer").name == "Codeformer"
    assert service.destination_dir("wan_safetensor") == tmp_path / "models" / "wan" / "Safetensor"
    assert service.destination_dir("wan_gguf") == tmp_path / "models" / "wan" / "GGUF"
    assert service.destination_dir("wan_diffusers") == tmp_path / "models" / "wan" / "Diffusers"
    assert service.destination_dir("wan_lora") == tmp_path / "models" / "wan" / "lora"
    assert service.destination_dir("flux_unet_safetensor") == tmp_path / "models" / "flux" / "UNet"
    assert service.destination_dir("flux_unet_gguf") == tmp_path / "models" / "flux" / "GGUF"
    assert service.destination_dir("flux_text_encoder") == tmp_path / "models" / "flux" / "Textencoder"
    assert service.destination_dir("flux_vae") == tmp_path / "models" / "flux" / "VAE"
    assert service.destination_dir("flux2_unet_safetensor") == tmp_path / "models" / "flux2" / "UNet"
    assert service.destination_dir("flux2_unet_gguf") == tmp_path / "models" / "flux2" / "GGUF"
    assert service.destination_dir("flux2_components") == tmp_path / "models" / "flux2" / "Components"
    assert service.destination_dir("flux2_diffusers") == tmp_path / "models" / "flux2" / "Diffusers"
    assert service.destination_dir("z_image_unet_safetensor") == tmp_path / "models" / "z-image" / "UNet"
    assert service.destination_dir("z_image_unet_gguf") == tmp_path / "models" / "z-image" / "GGUF"
    assert service.destination_dir("z_image_components") == tmp_path / "models" / "z-image" / "Components"
    assert service.destination_dir("krea2_unet_safetensor") == tmp_path / "models" / "krea2" / "UNet"
    assert service.destination_dir("krea2_text_encoder") == tmp_path / "models" / "krea2" / "Textencoder"
    assert service.destination_dir("krea2_vae") == tmp_path / "models" / "krea2" / "VAE"
    assert service.destination_dir("krea2_diffusers") == tmp_path / "models" / "krea2" / "Diffusers"
    assert service.destination_dir("anima_unet_safetensor") == tmp_path / "models" / "anima" / "UNet"
    assert service.destination_dir("anima_text_encoder") == tmp_path / "models" / "anima" / "Textencoder"
    assert service.destination_dir("anima_vae") == tmp_path / "models" / "anima" / "VAE"
    assert service.destination_dir("qwen_image_diffusers") == tmp_path / "models" / "qwen-image" / "Diffusers"
    assert service.destination_dir("qwen_image_nunchaku") == tmp_path / "models" / "qwen-image" / "Nunchaku"
    assert service.destination_dir("sana_diffusers") == tmp_path / "models" / "sana" / "Diffusers"
    assert service.destination_dir("sana_video_diffusers") == tmp_path / "models" / "sana-video" / "Diffusers"
    assert service.destination_dir("ltx_checkpoint") == tmp_path / "models" / "ltx" / "checkpoints"
    assert service.destination_dir("ltx_gguf") == tmp_path / "models" / "ltx" / "GGUF"
    assert service.destination_dir("ltx_upscaler") == tmp_path / "models" / "ltx" / "upscalers"
    assert service.destination_dir("ltx_lora") == tmp_path / "models" / "ltx" / "loras"
    assert service.destination_dir("ltx_vae") == tmp_path / "models" / "ltx" / "vae"
    assert service.destination_dir("ltx_audio_vae") == tmp_path / "models" / "ltx" / "audio_vae"
    assert service.destination_dir("ltx_text_encoder") == tmp_path / "models" / "ltx" / "text_encoder"
    assert service.destination_dir("llm_gguf") == tmp_path / "models" / "LLM" / "GGUF"
    assert service.destination_dir("llm_safetensor") == tmp_path / "models" / "LLM"


def test_ensure_dirs_creates_nested_category_folders(tmp_path: Path):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))

    service.ensure_dirs()

    assert (tmp_path / "models" / "wan" / "GGUF").is_dir()
    assert (tmp_path / "models" / "wan" / "Diffusers").is_dir()
    assert (tmp_path / "models" / "flux" / "UNet").is_dir()
    assert (tmp_path / "models" / "flux" / "GGUF").is_dir()
    assert (tmp_path / "models" / "flux" / "Textencoder").is_dir()
    assert (tmp_path / "models" / "flux" / "VAE").is_dir()
    assert (tmp_path / "models" / "flux2" / "GGUF").is_dir()
    assert (tmp_path / "models" / "flux2" / "Components").is_dir()
    assert (tmp_path / "models" / "z-image" / "GGUF").is_dir()
    assert (tmp_path / "models" / "z-image" / "Components").is_dir()
    assert (tmp_path / "models" / "krea2" / "UNet").is_dir()
    assert (tmp_path / "models" / "krea2" / "Textencoder").is_dir()
    assert (tmp_path / "models" / "anima" / "UNet").is_dir()
    assert (tmp_path / "models" / "anima" / "Textencoder").is_dir()
    assert (tmp_path / "models" / "ltx" / "checkpoints").is_dir()
    assert (tmp_path / "models" / "ltx" / "GGUF").is_dir()
    assert (tmp_path / "models" / "ltx" / "upscalers").is_dir()
    assert (tmp_path / "models" / "ltx" / "vae").is_dir()
    assert (tmp_path / "models" / "ltx" / "audio_vae").is_dir()
    assert (tmp_path / "models" / "ltx" / "text_encoder").is_dir()
    assert (tmp_path / "models" / "LLM" / "GGUF").is_dir()


def test_wan_download_categories_validate_file_type(tmp_path: Path):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    safe = ParsedRemote(source="direct", url="https://example.com/model.safetensors", filename="model.safetensors")
    gguf = ParsedRemote(source="direct", url="https://example.com/model.gguf", filename="model.gguf")

    assert service.destination_for("wan_safetensor", safe.filename) == (
        tmp_path / "models" / "wan" / "Safetensor" / "model.safetensors"
    )
    assert service.destination_for("wan_gguf", gguf.filename) == (
        tmp_path / "models" / "wan" / "GGUF" / "model.gguf"
    )
    with pytest.raises(ValueError, match="Wan transformer"):
        service.download_parsed(gguf, category="wan_safetensor")
    with pytest.raises(ValueError, match="Wan transformer"):
        service.download_parsed(safe, category="wan_gguf")


def test_flux_download_categories_validate_file_type(tmp_path: Path):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    gguf = ParsedRemote(source="direct", url="https://example.com/flux.gguf", filename="flux.gguf")
    safetensors = ParsedRemote(
        source="direct",
        url="https://example.com/clip_l.safetensors",
        filename="clip_l.safetensors",
    )

    assert service.destination_for("flux_unet_gguf", gguf.filename) == (
        tmp_path / "models" / "flux" / "GGUF" / "flux.gguf"
    )
    assert service.destination_for("flux_unet_safetensor", safetensors.filename) == (
        tmp_path / "models" / "flux" / "UNet" / "clip_l.safetensors"
    )
    assert service.destination_for("flux_text_encoder", safetensors.filename) == (
        tmp_path / "models" / "flux" / "Textencoder" / "clip_l.safetensors"
    )
    assert service.destination_for("flux_vae", safetensors.filename) == (
        tmp_path / "models" / "flux" / "VAE" / "clip_l.safetensors"
    )
    with pytest.raises(ValueError, match="Flux UNet"):
        service.download_parsed(safetensors, category="flux_unet_gguf")
    with pytest.raises(ValueError, match="Flux UNet"):
        service.download_parsed(gguf, category="flux_unet_safetensor")
    with pytest.raises(ValueError, match="Flux VAE"):
        service.download_parsed(gguf, category="flux_vae")
    with pytest.raises(ValueError, match="Flux text encoder"):
        service.download_parsed(gguf, category="flux_text_encoder")


def test_flux2_and_z_image_download_categories_validate_file_type(tmp_path: Path):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    gguf = ParsedRemote(source="direct", url="https://example.com/model.gguf", filename="model.gguf")
    safetensors = ParsedRemote(source="direct", url="https://example.com/model.safetensors", filename="model.safetensors")

    assert service.destination_for("flux2_unet_gguf", gguf.filename) == (
        tmp_path / "models" / "flux2" / "GGUF" / "model.gguf"
    )
    assert service.destination_for("flux2_unet_safetensor", safetensors.filename) == (
        tmp_path / "models" / "flux2" / "UNet" / "model.safetensors"
    )
    assert service.destination_for("z_image_unet_gguf", gguf.filename) == (
        tmp_path / "models" / "z-image" / "GGUF" / "model.gguf"
    )
    assert service.destination_for("z_image_unet_safetensor", safetensors.filename) == (
        tmp_path / "models" / "z-image" / "UNet" / "model.safetensors"
    )
    with pytest.raises(ValueError, match="Flux.2 Klein transformer"):
        service.download_parsed(safetensors, category="flux2_unet_gguf")
    with pytest.raises(ValueError, match="Z-Image transformer"):
        service.download_parsed(safetensors, category="z_image_unet_gguf")


def test_krea2_and_anima_download_categories_validate_file_type(tmp_path: Path):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    safetensors = ParsedRemote(source="direct", url="https://example.com/model.safetensors", filename="model.safetensors")
    gguf = ParsedRemote(source="direct", url="https://example.com/model.gguf", filename="model.gguf")

    assert service.destination_for("krea2_unet_safetensor", safetensors.filename) == (
        tmp_path / "models" / "krea2" / "UNet" / "model.safetensors"
    )
    assert service.destination_for("krea2_text_encoder", safetensors.filename) == (
        tmp_path / "models" / "krea2" / "Textencoder" / "model.safetensors"
    )
    assert service.destination_for("krea2_vae", safetensors.filename) == (
        tmp_path / "models" / "krea2" / "VAE" / "model.safetensors"
    )
    assert service.destination_for("anima_unet_safetensor", safetensors.filename) == (
        tmp_path / "models" / "anima" / "UNet" / "model.safetensors"
    )
    assert service.destination_for("anima_text_encoder", safetensors.filename) == (
        tmp_path / "models" / "anima" / "Textencoder" / "model.safetensors"
    )
    with pytest.raises(ValueError, match="Krea 2 transformer"):
        service.download_parsed(gguf, category="krea2_unet_safetensor")
    with pytest.raises(ValueError, match="Anima transformer"):
        service.download_parsed(gguf, category="anima_unet_safetensor")


def test_ltx_download_categories_validate_file_type(tmp_path: Path):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    safetensors = ParsedRemote(source="direct", url="https://example.com/ltx.safetensors", filename="ltx.safetensors")
    gguf = ParsedRemote(source="direct", url="https://example.com/ltx.gguf", filename="ltx.gguf")

    assert service.destination_for("ltx_checkpoint", safetensors.filename) == (
        tmp_path / "models" / "ltx" / "checkpoints" / "ltx.safetensors"
    )
    assert service.destination_for("ltx_gguf", gguf.filename) == (
        tmp_path / "models" / "ltx" / "GGUF" / "ltx.gguf"
    )
    assert service.destination_for("ltx_upscaler", safetensors.filename) == (
        tmp_path / "models" / "ltx" / "upscalers" / "ltx.safetensors"
    )
    assert service.destination_for("ltx_lora", safetensors.filename) == (
        tmp_path / "models" / "ltx" / "loras" / "ltx.safetensors"
    )
    assert service.destination_for("ltx_vae", safetensors.filename) == (
        tmp_path / "models" / "ltx" / "vae" / "ltx.safetensors"
    )
    assert service.destination_for("ltx_audio_vae", safetensors.filename) == (
        tmp_path / "models" / "ltx" / "audio_vae" / "ltx.safetensors"
    )
    with pytest.raises(ValueError, match="LTX 2.3 checkpoint"):
        service.download_parsed(gguf, category="ltx_checkpoint")
    with pytest.raises(ValueError, match="LTX 2.3 GGUF"):
        service.download_parsed(safetensors, category="ltx_gguf")


def test_llm_download_categories_validate_file_type(tmp_path: Path):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    gguf = ParsedRemote(source="direct", url="https://example.com/model.gguf", filename="model.gguf")
    safetensors = ParsedRemote(source="direct", url="https://example.com/model.safetensors", filename="model.safetensors")

    assert service.destination_for("llm_gguf", gguf.filename) == (
        tmp_path / "models" / "LLM" / "GGUF" / "model.gguf"
    )
    assert service.destination_for("llm_safetensor", safetensors.filename) == (
        tmp_path / "models" / "LLM" / "model.safetensors"
    )
    with pytest.raises(ValueError, match="LLM GGUF"):
        service.download_parsed(safetensors, category="llm_gguf")
    with pytest.raises(ValueError, match="LLM safetensors"):
        service.download_parsed(gguf, category="llm_safetensor")


def test_wan_diffusers_rejects_single_file_downloads(tmp_path: Path):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    remote = ParsedRemote(
        source="direct",
        url="https://example.com/diffusion_pytorch_model.safetensors",
        filename="diffusion_pytorch_model.safetensors",
    )

    with pytest.raises(ValueError, match="full Hugging Face repository folders"):
        service.download_parsed(remote, category="wan_diffusers")


def test_hf_snapshot_allowed_for_wan_diffusers(tmp_path: Path, monkeypatch):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    remote = service.parse_reference(
        source="huggingface",
        url_or_repo="Wan-AI/Wan2.2-I2V-A14B",
        filename="",
        category="wan_diffusers",
    )
    assert remote.snapshot is True
    assert remote.repo_id == "Wan-AI/Wan2.2-I2V-A14B"

    def fake_snapshot_download(*, repo_id, local_dir, token=None):
        target = Path(local_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "model_index.json").write_text("{}", encoding="utf-8")
        return str(target)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    path = service.download_parsed(remote, category="wan_diffusers")
    assert path == tmp_path / "models" / "wan" / "Diffusers" / "Wan2.2-I2V-A14B"
    assert (path / "model_index.json").is_file()


def test_wan_video_bundle_includes_filtered_components_base():
    assert "wan-ti2v-components" in QUICK_START_BUNDLES["video"]
    entry = next(item for item in model_download_module.MODEL_DOWNLOAD_CATALOG if item.key == "wan-ti2v-components")
    assert entry.snapshot is True
    assert "transformer/**" not in entry.snapshot_allow_patterns
    assert "text_encoder/model-*.safetensors" in entry.snapshot_allow_patterns
    assert "tokenizer/**" in entry.snapshot_allow_patterns


def test_wan_ti2v_diffusers_setup_has_a_complete_snapshot_bundle(tmp_path: Path):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))

    assert QUICK_START_BUNDLES["wan-ti2v-diffusers"] == ["wan-ti2v-diffusers-5b"]
    entry = service.find_catalog("wan-ti2v-diffusers-5b")
    assert entry is not None
    assert entry.snapshot is True
    assert service.snapshot_destination_for(entry.category, entry.repo_id) == (
        tmp_path / "models" / "wan" / "Diffusers" / "Wan2.2-TI2V-5B-Diffusers"
    )


def test_wan_ti2v_diffusers_catalog_readiness_requires_tokenizer_and_scheduler(tmp_path: Path):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    entry = service.find_catalog("wan-ti2v-diffusers-5b")
    assert entry is not None
    target = service.snapshot_destination_for(entry.category, entry.repo_id)
    (target / "text_encoder").mkdir(parents=True)
    (target / "text_encoder" / "config.json").write_text("{}", encoding="utf-8")
    (target / "text_encoder" / "model.safetensors.index.json").write_text(
        '{"weight_map":{"weight":"model-00001-of-00001.safetensors"}}', encoding="utf-8"
    )
    (target / "text_encoder" / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    (target / "transformer").mkdir()
    (target / "transformer" / "config.json").write_text("{}", encoding="utf-8")
    (target / "transformer" / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")
    (target / "vae").mkdir()
    (target / "vae" / "config.json").write_text("{}", encoding="utf-8")
    (target / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")
    (target / "model_index.json").write_text(
        '{"transformer":["diffusers","WanTransformer3DModel"],'
        '"text_encoder":["transformers","T5EncoderModel"],"vae":["diffusers","AutoencoderKL"]}',
        encoding="utf-8",
    )
    assert service._catalog_snapshot_ready(entry, target) is False
    for folder, name in (("tokenizer", "tokenizer.json"), ("scheduler", "scheduler_config.json")):
        (target / folder).mkdir()
        (target / folder / name).write_text("{}", encoding="utf-8")
    assert service._catalog_snapshot_ready(entry, target) is True


def test_flux_conditioning_bundle_installs_only_sidecars():
    keys = QUICK_START_BUNDLES["flux-components"]
    assert set(keys) == {"flux-t5-fp8", "flux-clip-l", "flux-ae-vae", "flux-clip-tokenizer", "flux-t5-tokenizer"}
    assert not any(key.startswith("flux-dev-") for key in keys)
    entries = {item.key: item for item in model_download_module.MODEL_DOWNLOAD_CATALOG if item.key in keys}
    assert entries["flux-t5-fp8"].category == "flux_text_encoder"
    assert entries["flux-clip-l"].category == "flux_text_encoder"
    assert entries["flux-ae-vae"].category == "flux_vae"
    assert entries["flux-clip-tokenizer"].repo_id == "openai/clip-vit-large-patch14"
    assert entries["flux-t5-tokenizer"].repo_id == "google/t5-v1_1-xxl"


def test_flux_distillt5_control_bundle_matches_backend_asset_paths(tmp_path: Path, monkeypatch):
    keys = QUICK_START_BUNDLES["flux-distillt5-control-components"]
    assert set(keys) == {
        "flux-distillt5-control", "flux-distillt5-tokenizer", "flux-clip-l", "flux-ae-vae", "flux-clip-tokenizer",
    }
    entries = {item.key: item for item in model_download_module.MODEL_DOWNLOAD_CATALOG if item.key in keys}
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    distill = entries["flux-distillt5-control"]
    tokenizer = entries["flux-distillt5-tokenizer"]
    assert distill.repo_id == "LifuWang/DistillT5"
    assert distill.category == "flux_text_encoder"
    assert distill.snapshot_allow_patterns == ("config.json", "model.safetensors")
    assert service.snapshot_destination_for(distill.category, distill.repo_id) == (
        tmp_path / "models" / "flux" / "Textencoder" / "DistillT5"
    )
    assert tokenizer.repo_id == "google/t5-v1_1-base"
    assert tokenizer.category == "flux_tokenizer"
    assert service.snapshot_destination_for(tokenizer.category, tokenizer.repo_id) == (
        tmp_path / "models" / "flux" / "tokenizer" / "t5-v1_1-base"
    )
    assert "flux-t5-fp8" not in keys

    distill_dir = service.snapshot_destination_for(distill.category, distill.repo_id)
    distill_dir.mkdir(parents=True)
    (distill_dir / "config.json").write_text("{}", encoding="utf-8")
    assert service.is_catalog_installed(distill) is False
    (distill_dir / "model.safetensors").write_bytes(b"weights")
    assert service.is_catalog_installed(distill) is True

    tokenizer_dir = service.snapshot_destination_for(tokenizer.category, tokenizer.repo_id)
    tokenizer_dir.mkdir(parents=True)
    (tokenizer_dir / "spiece.model").write_bytes(b"tokenizer")
    assert service.is_catalog_installed(tokenizer) is False
    (tokenizer_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    assert service.is_catalog_installed(tokenizer) is True


def test_flux_tokenizer_catalog_readiness_requires_every_runtime_file(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty-hub-cache"))
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "empty-hub-cache"))
    monkeypatch.setenv("TRANSFORMERS_CACHE", str(tmp_path / "empty-hub-cache"))
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", tmp_path / "empty-hub-cache")
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    clip = service.find_catalog("flux-clip-tokenizer")
    t5 = service.find_catalog("flux-t5-tokenizer")
    assert clip is not None and t5 is not None
    clip_target = service.snapshot_destination_for(clip.category, clip.repo_id)
    t5_target = service.snapshot_destination_for(t5.category, t5.repo_id)

    for name in ("vocab.json", "merges.txt"):
        path = clip_target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    assert service.is_catalog_installed(clip) is False
    (clip_target / "tokenizer_config.json").write_bytes(b"{}")
    assert service.is_catalog_installed(clip) is True

    (t5_target / "spiece.model").parent.mkdir(parents=True, exist_ok=True)
    (t5_target / "spiece.model").write_bytes(b"fixture")
    assert service.is_catalog_installed(t5) is False
    (t5_target / "tokenizer_config.json").write_bytes(b"{}")
    assert service.is_catalog_installed(t5) is True


def test_flux_tokenizer_catalog_recognizes_complete_huggingface_cache(tmp_path: Path, monkeypatch):
    cache = tmp_path / "hf-cache"
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(cache))
    monkeypatch.setenv("TRANSFORMERS_CACHE", str(cache))
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", cache)
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    entry = service.find_catalog("flux-clip-tokenizer")
    assert entry is not None
    snapshot = cache / "models--openai--clip-vit-large-patch14" / "snapshots" / "revision-a"
    snapshot.mkdir(parents=True)
    for name in ("vocab.json", "merges.txt", "tokenizer_config.json"):
        (snapshot / name).write_text("{}", encoding="utf-8")

    assert service.is_catalog_installed(entry) is True

    (snapshot / "merges.txt").unlink()
    assert service.is_catalog_installed(entry) is False


@pytest.mark.parametrize(
    ("key", "files"),
    [
        ("flux-clip-tokenizer", {"vocab.json": "{}", "merges.txt": "merge", "tokenizer_config.json": "{}"}),
        ("flux-t5-tokenizer", {"spiece.model": "model", "tokenizer_config.json": "{}"}),
        (
            "ltx-t5-tokenizer",
            {
                "config.json": '{"model_type":"t5","vocab_size":32128}',
                "special_tokens_map.json": "{}",
                "spiece.model": "model",
                "tokenizer_config.json": "{}",
            },
        ),
    ],
)
def test_flux_and_ltx_tokenizer_catalog_downloads_are_supported(tmp_path: Path, monkeypatch, key: str, files: dict[str, str]):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    entry = service.find_catalog(key)
    assert entry is not None
    downloaded = []

    def download_snapshot(remote, category, *, on_progress=None, validator=None):
        target = service.snapshot_destination_for(category, remote.repo_id)
        staging = target.parent / f".{target.name}.install-test"
        staging.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            (staging / name).write_text(content, encoding="utf-8")
        assert validator is None or validator(staging)
        target.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            (target / name).write_text(content, encoding="utf-8")
        downloaded.append((remote.repo_id, tuple(remote.snapshot_allow_patterns)))
        return target

    monkeypatch.setattr(service, "_download_hf_snapshot", download_snapshot)
    monkeypatch.setattr(service, "_invalidate_model_inventory", lambda: None)

    result = service.download_catalog(key)

    assert result == service.snapshot_destination_for(entry.category, entry.repo_id)
    assert downloaded and downloaded[0][0] == entry.repo_id
    assert service.is_catalog_installed(entry)


def test_wan_components_snapshot_readiness_checks_all_indexed_shards(tmp_path: Path):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    entry = service.find_catalog("wan-ti2v-components")
    assert entry is not None
    target = service.snapshot_destination_for(entry.category, entry.repo_id)
    for relative, content in {
        "model_index.json": "{}",
        "text_encoder/config.json": "{}",
        "text_encoder/model.safetensors.index.json": '{"weight_map":{"weight":"model-00001-of-00001.safetensors"}}',
        "tokenizer/tokenizer.json": "{}",
        "scheduler/scheduler_config.json": "{}",
    }.items():
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    assert service.is_catalog_installed(entry) is False
    shard = target / "text_encoder" / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"model-shard")
    assert service.is_catalog_installed(entry) is True

    index_path = target / "text_encoder" / "model.safetensors.index.json"
    index_path.write_text('{"weight_map":{"weight":"../outside.safetensors"}}', encoding="utf-8")
    (target / "outside.safetensors").write_bytes(b"outside")
    assert service.is_catalog_installed(entry) is False


def test_wan_components_snapshot_download_passes_allow_patterns(tmp_path: Path, monkeypatch):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    entry = service.find_catalog("wan-ti2v-components")
    assert entry is not None
    captured = {}

    def fake_snapshot_download(*, repo_id, local_dir, token=None, allow_patterns=None):
        captured.update(repo_id=repo_id, local_dir=local_dir, token=token, allow_patterns=allow_patterns)
        target = Path(local_dir)
        files = {
            "model_index.json": "{}",
            "text_encoder/config.json": "{}",
            "text_encoder/model.safetensors.index.json": '{"weight_map":{"weight":"model-00001-of-00001.safetensors"}}',
            "text_encoder/model-00001-of-00001.safetensors": "weight",
            "tokenizer/tokenizer.json": "{}",
            "scheduler/scheduler_config.json": "{}",
        }
        for relative, content in files.items():
            path = target / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        return str(target)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    path = service.download_catalog(entry.key)

    assert captured["repo_id"] == entry.repo_id
    assert captured["allow_patterns"] == list(entry.snapshot_allow_patterns)
    assert path == service.snapshot_destination_for(entry.category, entry.repo_id)
    assert service.is_catalog_installed(entry) is True


def test_flux2_full_snapshot_download_filters_repository_extras(tmp_path: Path, monkeypatch):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    entry = service.find_catalog("flux2-klein-4b-diffusers")
    assert entry is not None
    captured = {}

    def fake_snapshot_download(*, repo_id, local_dir, token=None, allow_patterns=None):
        captured.update(repo_id=repo_id, local_dir=local_dir, token=token, allow_patterns=allow_patterns)
        target = Path(local_dir)
        _write_support_components(target, family="flux2")
        transformer = target / "transformer"
        transformer.mkdir(parents=True, exist_ok=True)
        (transformer / "config.json").write_text(
            json.dumps({"_class_name": "Flux2Transformer2DModel", "in_channels": 128, "num_layers": 5}),
            encoding="utf-8",
        )
        _write_component_safetensors(transformer / "diffusion_pytorch_model.safetensors")
        return str(target)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    path = service.download_catalog(entry.key)

    assert captured["repo_id"] == entry.repo_id
    assert captured["allow_patterns"] == list(entry.snapshot_allow_patterns)
    assert "transformer/**" in captured["allow_patterns"]
    assert path == service.snapshot_destination_for(entry.category, entry.repo_id)
    assert Path(captured["local_dir"]) != path
    assert not Path(captured["local_dir"]).exists()
    assert service.is_catalog_installed(entry) is True


def test_flux2_full_snapshot_download_repairs_incomplete_destination_recoverably(tmp_path: Path, monkeypatch):
    from aiwf.infrastructure.model_inventory import scan_model_inventory

    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models")
    service = ModelDownloadService(flags)
    entry = service.find_catalog("flux2-klein-4b-diffusers")
    assert entry is not None
    target = service.snapshot_destination_for(entry.category, entry.repo_id)
    target.mkdir(parents=True)
    (target / "model_index.json").write_text('{"_class_name":"Flux2KleinPipeline"}', encoding="utf-8")
    (target / "user-note.txt").write_text("preserve previous folder", encoding="utf-8")

    def fake_snapshot_download(*, repo_id, local_dir, token=None, allow_patterns=None):
        staging = Path(local_dir)
        _write_support_components(staging, family="flux2")
        transformer = staging / "transformer"
        transformer.mkdir(parents=True, exist_ok=True)
        (transformer / "config.json").write_text(
            json.dumps({"_class_name": "Flux2Transformer2DModel", "in_channels": 128, "num_layers": 5}),
            encoding="utf-8",
        )
        _write_component_safetensors(transformer / "diffusion_pytorch_model.safetensors")
        return str(staging)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    installed = service.download_catalog(entry.key)

    assert installed == target
    assert service.is_catalog_installed(entry) is True
    recoveries = list((flags.resolved_models_dir() / ".aiwf-recovery" / entry.category).glob(f"{target.name}-*"))
    assert len(recoveries) == 1
    assert (recoveries[0] / "user-note.txt").read_text(encoding="utf-8") == "preserve previous folder"
    assert not list(target.parent.glob(f".{target.name}.install-*"))
    inventory = scan_model_inventory(flags)
    assert all(".aiwf-recovery" not in str(record.path) for record in inventory)


def test_flux2_component_download_repairs_incomplete_destination_recoverably(tmp_path: Path, monkeypatch):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    entry = service.find_catalog("flux2-klein-9b-components")
    assert entry is not None
    target = service.snapshot_destination_for(entry.category, entry.repo_id)
    target.mkdir(parents=True)
    (target / "model_index.json").write_text("{}", encoding="utf-8")
    (target / "partial.bin").write_bytes(b"partial")

    def fake_snapshot_download(*, repo_id, local_dir, token=None, allow_patterns=None):
        _write_support_components(Path(local_dir), family="flux2")
        return local_dir

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    installed = service.download_catalog(entry.key)

    assert installed == target
    assert service.is_catalog_installed(entry) is True
    recoveries = list((service.models_root() / ".aiwf-recovery" / entry.category).glob(f"{target.name}-*"))
    assert len(recoveries) == 1
    assert (recoveries[0] / "partial.bin").read_bytes() == b"partial"


def test_incomplete_replacement_snapshot_leaves_existing_folder_untouched(tmp_path: Path, monkeypatch):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    entry = service.find_catalog("flux2-klein-4b-diffusers")
    assert entry is not None
    target = service.snapshot_destination_for(entry.category, entry.repo_id)
    target.mkdir(parents=True)
    (target / "model_index.json").write_text('{"_class_name":"Flux2KleinPipeline"}', encoding="utf-8")
    (target / "user-note.txt").write_text("keep me", encoding="utf-8")

    def fake_snapshot_download(*, repo_id, local_dir, token=None, allow_patterns=None):
        staging = Path(local_dir)
        _write_support_components(staging, family="flux2")
        return str(staging)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    with pytest.raises(ValueError, match="incomplete"):
        service.download_catalog(entry.key)

    assert (target / "user-note.txt").read_text(encoding="utf-8") == "keep me"
    recovery_root = service.models_root() / ".aiwf-recovery" / entry.category
    assert not recovery_root.exists() or not any(recovery_root.iterdir())
    assert not list(target.parent.glob(f".{target.name}.install-*"))


def test_interrupted_snapshot_download_cleans_staging_and_preserves_existing_folder(tmp_path: Path, monkeypatch):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    entry = service.find_catalog("flux2-klein-4b-diffusers")
    assert entry is not None
    target = service.snapshot_destination_for(entry.category, entry.repo_id)
    target.mkdir(parents=True)
    (target / "user-note.txt").write_text("keep me", encoding="utf-8")

    def interrupted_snapshot_download(*, repo_id, local_dir, token=None, allow_patterns=None):
        (Path(local_dir) / "partial.bin").write_bytes(b"partial")
        raise KeyboardInterrupt()

    monkeypatch.setattr("huggingface_hub.snapshot_download", interrupted_snapshot_download)
    with pytest.raises(KeyboardInterrupt):
        service.download_catalog(entry.key)

    assert (target / "user-note.txt").read_text(encoding="utf-8") == "keep me"
    assert not (service.models_root() / ".aiwf-recovery").exists()
    assert not list(target.parent.glob(f".{target.name}.install-*"))


def test_snapshot_publish_failure_restores_incomplete_existing_folder(tmp_path: Path, monkeypatch):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    entry = service.find_catalog("flux2-klein-4b-diffusers")
    assert entry is not None
    target = service.snapshot_destination_for(entry.category, entry.repo_id)
    target.mkdir(parents=True)
    (target / "user-note.txt").write_text("keep me", encoding="utf-8")

    def fake_snapshot_download(*, repo_id, local_dir, token=None, allow_patterns=None):
        staging = Path(local_dir)
        _write_support_components(staging, family="flux2")
        transformer = staging / "transformer"
        transformer.mkdir(parents=True, exist_ok=True)
        (transformer / "config.json").write_text(
            json.dumps({"_class_name": "Flux2Transformer2DModel", "in_channels": 128, "num_layers": 5}),
            encoding="utf-8",
        )
        _write_component_safetensors(transformer / "diffusion_pytorch_model.safetensors")
        return str(staging)

    original_rename = model_download_module.os.rename
    failed = False

    def fail_staging_publish(source, destination):
        nonlocal failed
        if not failed and Path(source).name.startswith(f".{target.name}.install-") and Path(destination) == target:
            failed = True
            raise OSError("simulated publish failure")
        return original_rename(source, destination)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(model_download_module.os, "rename", fail_staging_publish)
    with pytest.raises(ValueError, match="destination changed"):
        service.download_catalog(entry.key)

    assert failed
    assert (target / "user-note.txt").read_text(encoding="utf-8") == "keep me"
    recovery_root = service.models_root() / ".aiwf-recovery" / entry.category
    assert not recovery_root.exists() or not any(recovery_root.iterdir())
    assert not list(target.parent.glob(f".{target.name}.install-*"))


def test_startup_recovers_snapshot_quarantined_before_interrupted_publish(tmp_path: Path):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    entry = service.find_catalog("flux2-klein-4b-diffusers")
    assert entry is not None
    target = service.snapshot_destination_for(entry.category, entry.repo_id)
    previous = (
        service.models_root()
        / ".aiwf-recovery"
        / entry.category
        / f"{target.name}-20261008T001234Z-abcd1234"
    )
    previous.mkdir(parents=True)
    (previous / "model_index.json").write_text('{"_class_name":"Flux2KleinPipeline"}', encoding="utf-8")

    recovered = service.recover_interrupted_snapshot_replacements()

    assert recovered == [target]
    assert (target / "model_index.json").is_file()
    assert not previous.exists()


def test_startup_recovery_leaves_quarantine_when_destination_exists(tmp_path: Path):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    entry = service.find_catalog("flux2-klein-4b-diffusers")
    assert entry is not None
    target = service.snapshot_destination_for(entry.category, entry.repo_id)
    target.mkdir(parents=True)
    (target / "current.txt").write_text("current", encoding="utf-8")
    previous = (
        service.models_root()
        / ".aiwf-recovery"
        / entry.category
        / f"{target.name}-20261008T001234Z-abcd1234"
    )
    previous.mkdir(parents=True)
    (previous / "old.txt").write_text("old", encoding="utf-8")

    recovered = service.recover_interrupted_snapshot_replacements()

    assert recovered == []
    assert (target / "current.txt").read_text(encoding="utf-8") == "current"
    assert (previous / "old.txt").read_text(encoding="utf-8") == "old"


def test_startup_recovers_preprocessor_category_snapshot_destination(tmp_path: Path):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    target = service.snapshot_destination_for("preprocessor", "annotators")
    previous = (
        service.models_root()
        / ".aiwf-recovery"
        / "preprocessor"
        / f"{target.name}-20261008T001234Z-abcd1234"
    )
    previous.mkdir(parents=True)
    (previous / "install-note.txt").write_text("preserved", encoding="utf-8")

    recovered = service.recover_interrupted_snapshot_replacements()

    assert recovered == [target]
    assert (target / "install-note.txt").read_text(encoding="utf-8") == "preserved"


def test_sana_video_catalog_install_requires_route_complete_snapshot(tmp_path: Path):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    entry = service.find_catalog("sana-video-2b-480p-diffusers")
    assert entry is not None
    target = service.snapshot_destination_for(entry.category, entry.repo_id)
    transformer = target / "transformer"
    transformer.mkdir(parents=True)
    (target / "model_index.json").write_text(
        '{"_class_name":"SanaVideoPipeline","transformer":["diffusers","SanaVideoTransformer3DModel"]}',
        encoding="utf-8",
    )
    (transformer / "config.json").write_text('{"in_channels":16}', encoding="utf-8")
    (transformer / "diffusion_pytorch_model.safetensors").write_bytes(b"weight")

    # Model-download setup must use the same route-specific requirements as
    # preflight; a lone transformer does not make this Sana Video snapshot ready.
    assert service._snapshot_target_ready(entry.category, target) is False


def test_incomplete_catalog_snapshot_is_removed_from_staging_not_published(tmp_path: Path, monkeypatch):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    entry = service.find_catalog("sana-video-2b-480p-diffusers")
    assert entry is not None

    def fake_snapshot_download(*, repo_id, local_dir, token=None, allow_patterns=None):
        staging = Path(local_dir)
        (staging / "model_index.json").write_text(
            '{"_class_name":"SanaVideoPipeline"}', encoding="utf-8"
        )
        return str(staging)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    with pytest.raises(ValueError, match="incomplete"):
        service.download_catalog(entry.key)

    target = service.snapshot_destination_for(entry.category, entry.repo_id)
    assert not target.exists()
    assert list(target.parent.glob(f".{target.name}.install-*")) == []
    assert service.is_catalog_installed(entry) is False


def test_hf_snapshot_allowed_for_checkpoint_diffusers_folder(tmp_path: Path, monkeypatch):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    remote = service.parse_reference(
        source="huggingface",
        url_or_repo="stabilityai/stable-diffusion-3.5-medium",
        filename="",
        category="checkpoint",
    )
    assert remote.snapshot is True
    assert remote.repo_id == "stabilityai/stable-diffusion-3.5-medium"

    def fake_snapshot_download(*, repo_id, local_dir, token=None):
        target = Path(local_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "model_index.json").write_text(
            '{"_class_name": "StableDiffusion3Pipeline"}',
            encoding="utf-8",
        )
        return str(target)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    path = service.download_parsed(remote, category="checkpoint")
    assert path == tmp_path / "models" / "Stable-diffusion" / "stable-diffusion-3.5-medium"
    assert (path / "model_index.json").is_file()


def test_hf_snapshot_allowed_for_ltx_text_encoder(tmp_path: Path, monkeypatch):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    remote = service.parse_reference(
        source="huggingface",
        url_or_repo="google/gemma-3-12b-it-qat-q4_0-unquantized",
        filename="",
        category="ltx_text_encoder",
    )
    assert remote.snapshot is True

    def fake_snapshot_download(*, repo_id, local_dir, token=None):
        target = Path(local_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.json").write_text("{}", encoding="utf-8")
        return str(target)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    path = service.download_parsed(remote, category="ltx_text_encoder")
    assert path == tmp_path / "models" / "ltx" / "text_encoder" / "gemma-3-12b-it-qat-q4_0-unquantized"
    assert (path / "config.json").is_file()


def test_ltx_tokenizer_catalog_install_places_and_validates_small_snapshot(tmp_path: Path, monkeypatch):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    entry = service.find_catalog("ltx-t5-tokenizer")
    assert entry is not None

    def fake_snapshot_download(*, repo_id, local_dir, token=None, allow_patterns=None):
        assert repo_id == "google/t5-v1_1-xxl"
        assert set(allow_patterns or ()) == {
            "config.json", "special_tokens_map.json", "spiece.model", "tokenizer_config.json",
        }
        target = Path(local_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.json").write_text('{"model_type":"t5","vocab_size":32128}', encoding="utf-8")
        (target / "special_tokens_map.json").write_text('{"pad_token":"<pad>"}', encoding="utf-8")
        (target / "spiece.model").write_bytes(b"sentencepiece")
        (target / "tokenizer_config.json").write_text('{"extra_ids":100}', encoding="utf-8")
        return str(target)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    path = service.download_catalog("ltx-t5-tokenizer")

    assert path == tmp_path / "models" / "ltx" / "tokenizer" / "t5-v1_1-xxl"
    assert service._catalog_snapshot_ready(entry, path) is True


@pytest.mark.parametrize(
    "category,repo_id,expected",
    [
        ("flux2_components", "black-forest-labs/FLUX.2-klein-4B", ("flux2", "Components", "FLUX.2-klein-4B")),
        ("flux2_diffusers", "black-forest-labs/FLUX.2-klein-4B", ("flux2", "Diffusers", "FLUX.2-klein-4B")),
        ("z_image_components", "Tongyi-MAI/Z-Image-Turbo", ("z-image", "Components", "Z-Image-Turbo")),
        ("krea2_diffusers", "krea/Krea-2-Turbo", ("krea2", "Diffusers", "Krea-2-Turbo")),
        ("qwen_image_diffusers", "Qwen/Qwen-Image-2512", ("qwen-image", "Diffusers", "Qwen-Image-2512")),
        (
            "sana_diffusers",
            "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
            ("sana", "Diffusers", "Sana_Sprint_1.6B_1024px_diffusers"),
        ),
    ],
)
def test_hf_snapshot_allowed_for_image_runtime_folders(
    tmp_path: Path,
    monkeypatch,
    category,
    repo_id,
    expected,
):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    remote = service.parse_reference(
        source="huggingface",
        url_or_repo=repo_id,
        filename="",
        category=category,
    )
    assert remote.snapshot is True

    def fake_snapshot_download(*, repo_id, local_dir, token=None):
        target = Path(local_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "model_index.json").write_text("{}", encoding="utf-8")
        return str(target)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    path = service.download_parsed(remote, category=category)
    assert path == tmp_path / "models" / Path(*expected)
    assert (path / "model_index.json").is_file()


def test_snapshot_catalog_installed_requires_category_marker(tmp_path: Path):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    entry = service.find_catalog("hf-sd35-medium")
    assert entry is not None
    target = service.snapshot_destination_for(entry.category, entry.repo_id)
    (target / ".cache").mkdir(parents=True)

    assert service.is_catalog_installed(entry) is False
    (target / "model_index.json").write_text("{}", encoding="utf-8")
    (target / "transformer").mkdir()
    (target / "transformer" / "config.json").write_text("{}", encoding="utf-8")
    (target / "transformer" / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")
    (target / "model_index.json").write_text(
        '{"transformer":["diffusers","SD3Transformer2DModel"]}', encoding="utf-8"
    )
    assert service.is_catalog_installed(entry) is True


def test_snapshot_catalog_recognizes_shared_diffusers_folder(tmp_path: Path):
    shared = tmp_path / "ComfyUI Models"
    service = ModelDownloadService(RuntimeFlags(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        extra_model_dirs=[shared],
    ))
    entry = service.find_catalog("hf-sd35-medium")
    assert entry is not None
    target = shared / "Diffusers" / Path(entry.repo_id).name
    target.mkdir(parents=True)
    (target / "model_index.json").write_text(
        '{"transformer":["diffusers","SD3Transformer2DModel"]}', encoding="utf-8"
    )
    transformer = target / "transformer"
    transformer.mkdir()
    (transformer / "config.json").write_text("{}", encoding="utf-8")
    (transformer / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")

    assert service.is_catalog_installed(entry) is True


def test_shared_snapshot_import_is_size_gated_confirmed_and_source_preserving(tmp_path: Path, monkeypatch):
    shared = tmp_path / "shared-models"
    service = ModelDownloadService(RuntimeFlags(
        data_dir=tmp_path / "data", models_dir=tmp_path / "models", extra_model_dirs=[shared]
    ))
    entry = service.find_catalog("hf-sd35-medium")
    assert entry is not None
    source = shared / "Diffusers" / Path(entry.repo_id).name
    transformer = source / "transformer"
    transformer.mkdir(parents=True)
    (source / "model_index.json").write_text(
        '{"transformer":["diffusers","SD3Transformer2DModel"]}', encoding="utf-8"
    )
    (transformer / "config.json").write_text("{}", encoding="utf-8")
    (transformer / "diffusion_pytorch_model.safetensors").write_bytes(b"synthetic weights")
    size = service._tree_size_without_links(source)
    monkeypatch.setattr("aiwf.services.model_download.shutil.disk_usage", lambda _path: SimpleNamespace(free=10**9))

    preview = service.preview_shared_catalog_snapshot_import(entry)
    assert preview is not None
    assert preview["source"] == str(source)
    assert preview["sizeBytes"] == size
    assert preview["requiredBytes"] > size
    assert preview["enoughSpace"] is True

    target = service.snapshot_destination_for(entry.category, entry.repo_id)
    assert service.copy_shared_catalog_snapshot_to_primary(
        entry, expected_source=str(source), expected_size_bytes=size - 1
    ) is None
    copied = service.copy_shared_catalog_snapshot_to_primary(
        entry, expected_source=str(source), expected_size_bytes=size
    )
    assert copied == {"source": str(source), "target": str(target)}
    assert source.is_dir()
    assert service._tree_size_without_links(target) == size
    assert service._catalog_snapshot_ready(entry, target)


def test_flux_kontext_catalog_requires_component_weights_and_tokenizers(tmp_path: Path):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    entry = service.find_catalog("flux-kontext-diffusers")
    assert entry is not None
    parsed = service.parse_reference(source="huggingface", url_or_repo=entry.repo_id, category=entry.category)
    assert parsed.repo_id == "black-forest-labs/FLUX.1-Kontext-dev"
    target = service.snapshot_destination_for(entry.category, entry.repo_id)
    assert target == tmp_path / "models" / "flux" / "Components" / "FLUX.1-Kontext-dev"
    for name in ("transformer", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "vae"):
        (target / name).mkdir(parents=True)
    (target / "model_index.json").write_text(json.dumps({"_class_name": "FluxKontextPipeline"}), encoding="utf-8")

    assert service._catalog_snapshot_ready(entry, target) is False


    declarations = {
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "text_encoder": ["transformers", "CLIPTextModel"],
        "text_encoder_2": ["transformers", "T5EncoderModel"],
        "tokenizer": ["transformers", "CLIPTokenizer"],
        "tokenizer_2": ["transformers", "T5TokenizerFast"],
        "transformer": ["diffusers", "FluxTransformer2DModel"],
        "vae": ["diffusers", "AutoencoderKL"],
    }
    (target / "model_index.json").write_text(
        json.dumps({"_class_name": "FluxKontextPipeline", **declarations}), encoding="utf-8"
    )
    for name, values in {
        "scheduler/scheduler_config.json": {"_class_name": "FlowMatchEulerDiscreteScheduler"},
        "text_encoder/config.json": {"_class_name": "CLIPTextModel"},
        "text_encoder_2/config.json": {"_class_name": "T5EncoderModel"},
        "tokenizer/tokenizer_config.json": {"tokenizer_class": "CLIPTokenizer"},
        "tokenizer_2/tokenizer_config.json": {"tokenizer_class": "T5TokenizerFast"},
        "transformer/config.json": {"_class_name": "FluxTransformer2DModel"},
        "vae/config.json": {"_class_name": "AutoencoderKL"},
    }.items():
        file = target / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(json.dumps(values), encoding="utf-8")
    (target / "tokenizer" / "vocab.json").write_text("{}", encoding="utf-8")
    (target / "tokenizer" / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    (target / "tokenizer_2" / "spiece.model").write_bytes(b"sentencepiece")
    for component, name in (
        ("text_encoder", "pytorch_model.bin"),
        ("text_encoder_2", "pytorch_model.bin"),
        ("transformer", "diffusion_pytorch_model.bin"),
        ("vae", "diffusion_pytorch_model.bin"),
    ):
        (target / component / name).write_bytes(b"weights")

    assert service._catalog_snapshot_ready(entry, target) is True
    (target / "transformer" / "diffusion_pytorch_model.bin").unlink()
    assert service._catalog_snapshot_ready(entry, target) is False
    gguf_entry = service.find_catalog("flux-kontext-gguf-components")
    assert gguf_entry is not None
    assert gguf_entry.snapshot_allow_patterns
    assert service._catalog_snapshot_ready(gguf_entry, target) is True
    assert "transformer/**" not in gguf_entry.snapshot_allow_patterns


@pytest.mark.parametrize(
    ("key", "required"),
    [
        ("hf-sd15-singlefile-config", ("text_encoder/config.json", "tokenizer/tokenizer_config.json", "tokenizer/vocab.json", "tokenizer/merges.txt", "unet/config.json", "vae/config.json")),
        ("hf-sd15-inpaint-singlefile-config", ("text_encoder/config.json", "tokenizer/tokenizer_config.json", "tokenizer/vocab.json", "tokenizer/merges.txt", "unet/config.json", "vae/config.json")),
        ("hf-sdxl-singlefile-config", ("text_encoder/config.json", "text_encoder_2/config.json", "tokenizer/tokenizer_config.json", "tokenizer/vocab.json", "tokenizer/merges.txt", "tokenizer_2/tokenizer_config.json", "tokenizer_2/vocab.json", "tokenizer_2/merges.txt", "unet/config.json", "vae/config.json")),
        ("hf-sdxl-inpaint-singlefile-config", ("text_encoder/config.json", "text_encoder_2/config.json", "tokenizer/tokenizer_config.json", "tokenizer/vocab.json", "tokenizer/merges.txt", "tokenizer_2/tokenizer_config.json", "tokenizer_2/vocab.json", "tokenizer_2/merges.txt", "unet/config.json", "vae/config.json")),
        ("hf-sdxl-refiner-singlefile-config", ("text_encoder_2/config.json", "tokenizer_2/tokenizer_config.json", "tokenizer_2/vocab.json", "tokenizer_2/merges.txt", "unet/config.json", "vae/config.json")),
        ("hf-sd35-singlefile-config", ("text_encoder/config.json", "text_encoder_2/config.json", "text_encoder_3/config.json", "tokenizer/tokenizer_config.json", "tokenizer/vocab.json", "tokenizer/merges.txt", "tokenizer_2/tokenizer_config.json", "tokenizer_2/vocab.json", "tokenizer_2/merges.txt", "tokenizer_3/tokenizer_config.json", "tokenizer_3/spiece.model", "transformer/config.json", "vae/config.json")),
    ],
)
def test_sd_singlefile_config_catalog_requires_local_pipeline_configs(tmp_path: Path, key: str, required: tuple[str, ...]):
    import json
    from aiwf.infrastructure.diffusers.single_file_config import (
        SINGLE_FILE_CONFIG_REQUIRED_FILES,
        _EXPECTED_PIPELINE_PREFIX,
        _REQUIRED_COMPONENTS,
    )

    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    entry = service.find_catalog(key)
    assert entry is not None
    target = service.snapshot_destination_for(entry.category, entry.repo_id)
    assert service._catalog_snapshot_ready(entry, target) is False
    family = {
        "hf-sd15-singlefile-config": "sd15",
        "hf-sd15-inpaint-singlefile-config": "sd15_inpaint",
        "hf-sdxl-singlefile-config": "sdxl",
        "hf-sdxl-inpaint-singlefile-config": "sdxl_inpaint",
        "hf-sdxl-refiner-singlefile-config": "sdxl_refiner",
        "hf-sd35-singlefile-config": "sd35",
    }[key]
    model_index = {"_class_name": _EXPECTED_PIPELINE_PREFIX[family]}
    model_index.update({name: ["diffusers", f"{name.title()}Class"] for name in _REQUIRED_COMPONENTS[family]})
    for relative in SINGLE_FILE_CONFIG_REQUIRED_FILES[family]:
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == "model_index.json":
            path.write_text(json.dumps(model_index), encoding="utf-8")
        elif relative.endswith(".json"):
            path.write_text("{}", encoding="utf-8")
        else:
            path.write_bytes(b"asset")
    assert service._catalog_snapshot_ready(entry, target) is True
    for relative in required:
        path = target / relative
        original = path.read_bytes()
        path.unlink()
        assert service._catalog_snapshot_ready(entry, target) is False, relative
        path.write_bytes(original)


def test_sd_singlefile_config_bundle_download_uses_scoped_local_snapshot(tmp_path: Path, monkeypatch):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    calls = []

    def fake_snapshot_download(**kwargs):
        import json
        from aiwf.infrastructure.diffusers.single_file_config import _REQUIRED_COMPONENTS

        calls.append(kwargs)
        target = Path(kwargs["local_dir"])
        model_index = {"_class_name": "StableDiffusionPipeline"}
        model_index.update({name: ["diffusers", f"{name.title()}Class"] for name in _REQUIRED_COMPONENTS["sd15"]})
        for relative in (
            "model_index.json", "scheduler/scheduler_config.json", "text_encoder/config.json",
            "tokenizer/tokenizer_config.json", "tokenizer/vocab.json", "tokenizer/merges.txt",
            "unet/config.json", "vae/config.json",
        ):
            path = target / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if relative == "model_index.json":
                path.write_text(json.dumps(model_index), encoding="utf-8")
            elif relative.endswith(".json"):
                path.write_text("{}", encoding="utf-8")
            else:
                path.write_bytes(b"asset")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    installed = service.download_catalog("hf-sd15-singlefile-config")
    assert installed == service.snapshot_destination_for("sd_singlefile_config", "stable-diffusion-v1-5/stable-diffusion-v1-5")
    assert calls[0]["allow_patterns"] == list(service.find_catalog("hf-sd15-singlefile-config").snapshot_allow_patterns)
    assert service.is_catalog_installed(service.find_catalog("hf-sd15-singlefile-config")) is True


def test_shared_snapshot_import_does_not_dereference_a_link_added_after_preview(tmp_path: Path, monkeypatch):
    shared = tmp_path / "shared-models"
    service = ModelDownloadService(RuntimeFlags(
        data_dir=tmp_path / "data", models_dir=tmp_path / "models", extra_model_dirs=[shared]
    ))
    entry = service.find_catalog("hf-sd35-medium")
    assert entry is not None
    source = shared / "Diffusers" / Path(entry.repo_id).name
    transformer = source / "transformer"
    transformer.mkdir(parents=True)
    (source / "model_index.json").write_text(
        '{"transformer":["diffusers","SD3Transformer2DModel"]}', encoding="utf-8"
    )
    (transformer / "config.json").write_text("{}", encoding="utf-8")
    weight = transformer / "diffusion_pytorch_model.safetensors"
    weight.write_bytes(b"safe weights")
    external = tmp_path / "external.safetensors"
    external.write_bytes(b"evil weights")
    size = service._tree_size_without_links(source)
    monkeypatch.setattr("aiwf.services.model_download.shutil.disk_usage", lambda _path: SimpleNamespace(free=10**9))
    original_copytree = model_download_module.shutil.copytree

    def replace_with_link_then_copy(src, dst, **kwargs):
        weight.unlink()
        try:
            weight.symlink_to(external)
        except OSError as exc:
            pytest.skip(f"filesystem does not permit symlink creation: {exc}")
        monkeypatch.setattr(model_download_module.shutil, "copytree", original_copytree)
        return original_copytree(src, dst, **kwargs)

    monkeypatch.setattr(model_download_module.shutil, "copytree", replace_with_link_then_copy)
    copied = service.copy_shared_catalog_snapshot_to_primary(
        entry, expected_source=str(source), expected_size_bytes=size
    )

    assert copied is None
    assert not service.snapshot_destination_for(entry.category, entry.repo_id).exists()


def test_shared_snapshot_import_refuses_insufficient_space(tmp_path: Path, monkeypatch):
    shared = tmp_path / "shared-models"
    service = ModelDownloadService(RuntimeFlags(
        data_dir=tmp_path / "data", models_dir=tmp_path / "models", extra_model_dirs=[shared]
    ))
    entry = service.find_catalog("hf-sd35-medium")
    assert entry is not None
    source = shared / "Diffusers" / Path(entry.repo_id).name
    transformer = source / "transformer"
    transformer.mkdir(parents=True)
    (source / "model_index.json").write_text(
        '{"transformer":["diffusers","SD3Transformer2DModel"]}', encoding="utf-8"
    )
    (transformer / "config.json").write_text("{}", encoding="utf-8")
    (transformer / "diffusion_pytorch_model.safetensors").write_bytes(b"fixture")
    monkeypatch.setattr("aiwf.services.model_download.shutil.disk_usage", lambda _path: SimpleNamespace(free=0))

    preview = service.preview_shared_catalog_snapshot_import(entry)
    assert preview is not None and preview["enoughSpace"] is False
    assert service.copy_shared_catalog_snapshot_to_primary(
        entry, expected_source=str(source), expected_size_bytes=preview["sizeBytes"]
    ) is None
    assert source.is_dir()
    assert not service.snapshot_destination_for(entry.category, entry.repo_id).exists()


def test_snapshot_catalog_rejects_shared_marker_only_folder(tmp_path: Path):
    shared = tmp_path / "ComfyUI Models"
    service = ModelDownloadService(RuntimeFlags(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        extra_model_dirs=[shared],
    ))
    entry = service.find_catalog("hf-sd35-medium")
    assert entry is not None
    target = shared / "Diffusers" / Path(entry.repo_id).name
    target.mkdir(parents=True)
    (target / "model_index.json").write_text("{}", encoding="utf-8")

    assert service.is_catalog_installed(entry) is False


def test_catalog_recognizes_flux_assets_in_shared_comfyui_layout(tmp_path: Path):
    shared = tmp_path / "ComfyUI Models"
    flags = RuntimeFlags(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        extra_model_dirs=[shared],
    )
    service = ModelDownloadService(flags)
    entry = service.find_catalog("flux-clip-l")
    assert entry is not None
    filename = service._catalog_local_filename_hint(entry)
    asset = shared / "text_encoders" / "CLIP-L" / filename
    asset.parent.mkdir(parents=True)
    _write_large_safetensors(asset, service._catalog_min_bytes(entry) + 1)

    assert service.is_catalog_installed(entry) is True


def test_catalog_does_not_accept_unverified_misplaced_asset_by_name_and_size(tmp_path: Path):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    entry = service.find_catalog("flux-ae-vae")
    assert entry is not None
    filename = service._catalog_local_filename_hint(entry)
    misplaced = tmp_path / "models" / "Unsorted" / "misc" / filename
    misplaced.parent.mkdir(parents=True)
    misplaced.write_bytes(b"x" * (service._catalog_min_bytes(entry) + 1))

    assert service.is_catalog_installed(entry) is False
    assert service.is_catalog_installed(entry, search_misplaced=True) is False
    assert service.place_misplaced_catalog_asset(entry) is False
    assert misplaced.is_file()


def test_explicit_catalog_setup_places_one_header_verified_flux_vae(tmp_path: Path, monkeypatch):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    entry = service.find_catalog("flux-ae-vae")
    assert entry is not None
    filename = service._catalog_local_filename_hint(entry)
    misplaced = tmp_path / "models" / "Unsorted" / "misc" / filename
    misplaced.parent.mkdir(parents=True)
    _write_large_safetensors(misplaced, service._catalog_min_bytes(entry) + 1)
    monkeypatch.setattr(
        "aiwf.infrastructure.model_header.read_model_info",
        lambda _path: SimpleNamespace(arch="flux-vae", role="vae", tensor_count=12),
    )

    assert service.is_catalog_installed(entry, search_misplaced=True) is True
    assert service.place_misplaced_catalog_asset(entry) is True
    destination = service.destination_for(entry.category, filename)
    assert destination.is_file()
    assert not misplaced.exists()


def test_explicit_catalog_setup_keeps_misplaced_wrong_family_asset(tmp_path: Path, monkeypatch):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    entry = service.find_catalog("flux-ae-vae")
    assert entry is not None
    filename = service._catalog_local_filename_hint(entry)
    misplaced = tmp_path / "models" / "Unsorted" / filename
    misplaced.parent.mkdir(parents=True)
    with misplaced.open("wb") as handle:
        handle.truncate(service._catalog_min_bytes(entry) + 1)
    monkeypatch.setattr(
        "aiwf.infrastructure.model_header.read_model_info",
        lambda _path: SimpleNamespace(arch="wan-vae", role="vae", tensor_count=12),
    )

    assert service.place_misplaced_catalog_asset(entry) is False
    assert misplaced.is_file()


def test_explicit_catalog_setup_copies_verified_shared_asset_and_preserves_source(tmp_path: Path, monkeypatch):
    shared = tmp_path / "shared-models"
    flags = RuntimeFlags(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        extra_model_dirs=[shared],
    )
    service = ModelDownloadService(flags)
    entry = service.find_catalog("flux-ae-vae")
    assert entry is not None
    filename = service._catalog_local_filename_hint(entry)
    source = shared / "Unsorted" / filename
    source.parent.mkdir(parents=True)
    _write_large_safetensors(source, service._catalog_min_bytes(entry) + 1)
    monkeypatch.setattr(
        "aiwf.infrastructure.model_header.read_model_info",
        lambda _path: SimpleNamespace(arch="flux-vae", role="vae", tensor_count=12),
    )

    copied = service.copy_shared_catalog_asset_to_primary(entry)

    target = service.destination_for(entry.category, filename)
    assert copied == {"source": str(source), "target": str(target)}
    assert source.is_file()
    assert target.read_bytes() == source.read_bytes()


def test_shared_catalog_asset_import_rejects_destination_symlink_escape(tmp_path: Path, monkeypatch):
    shared = tmp_path / "shared-models"
    models = tmp_path / "models"
    outside = tmp_path / "outside"
    models.mkdir()
    outside.mkdir()
    try:
        (models / "flux").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")
    service = ModelDownloadService(RuntimeFlags(
        data_dir=tmp_path / "data", models_dir=models, extra_model_dirs=[shared]
    ))
    entry = service.find_catalog("flux-ae-vae")
    assert entry is not None
    filename = service._catalog_local_filename_hint(entry)
    source = shared / "Unsorted" / filename
    source.parent.mkdir(parents=True)
    source.write_bytes(b"x" * (service._catalog_min_bytes(entry) + 1))
    monkeypatch.setattr(
        "aiwf.infrastructure.model_header.read_model_info",
        lambda _path: SimpleNamespace(arch="flux-vae", role="vae", tensor_count=12),
    )

    assert service.copy_shared_catalog_asset_to_primary(entry) is None
    assert source.is_file()
    assert not (outside / "VAE").exists()


def test_shared_catalog_asset_import_rejects_primary_root_inside_shared_root(tmp_path: Path, monkeypatch):
    shared = tmp_path / "shared-parent"
    models = shared / "AIWF"
    service = ModelDownloadService(RuntimeFlags(
        data_dir=tmp_path / "data", models_dir=models, extra_model_dirs=[shared]
    ))
    entry = service.find_catalog("flux-ae-vae")
    assert entry is not None
    filename = service._catalog_local_filename_hint(entry)
    source = shared / "Unsorted" / filename
    source.parent.mkdir(parents=True)
    source.write_bytes(b"x" * (service._catalog_min_bytes(entry) + 1))
    monkeypatch.setattr(
        "aiwf.infrastructure.model_header.read_model_info",
        lambda _path: SimpleNamespace(arch="flux-vae", role="vae", tensor_count=12),
    )

    assert service.copy_shared_catalog_asset_to_primary(entry) is None
    assert source.is_file()
    assert not service.destination_for(entry.category, filename).exists()


def test_shared_catalog_asset_import_refuses_insufficient_space(tmp_path: Path, monkeypatch):
    shared = tmp_path / "shared-models"
    service = ModelDownloadService(RuntimeFlags(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        extra_model_dirs=[shared],
    ))
    entry = service.find_catalog("flux-ae-vae")
    assert entry is not None
    filename = service._catalog_local_filename_hint(entry)
    source = shared / "Unsorted" / filename
    source.parent.mkdir(parents=True)
    source.write_bytes(b"x" * (service._catalog_min_bytes(entry) + 1))
    monkeypatch.setattr(
        "aiwf.infrastructure.model_header.read_model_info",
        lambda _path: SimpleNamespace(arch="flux-vae", role="vae", tensor_count=12),
    )
    monkeypatch.setattr(
        "aiwf.services.model_download.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=0),
    )

    assert service.copy_shared_catalog_asset_to_primary(entry) is None
    assert source.is_file()
    assert not service.destination_for(entry.category, filename).exists()


def test_shared_snapshot_preview_rejects_destination_symlink_escape(tmp_path: Path, monkeypatch):
    shared = tmp_path / "shared-models"
    models = tmp_path / "models"
    outside = tmp_path / "outside"
    models.mkdir()
    outside.mkdir()
    try:
        (models / "Stable-diffusion").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")
    service = ModelDownloadService(RuntimeFlags(
        data_dir=tmp_path / "data", models_dir=models, extra_model_dirs=[shared]
    ))
    entry = service.find_catalog("hf-sd35-medium")
    assert entry is not None
    source = shared / "Diffusers" / Path(entry.repo_id).name
    source.mkdir(parents=True)
    monkeypatch.setattr(
        service,
        "_catalog_snapshot_ready",
        lambda _entry, candidate: Path(candidate).resolve() == source.resolve(),
    )

    assert service.preview_shared_catalog_snapshot_import(entry) is None
    assert not any(outside.iterdir())


def test_explicit_catalog_setup_does_not_copy_wrong_family_from_shared_root(tmp_path: Path, monkeypatch):
    shared = tmp_path / "shared-models"
    flags = RuntimeFlags(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        extra_model_dirs=[shared],
    )
    service = ModelDownloadService(flags)
    entry = service.find_catalog("flux-ae-vae")
    assert entry is not None
    filename = service._catalog_local_filename_hint(entry)
    source = shared / "Unsorted" / filename
    source.parent.mkdir(parents=True)
    source.write_bytes(b"x" * (service._catalog_min_bytes(entry) + 1))
    monkeypatch.setattr(
        "aiwf.infrastructure.model_header.read_model_info",
        lambda _path: SimpleNamespace(arch="wan-vae", role="vae", tensor_count=12),
    )

    assert service.copy_shared_catalog_asset_to_primary(entry) is None
    assert source.is_file()
    assert not service.destination_for(entry.category, filename).exists()


def test_flux_setup_finds_renamed_vae_by_specific_header_identity(tmp_path: Path, monkeypatch):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    entry = service.find_catalog("flux-ae-vae")
    assert entry is not None
    misplaced = tmp_path / "models" / "Unsorted" / "user-renamed.safetensors"
    misplaced.parent.mkdir(parents=True)
    _write_large_safetensors(misplaced, service._catalog_min_bytes(entry) + 1)
    monkeypatch.setattr(
        "aiwf.infrastructure.model_header.read_model_info",
        lambda _path: SimpleNamespace(arch="flux-vae", role="vae", tensor_count=12),
    )

    assert service.is_catalog_installed(entry, search_misplaced=True) is True
    assert service.place_misplaced_catalog_asset(entry) is True
    assert service.destination_for(entry.category, service._catalog_local_filename_hint(entry)).is_file()
    assert not misplaced.exists()


@pytest.mark.parametrize(
    ("catalog_key", "arch", "role"),
    [("flux-clip-l", "clip", "text-encoder"), ("flux-t5-fp8", "t5xxl-encoder", "text-encoder")],
)
def test_flux_setup_does_not_guess_renamed_encoder_variant(tmp_path: Path, monkeypatch, catalog_key, arch, role):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    entry = service.find_catalog(catalog_key)
    assert entry is not None
    misplaced = tmp_path / "models" / "Unsorted" / "user-renamed.safetensors"
    misplaced.parent.mkdir(parents=True)
    with misplaced.open("wb") as handle:
        handle.truncate(service._catalog_min_bytes(entry) + 1)
    monkeypatch.setattr(
        "aiwf.infrastructure.model_header.read_model_info",
        lambda _path: SimpleNamespace(arch=arch, role=role, precision="F16", tensor_count=12),
    )

    assert service.is_catalog_installed(entry, search_misplaced=True) is False
    assert service.place_misplaced_catalog_asset(entry) is False
    assert misplaced.is_file()


def test_flux_setup_rejects_fp16_data_named_as_fp8_t5(tmp_path: Path, monkeypatch):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    entry = service.find_catalog("flux-t5-fp8")
    assert entry is not None
    path = tmp_path / "models" / "Unsorted" / service._catalog_local_filename_hint(entry)
    path.parent.mkdir(parents=True)
    with path.open("wb") as handle:
        handle.truncate(service._catalog_min_bytes(entry) + 1)
    monkeypatch.setattr(
        "aiwf.infrastructure.model_header.read_model_info",
        lambda _path: SimpleNamespace(arch="t5xxl-encoder", role="text-encoder", precision="F16", tensor_count=12),
    )

    assert service.is_catalog_installed(entry, search_misplaced=True) is False
    assert service.place_misplaced_catalog_asset(entry) is False
    assert path.is_file()


def test_flux_setup_does_not_auto_place_exact_named_generic_clip_encoder(tmp_path: Path, monkeypatch):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    entry = service.find_catalog("flux-clip-l")
    assert entry is not None
    path = tmp_path / "models" / "Unsorted" / service._catalog_local_filename_hint(entry)
    path.parent.mkdir(parents=True)
    with path.open("wb") as handle:
        handle.truncate(service._catalog_min_bytes(entry) + 1)
    monkeypatch.setattr(
        "aiwf.infrastructure.model_header.read_model_info",
        lambda _path: SimpleNamespace(arch="clip", role="text-encoder", precision="F16", tensor_count=12),
    )

    assert service.is_catalog_installed(entry, search_misplaced=True) is False
    assert service.place_misplaced_catalog_asset(entry) is False
    assert path.is_file()


def test_flux_bundle_can_place_clip_l_from_typed_shared_folder(tmp_path: Path, monkeypatch):
    shared = tmp_path / "ComfyUI Models"
    service = ModelDownloadService(RuntimeFlags(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        extra_model_dirs=[shared],
    ))
    entry = service.find_catalog("flux-clip-l")
    assert entry is not None
    source = shared / "text_encoders" / "CLIP-L" / "123M fp16" / "clip_l.safetensors"
    source.parent.mkdir(parents=True)
    _write_large_safetensors(source, service._catalog_min_bytes(entry) + 1)
    monkeypatch.setattr(
        "aiwf.infrastructure.model_header.read_model_info",
        lambda _path: SimpleNamespace(arch="clip", role="text-encoder", precision="F16", tensor_count=196),
    )

    copied = service.copy_shared_catalog_asset_to_primary(entry)
    destination = service.destination_for(entry.category, service._catalog_local_filename_hint(entry))

    assert copied == {"source": str(source.resolve()), "target": str(destination.resolve())}
    assert destination.is_file()
    assert source.is_file()


def test_flux_bundle_can_import_clip_l_from_aiwf_shared_layout(tmp_path: Path, monkeypatch):
    shared = tmp_path / "AIWF shared"
    service = ModelDownloadService(RuntimeFlags(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        extra_model_dirs=[shared],
    ))
    entry = service.find_catalog("flux-clip-l")
    assert entry is not None
    source = shared / "flux" / "Textencoder" / "clip_l.safetensors"
    source.parent.mkdir(parents=True)
    _write_large_safetensors(source, service._catalog_min_bytes(entry) + 1)
    monkeypatch.setattr(
        "aiwf.infrastructure.model_header.read_model_info",
        lambda _path: SimpleNamespace(arch="clip", role="text-encoder", precision="F16", tensor_count=196),
    )

    copied = service.copy_shared_catalog_asset_to_primary(entry)
    destination = service.destination_for(entry.category, service._catalog_local_filename_hint(entry))

    assert copied == {"source": str(source.resolve()), "target": str(destination.resolve())}
    assert destination.is_file()
    assert source.is_file()


def test_flux_bundle_leaves_untyped_shared_clip_encoder_in_place(tmp_path: Path, monkeypatch):
    shared = tmp_path / "ComfyUI Models"
    service = ModelDownloadService(RuntimeFlags(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        extra_model_dirs=[shared],
    ))
    entry = service.find_catalog("flux-clip-l")
    assert entry is not None
    source = shared / "text_encoders" / "clip_l.safetensors"
    source.parent.mkdir(parents=True)
    with source.open("wb") as handle:
        handle.truncate(service._catalog_min_bytes(entry) + 1)
    monkeypatch.setattr(
        "aiwf.infrastructure.model_header.read_model_info",
        lambda _path: SimpleNamespace(arch="clip", role="text-encoder", precision="F16", tensor_count=196),
    )

    copied = service.copy_shared_catalog_asset_to_primary(entry)
    destination = service.destination_for(entry.category, service._catalog_local_filename_hint(entry))

    assert copied is None
    assert source.is_file()
    assert not destination.exists()


def test_flux_setup_leaves_ambiguous_renamed_vaes_in_place(tmp_path: Path, monkeypatch):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    entry = service.find_catalog("flux-ae-vae")
    assert entry is not None
    unsorted = tmp_path / "models" / "Unsorted"
    unsorted.mkdir(parents=True)
    files = [unsorted / f"vae-{index}.safetensors" for index in range(2)]
    for path in files:
        with path.open("wb") as handle:
            handle.truncate(service._catalog_min_bytes(entry) + 1)
    monkeypatch.setattr(
        "aiwf.infrastructure.model_header.read_model_info",
        lambda _path: SimpleNamespace(arch="flux-vae", role="vae", tensor_count=12),
    )

    assert service.is_catalog_installed(entry, search_misplaced=True) is False
    assert service.place_misplaced_catalog_asset(entry) is False
    assert all(path.is_file() for path in files)


def test_catalog_does_not_treat_empty_shared_placeholder_as_installed(tmp_path: Path):
    shared = tmp_path / "ComfyUI Models"
    flags = RuntimeFlags(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        extra_model_dirs=[shared],
    )
    service = ModelDownloadService(flags)
    entry = service.find_catalog("flux-ae-vae")
    assert entry is not None
    asset = shared / "vae" / "Flux1" / service._catalog_local_filename_hint(entry)
    asset.parent.mkdir(parents=True)
    asset.touch()

    assert service.is_catalog_installed(entry) is False


def test_ltx_checkpoint_is_discovered_in_its_shared_canonical_folder(tmp_path: Path):
    shared = tmp_path / "shared-models"
    service = ModelDownloadService(RuntimeFlags(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        extra_model_dirs=[shared],
    ))
    entry = service.find_catalog("ltx23-full-dev")
    assert entry is not None
    asset = shared / "ltx" / "checkpoints" / entry.filename
    asset.parent.mkdir(parents=True)
    _write_large_safetensors(asset, service._catalog_min_bytes(entry) + 1)
    assert service.is_catalog_installed(entry) is True


def test_ltx_checkpoint_misplaced_asset_requires_matching_header_and_can_be_placed(tmp_path: Path, monkeypatch):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path / "data", models_dir=tmp_path / "models"))
    entry = service.find_catalog("ltx23-full-dev")
    assert entry is not None
    misplaced = tmp_path / "models" / "Unsorted" / entry.filename
    misplaced.parent.mkdir(parents=True)
    _write_large_safetensors(misplaced, service._catalog_min_bytes(entry) + 1)

    monkeypatch.setattr(
        "aiwf.infrastructure.model_header.read_model_info",
        lambda _path: SimpleNamespace(arch="wrong-family", role="unknown", tensor_count=1, precision="BF16"),
    )
    assert service.is_catalog_installed(entry, search_misplaced=True) is False
    assert service.place_misplaced_catalog_asset(entry) is False
    assert misplaced.is_file()

    monkeypatch.setattr(
        "aiwf.infrastructure.model_header.read_model_info",
        lambda _path: SimpleNamespace(arch="ltx-transformer", role="unknown", tensor_count=1, precision="BF16"),
    )
    assert service.place_misplaced_catalog_asset(entry) is True
    assert not misplaced.exists()
    assert service.destination_for(entry.category, entry.filename).is_file()


def test_krea2_snapshot_catalog_installed_rejects_missing_shard(tmp_path: Path):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    entry = service.find_catalog("krea2-turbo-diffusers")
    assert entry is not None
    target = service.snapshot_destination_for(entry.category, entry.repo_id)
    shard_dir = target / "transformer"
    shard_dir.mkdir(parents=True)
    (target / "model_index.json").write_text(
        '{"transformer":["diffusers","FluxTransformer2DModel"]}', encoding="utf-8"
    )
    (shard_dir / "config.json").write_text("{}", encoding="utf-8")
    (shard_dir / "diffusion_pytorch_model.safetensors.index.json").write_text(
        '{"weight_map":{"layer.weight":"diffusion_pytorch_model-00001-of-00003.safetensors"}}',
        encoding="utf-8",
    )

    assert service.is_catalog_installed(entry) is False
    (shard_dir / "diffusion_pytorch_model-00001-of-00003.safetensors").write_bytes(b"weights")
    assert service.is_catalog_installed(entry) is True


def test_krea2_snapshot_catalog_rejects_empty_shard_placeholder(tmp_path: Path):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    entry = service.find_catalog("krea2-turbo-diffusers")
    assert entry is not None
    target = service.snapshot_destination_for(entry.category, entry.repo_id)
    transformer = target / "transformer"
    transformer.mkdir(parents=True)
    (target / "model_index.json").write_text(
        '{"transformer":["diffusers","FluxTransformer2DModel"]}', encoding="utf-8"
    )
    (transformer / "config.json").write_text("{}", encoding="utf-8")
    (transformer / "diffusion_pytorch_model.safetensors.index.json").write_text(
        '{"weight_map":{"layer.weight":"diffusion_pytorch_model-00001-of-00003.safetensors"}}',
        encoding="utf-8",
    )
    (transformer / "diffusion_pytorch_model-00001-of-00003.safetensors").touch()

    assert service.is_catalog_installed(entry) is False


@pytest.mark.parametrize(
    "category,repo_id,expected",
    [
        ("controlnet", "xinsir/controlnet-union-sdxl-1.0", ("ControlNet", "controlnet-union-sdxl-1.0")),
        ("preprocessor", "lllyasviel/Annotators", ("ControlNet", "Annotators")),
    ],
)
def test_hf_snapshot_allowed_for_controlnet_and_preprocessors(
    tmp_path: Path,
    monkeypatch,
    category,
    repo_id,
    expected,
):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    remote = service.parse_reference(
        source="huggingface",
        url_or_repo=repo_id,
        filename="",
        category=category,
    )
    assert remote.snapshot is True

    def fake_snapshot_download(*, repo_id, local_dir, token=None):
        target = Path(local_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.json").write_text("{}", encoding="utf-8")
        return str(target)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    path = service.download_parsed(remote, category=category)
    assert path == tmp_path / "models" / Path(*expected)
    assert (path / "config.json").is_file()


def test_sdxl_controlnet_catalog_entries_install_as_diffusers_folders(tmp_path: Path):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    entry = service.find_catalog("cnxl-canny")
    assert entry is not None
    assert entry.snapshot is True
    assert entry.filename == ""

    target = service.snapshot_destination_for(entry.category, entry.repo_id)
    assert target == tmp_path / "models" / "ControlNet" / "controlnet-canny-sdxl-1.0"
    target.mkdir(parents=True)
    (target / "diffusion_pytorch_model.safetensors").write_bytes(b"x")
    assert service.is_catalog_installed(entry) is False
    (target / "config.json").write_text("{}", encoding="utf-8")
    assert service.is_catalog_installed(entry) is True


def test_duplicate_catalog_filenames_use_distinct_local_names(tmp_path: Path, monkeypatch):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    monkeypatch.setattr(service, "_catalog_min_bytes", lambda entry: 1)
    sdxl = service.find_catalog("hf-lora-lcm-sdxl")
    sd15 = service.find_catalog("hf-lora-lcm-sd15")
    assert sdxl is not None
    assert sd15 is not None

    sdxl_remote = service._catalog_to_remote(sdxl)
    sd15_remote = service._catalog_to_remote(sd15)

    assert sdxl_remote.filename == "pytorch_lora_weights.safetensors"
    assert sd15_remote.filename == "pytorch_lora_weights.safetensors"
    assert sdxl_remote.local_filename == "hf-lora-lcm-sdxl-pytorch_lora_weights.safetensors"
    assert sd15_remote.local_filename == "hf-lora-lcm-sd15-pytorch_lora_weights.safetensors"
    assert sdxl_remote.local_filename != sd15_remote.local_filename

    sdxl_dest = service.destination_for(sdxl.category, sdxl_remote.local_filename)
    sdxl_dest.parent.mkdir(parents=True)
    _write_component_safetensors(sdxl_dest)
    assert service.is_catalog_installed(sdxl) is True
    assert service.is_catalog_installed(sd15) is False


def test_catalog_lists_entries(tmp_path: Path):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    keys = {item.key for item in service.list_catalog()}
    assert "hf-sd15-pruned" in keys
    assert "hf-sd35-medium" in keys
    assert "hf-sd35-large-turbo" in keys
    assert "flux-dev-q4km" in keys
    assert "flux-dev-q5km" in keys
    assert "flux-t5-q4km" not in keys  # Flux's teacher route only loads T5 safetensors.
    assert "flux-t5-fp16" in keys
    assert "flux-clip-l" in keys
    assert "flux-ae-vae" in keys
    assert "flux2-klein-4b-components" in keys
    assert "flux2-klein-base-4b-diffusers" in keys
    assert "flux2-klein-9b-components" in keys
    assert "fluxtrait-klein9b-v2-q4km" in keys
    assert "fluxtrait-zimage-v2-q4" in keys
    assert "z-image-turbo-components" in keys
    assert "krea2-turbo-diffusers" in keys
    assert "krea2-raw-diffusers" in keys
    assert "krea2-turbo-fp8-comfy" in keys
    assert "krea2-turbo-nvfp4-comfy" in keys
    assert "krea2-qwen3vl-fp8-comfy" in keys
    assert "anima-base-v1" not in keys
    assert "anima-qwen-06b-text-encoder" not in keys
    assert service.find_catalog("anima-base-v1").coming_soon is True
    assert service.find_catalog("anima-qwen-06b-text-encoder").coming_soon is True
    assert "ltx23-distilled" in keys
    assert "ltx23-upscaler-x2" in keys
    assert "ltx23-gemma-q4" in keys
    assert "cn15-v11-full-suite" in keys
    assert "cn15-canny" in keys          # renamed from cn-canny-light
    assert "civit-dreamshaper-8" in keys
    assert "cnxl-union-full" in keys
    assert "cnxl-canny" in keys          # SDXL ControlNet
    assert "emb-easynegative" in keys    # embeddings
    assert "pre-annotators-full-suite" in keys
    assert "pre-dwpose" in keys          # preprocessors
    assert "gdino-swinb" in keys         # GroundingDINO


def test_default_sd15_install_seed_uses_small_fp16_single_file(tmp_path: Path):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    entry = service.find_catalog(DEFAULT_SD15_CATALOG_KEY)

    assert entry is not None
    assert entry.repo_id == "Comfy-Org/stable-diffusion-v1-5-archive"
    assert entry.filename == "v1-5-pruned-emaonly-fp16.safetensors"
    assert entry.category == "checkpoint"
    assert entry.size_mb is not None and entry.size_mb <= 2100


def test_flux_quick_start_uses_supported_runtime_assets(tmp_path: Path):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))

    assert QUICK_START_BUNDLES["flux"] == [
        "flux-dev-q4km", "flux-t5-fp8", "flux-clip-l", "flux-ae-vae",
        "flux-clip-tokenizer", "flux-t5-tokenizer",
    ]
    entries = [service.find_catalog(key) for key in QUICK_START_BUNDLES["flux"]]
    assert all(entry is not None for entry in entries)
    assert [entry.source for entry in entries if entry is not None] == [
        "huggingface",
        "huggingface",
        "huggingface",
        "huggingface",
        "huggingface",
        "huggingface",
    ]
    assert [entry.category for entry in entries if entry is not None] == [
        "flux_unet_gguf",
        "flux_text_encoder",
        "flux_text_encoder",
        "flux_vae",
        "flux_tokenizer",
        "flux_tokenizer",
    ]

    vae = service.find_catalog("flux-ae-vae")
    assert vae is not None
    remote = service._catalog_to_remote(vae)
    assert remote.filename == "ae.safetensors"
    assert remote.repo_filename == "split_files/vae/ae.safetensors"


def test_flux2_z_image_qwen_and_sana_quick_start_use_runtime_assets(tmp_path: Path):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))

    assert QUICK_START_BUNDLES["flux2"] == ["flux2-klein-4b-diffusers"]
    flux2_entries = [service.find_catalog(key) for key in QUICK_START_BUNDLES["flux2"]]
    assert all(entry is not None for entry in flux2_entries)
    assert [entry.category for entry in flux2_entries if entry is not None] == [
        "flux2_diffusers",
    ]
    full_flux2 = service.find_catalog("flux2-klein-4b-diffusers")
    flux2_components = service.find_catalog("flux2-klein-4b-components")
    zimage_components = service.find_catalog("z-image-turbo-components")
    assert full_flux2 is not None
    expected_full_patterns = (
        "model_index.json", "scheduler/**", "text_encoder/**", "tokenizer/**",
        "transformer/**", "vae/**",
    )
    assert full_flux2.snapshot_allow_patterns == expected_full_patterns
    expected_component_patterns = (
        "model_index.json", "scheduler/**", "text_encoder/**", "tokenizer/**",
        "vae/**", "transformer/config.json",
    )
    assert flux2_components is not None
    assert flux2_components.snapshot_allow_patterns == expected_component_patterns
    assert zimage_components is not None
    assert zimage_components.snapshot_allow_patterns == expected_component_patterns
    assert service.snapshot_destination_for("flux2_diffusers", "black-forest-labs/FLUX.2-klein-4B") == (
        tmp_path / "models" / "flux2" / "Diffusers" / "FLUX.2-klein-4B"
    )

    assert QUICK_START_BUNDLES["zimage"] == ["fluxtrait-zimage-v2-q4", "z-image-turbo-components"]
    z_entries = [service.find_catalog(key) for key in QUICK_START_BUNDLES["zimage"]]
    assert all(entry is not None for entry in z_entries)
    assert [entry.category for entry in z_entries if entry is not None] == [
        "z_image_unet_gguf",
        "z_image_components",
    ]
    assert service.destination_for("z_image_unet_gguf", "fluxtraitFLUX2KleinFLUXZ_zImageV2GgufQ4.gguf") == (
        tmp_path / "models" / "z-image" / "GGUF" / "fluxtraitFLUX2KleinFLUXZ_zImageV2GgufQ4.gguf"
    )
    assert service.snapshot_destination_for("z_image_components", "Tongyi-MAI/Z-Image-Turbo") == (
        tmp_path / "models" / "z-image" / "Components" / "Z-Image-Turbo"
    )
    from aiwf.services.model_download_catalog import quick_start_bundles_for_platform
    windows_zimage = [service.find_catalog(key) for key in quick_start_bundles_for_platform(windows=True)["zimage"]]
    assert [entry.key for entry in windows_zimage if entry is not None] == ["fluxtrait-zimage-v2-bf16", "z-image-turbo-components"]


@pytest.mark.parametrize("key", ["flux2-klein-4b-components", "z-image-turbo-components"])
def test_component_snapshot_catalog_readiness_rejects_partial_tree(tmp_path: Path, key: str):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    entry = service.find_catalog(key)
    assert entry is not None
    target = service.snapshot_destination_for(entry.category, entry.repo_id)
    target.mkdir(parents=True)
    (target / "model_index.json").write_text("{}", encoding="utf-8")
    assert service._catalog_snapshot_ready(entry, target) is False
    for name in ("scheduler", "text_encoder", "tokenizer", "vae"):
        (target / name).mkdir()
    (target / "scheduler" / "scheduler_config.json").write_text("{}", encoding="utf-8")
    (target / "text_encoder" / "model.safetensors").write_bytes(b"weights")
    (target / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    (target / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")
    assert service._catalog_snapshot_ready(entry, target) is False


@pytest.mark.parametrize(
    ("key", "family"),
    [("flux2-klein-4b-components", "flux2"), ("z-image-turbo-components", "z_image")],
)
def test_component_snapshot_catalog_accepts_valid_support_contract(tmp_path: Path, key: str, family: str):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    entry = service.find_catalog(key)
    assert entry is not None
    target = service.snapshot_destination_for(entry.category, entry.repo_id)

    _write_support_components(target, family=family)

    assert service._catalog_snapshot_ready(entry, target) is True
    tokenizer = target / "tokenizer"
    (tokenizer / "tokenizer.json").unlink()
    (tokenizer / "vocab.json").write_text('{"x": 0}', encoding="utf-8")
    assert service._catalog_snapshot_ready(entry, target) is False
    (tokenizer / "merges.txt").write_text("#version: 0.2\nx x</w>", encoding="utf-8")
    assert service._catalog_snapshot_ready(entry, target) is True


def test_flux2_full_pipeline_readiness_requires_transformer_config_and_weights(tmp_path: Path):
    from aiwf.infrastructure.diffusers.checkpoints import flux2_klein_missing_local_files

    target = tmp_path / "models" / "flux2" / "Diffusers" / "FLUX.2-klein-4B"
    _write_support_components(target, family="flux2")
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    entry = service.find_catalog("flux2-klein-4b-diffusers")
    assert entry is not None
    assert service._catalog_snapshot_ready(entry, target) is False

    missing = flux2_klein_missing_local_files(target)
    assert target / "transformer" / "config.json" in missing
    assert target / "transformer" / "diffusion_pytorch_model.safetensors" in missing

    (target / "transformer").mkdir(parents=True)
    (target / "transformer" / "config.json").write_text(
        json.dumps({"_class_name": "Flux2Transformer2DModel", "in_channels": 128, "num_layers": 5}),
        encoding="utf-8",
    )
    _write_component_safetensors(target / "transformer" / "diffusion_pytorch_model.safetensors")
    assert flux2_klein_missing_local_files(target) == []

    assert service._catalog_snapshot_ready(entry, target) is True
    (target / "transformer" / "diffusion_pytorch_model.safetensors").unlink()
    assert service._catalog_snapshot_ready(entry, target) is False


def test_ltx_gemma_catalog_readiness_requires_runtime_assets(tmp_path: Path):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    entry = service.find_catalog("ltx23-gemma-q4")
    assert entry is not None
    target = service.snapshot_destination_for(entry.category, entry.repo_id)
    target.mkdir(parents=True)
    (target / "config.json").write_text("{}", encoding="utf-8")
    assert service._catalog_snapshot_ready(entry, target) is False

    assert QUICK_START_BUNDLES["qwen-image"] == ["qwen-image-2512-diffusers"]
    qwen = service.find_catalog("qwen-image-2512-diffusers")
    assert qwen is not None
    assert qwen.category == "qwen_image_diffusers"
    assert service.snapshot_destination_for(qwen.category, qwen.repo_id) == (
        tmp_path / "models" / "qwen-image" / "Diffusers" / "Qwen-Image-2512"
    )
    assert QUICK_START_BUNDLES["qwen-nunchaku"] == [
        "qwen-image-diffusers", "qwen-nunchaku-image-lightning-int4-r32",
    ]
    qwen_nunchaku = service.find_catalog("qwen-nunchaku-image-lightning-int4-r32")
    assert qwen_nunchaku is not None
    assert qwen_nunchaku.category == "qwen_image_nunchaku"
    assert qwen_nunchaku.coming_soon is False
    assert service.destination_for(qwen_nunchaku.category, qwen_nunchaku.filename) == (
        tmp_path
        / "models"
        / "qwen-image"
        / "Nunchaku"
        / "svdq-int4_r32-qwen-image-lightningv1.0-4steps.safetensors"
    )

    assert QUICK_START_BUNDLES["sana"] == ["sana-sprint-06b-diffusers"]
    sana = service.find_catalog("sana-sprint-06b-diffusers")
    assert sana is not None
    assert sana.category == "sana_diffusers"
    assert service.snapshot_destination_for(sana.category, sana.repo_id) == (
        tmp_path / "models" / "sana" / "Diffusers" / "Sana_Sprint_0.6B_1024px_diffusers"
    )
    assert QUICK_START_BUNDLES["sana-video"] == ["sana-video-2b-480p-diffusers"]
    sana_video = service.find_catalog("sana-video-2b-480p-diffusers")
    assert sana_video is not None
    assert sana_video.category == "sana_video_diffusers"
    assert service.snapshot_destination_for(sana_video.category, sana_video.repo_id) == (
        tmp_path / "models" / "sana-video" / "Diffusers" / "SANA-Video_2B_480p_diffusers"
    )

    assert QUICK_START_BUNDLES["krea2"] == ["krea2-turbo-diffusers"]
    assert QUICK_START_BUNDLES["krea2-low"][0] == "krea2-turbo-nvfp4-comfy"
    assert QUICK_START_BUNDLES["krea2-mid"][0] == "krea2-turbo-fp8-comfy"
    assert QUICK_START_BUNDLES["krea2-high"][0] == "krea2-turbo-bf16-comfy"
    krea2 = service.find_catalog("krea2-turbo-diffusers")
    assert krea2 is not None
    assert krea2.category == "krea2_diffusers"
    assert krea2.snapshot is True
    assert service.snapshot_destination_for(krea2.category, krea2.repo_id) == (
        tmp_path / "models" / "krea2" / "Diffusers" / "Krea-2-Turbo"
    )

    assert "anima" not in QUICK_START_BUNDLES
    assert "anima-low" not in QUICK_START_BUNDLES
    assert "anima-mid" not in QUICK_START_BUNDLES
    assert "anima-high" not in QUICK_START_BUNDLES
    anima = service.find_catalog("anima-base-v1")
    assert anima is not None
    assert anima.category == "anima_unet_safetensor"
    assert anima.coming_soon is True


def test_ltx_quick_start_uses_ltx_categories(tmp_path: Path):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))

    assert QUICK_START_BUNDLES["ltx23"] == ["ltx23-distilled", "ltx23-upscaler-x2", "ltx23-gemma-q4"]
    entries = [service.find_catalog(key) for key in QUICK_START_BUNDLES["ltx23"]]
    assert all(entry is not None for entry in entries)
    assert [entry.category for entry in entries if entry is not None] == [
        "ltx_checkpoint",
        "ltx_upscaler",
        "ltx_text_encoder",
    ]
    gemma = service.find_catalog("ltx23-gemma-q4")
    assert gemma is not None
    assert gemma.snapshot is True
    assert service.snapshot_destination_for(gemma.category, gemma.repo_id) == (
        tmp_path / "models" / "ltx" / "text_encoder" / "gemma-3-12b-it-qat-q4_0-unquantized"
    )
    assert QUICK_START_BUNDLES["ltx23-one-stage"] == ["ltx23-full-dev", "ltx23-gemma-q4"]
    full_dev = service.find_catalog("ltx23-full-dev")
    assert full_dev is not None
    assert full_dev.filename == "ltx-2.3-22b-dev-bf16.safetensors"
    assert QUICK_START_BUNDLES["ltx23-one-stage-fp8"] == ["ltx23-full-dev-fp8", "ltx23-gemma-q4"]
    full_dev_fp8 = service.find_catalog("ltx23-full-dev-fp8")
    assert full_dev_fp8 is not None
    assert full_dev_fp8.repo_id == "Lightricks/LTX-2.3-fp8"
    assert full_dev_fp8.filename == "ltx-2.3-22b-dev-fp8.safetensors"
    assert full_dev_fp8.size_mb == 29100
    assert "ltx-2-community-license-agreement" in full_dev_fp8.notes
    assert service.destination_for(full_dev_fp8.category, full_dev_fp8.filename) == (
        tmp_path / "models" / "ltx" / "checkpoints" / "ltx-2.3-22b-dev-fp8.safetensors"
    )
    ltx_2b_entries = [service.find_catalog(key) for key in QUICK_START_BUNDLES["ltx-2b"]]
    ltx_tokenizer = service.find_catalog("ltx-t5-tokenizer")
    assert ltx_tokenizer is not None
    assert ltx_tokenizer.snapshot is True
    assert ltx_tokenizer.repo_id == "google/t5-v1_1-xxl"
    assert service.snapshot_destination_for(ltx_tokenizer.category, ltx_tokenizer.repo_id) == (
        tmp_path / "models" / "ltx" / "tokenizer" / "t5-v1_1-xxl"
    )
    assert all(entry is not None for entry in ltx_2b_entries)
    ltx_t5 = service.find_catalog("flux-t5-fp16")
    assert ltx_t5 is not None
    assert "Flux / LTX" in ltx_t5.title
    assert "LTX 2B requires this FP16 file" in ltx_t5.notes
    ltx_2b = service.find_catalog("ltx-2b-v095")
    assert ltx_2b is not None
    assert ltx_2b.filename == "ltx-video-2b-v0.9.5.safetensors"


def test_fluxtrait_civitai_variants_stay_in_flux_categories(tmp_path: Path):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))

    fp8 = service.find_catalog("fluxtrait-v10-fp8")
    q4 = service.find_catalog("fluxtrait-v20-q4km")
    q5 = service.find_catalog("fluxtrait-v20-q5km")
    fusion_q4 = service.find_catalog("flux-fusion-v2-q4km")

    assert fp8 is not None
    assert q4 is not None
    assert q5 is not None
    assert fusion_q4 is not None
    assert fp8.category == "flux_unet_safetensor"
    assert q4.category == "flux_unet_gguf"
    assert q5.category == "flux_unet_gguf"
    assert fusion_q4.category == "flux_unet_gguf"
    assert service.destination_for(fp8.category, fp8.filename) == (
        tmp_path / "models" / "flux" / "UNet" / "fluxtraitFLUX2KleinFLUXZ_v10FP8.safetensors"
    )
    assert service.destination_for(q4.category, q4.filename) == (
        tmp_path / "models" / "flux" / "GGUF" / "fluxtraitFLUX2KleinFLUXZ_v20Q4KM.gguf"
    )
    assert service.destination_for(fusion_q4.category, fusion_q4.filename) == (
        tmp_path / "models" / "flux" / "GGUF" / "fluxFusionV24StepsGGUFNF4_V2GGUFQ4KM.gguf"
    )


def test_incomplete_catalog_file_is_replaced_before_retry(tmp_path: Path, monkeypatch):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    entry = CatalogEntry(
        key="flux-test-q4",
        title="Flux test",
        category="flux_unet_gguf",
        source="direct",
        url="https://example.com/flux-test.gguf",
        size_mb=2,
    )
    monkeypatch.setattr(model_download_module, "MODEL_DOWNLOAD_CATALOG", [entry])

    stale = service.destination_for("flux_unet_gguf", "flux-test.gguf")
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"not a model")

    def fake_stream_download(url, dest, *, on_progress=None, headers=None, chunk_size=1024 * 256):
        assert url == "https://example.com/flux-test.gguf"
        assert not dest.exists()
        _write_minimal_gguf(dest, pad_to_bytes=2 * 1024 * 1024)
        if on_progress:
            on_progress(dest.stat().st_size, dest.stat().st_size)
        return dest

    monkeypatch.setattr(model_download_module, "stream_download", fake_stream_download)

    path = service.download_catalog("flux-test-q4")

    assert path == stale
    assert path.stat().st_size == 2 * 1024 * 1024
    assert list(stale.parent.glob("flux-test.gguf.incomplete-*.bad"))
    assert service._catalog_file_ready(entry, path)


def test_catalog_file_readiness_rejects_large_truncated_gguf(tmp_path: Path):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    entry = CatalogEntry(
        key="flux-test-q4",
        title="Flux test",
        category="flux_unet_gguf",
        source="direct",
        url="https://example.com/flux-test.gguf",
        filename="flux-test.gguf",
        size_mb=1,
    )
    malformed = service.destination_for(entry.category, entry.filename)
    _write_minimal_gguf(malformed, tensor_elements=512 * 1024)
    size_before_truncation = malformed.stat().st_size
    with malformed.open("r+b") as stream:
        stream.truncate(size_before_truncation - 512 * 1024)

    assert malformed.stat().st_size >= service._catalog_min_bytes(entry)
    assert not service._catalog_file_ready(entry, malformed)
    assert not service.is_catalog_installed(entry)


def test_catalog_file_readiness_accepts_valid_minimal_gguf(tmp_path: Path):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    entry = CatalogEntry(
        key="flux-test-q4",
        title="Flux test",
        category="flux_unet_gguf",
        source="direct",
        url="https://example.com/flux-test.gguf",
        filename="flux-test.gguf",
        size_mb=1,
    )
    valid = service.destination_for(entry.category, entry.filename)
    _write_minimal_gguf(valid, pad_to_bytes=2 * 1024 * 1024)

    assert valid.stat().st_size >= service._catalog_min_bytes(entry)
    assert service._catalog_file_ready(entry, valid)
    assert service.is_catalog_installed(entry)


def test_catalog_file_without_size_metadata_rejects_empty_placeholder(tmp_path: Path):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    entry = CatalogEntry(
        key="ltx-test",
        title="LTX test asset",
        category="ltx",
        source="direct",
        url="https://example.com/ltx-test.safetensors",
        filename="ltx-test.safetensors",
    )
    empty = tmp_path / "ltx-test.safetensors"
    empty.write_bytes(b"")

    assert service._catalog_min_bytes(entry) == 0
    assert not service._catalog_file_ready(entry, empty)

    _write_component_safetensors(empty)
    assert service._catalog_file_ready(entry, empty)


def test_catalog_file_readiness_rejects_large_truncated_safetensors(tmp_path: Path):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    entry = CatalogEntry(
        key="ltx-test",
        title="LTX test asset",
        category="ltx",
        source="direct",
        url="https://example.com/ltx-test.safetensors",
        filename="ltx-test.safetensors",
        size_mb=1,
    )
    malformed = tmp_path / "ltx-test.safetensors"
    malformed.write_bytes(b"x" * (2 * 1024 * 1024))

    assert malformed.stat().st_size >= service._catalog_min_bytes(entry)
    assert not service._catalog_file_ready(entry, malformed)


def test_direct_private_download_url_blocked(tmp_path: Path):
    service = ModelDownloadService(
        RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", block_private_download_urls=True)
    )
    remote = ParsedRemote(source="direct", url="http://127.0.0.1/model.safetensors", filename="model.safetensors")

    with pytest.raises(ValueError, match="Private"):
        service.download_parsed(remote, category="checkpoint")


def test_civitai_download_sets_user_agent(tmp_path: Path, monkeypatch):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    remote = ParsedRemote(
        source="civitai",
        url="https://civitai.com/api/download/models/123",
        filename="flux.gguf",
    )

    def fake_stream_download(url, dest, *, on_progress=None, headers=None, chunk_size=1024 * 256):
        assert url == remote.url
        assert headers is not None
        assert headers["User-Agent"] == "AIWF-Studio/1.0"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x")
        return dest

    monkeypatch.setattr(model_download_module, "stream_download", fake_stream_download)

    path = service.download_parsed(remote, category="flux_unet_gguf")

    assert path.name == "flux.gguf"


def test_parse_civitai_model_url(monkeypatch):
    def fake_json(path, *, token=None):
        assert path == "/models/4384"
        return {
            "modelVersions": [
                {
                    "id": 99,
                    "files": [
                        {
                            "primary": True,
                            "name": "dreamshaper.safetensors",
                            "downloadUrl": "https://civitai.com/api/download/models/99",
                        }
                    ],
                }
            ],
        }

    monkeypatch.setattr("aiwf.services.model_download._fetch_civitai_json", fake_json)
    remote = _parse_civitai_reference("https://civitai.com/models/4384/dreamshaper")
    assert remote.filename == "dreamshaper.safetensors"
    assert remote.civitai_model_id == 4384
