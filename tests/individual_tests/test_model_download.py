from __future__ import annotations

from pathlib import Path

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
    assert service.destination_dir("z_image_unet_safetensor") == tmp_path / "models" / "z-image" / "UNet"
    assert service.destination_dir("z_image_unet_gguf") == tmp_path / "models" / "z-image" / "GGUF"
    assert service.destination_dir("z_image_components") == tmp_path / "models" / "z-image" / "Components"
    assert service.destination_dir("ltx_checkpoint") == tmp_path / "models" / "ltx" / "checkpoints"
    assert service.destination_dir("ltx_upscaler") == tmp_path / "models" / "ltx" / "upscalers"
    assert service.destination_dir("ltx_lora") == tmp_path / "models" / "ltx" / "loras"
    assert service.destination_dir("ltx_text_encoder") == tmp_path / "models" / "ltx" / "text_encoder"


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
    assert (tmp_path / "models" / "ltx" / "checkpoints").is_dir()
    assert (tmp_path / "models" / "ltx" / "upscalers").is_dir()
    assert (tmp_path / "models" / "ltx" / "text_encoder").is_dir()


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


def test_ltx_download_categories_validate_file_type(tmp_path: Path):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    safetensors = ParsedRemote(source="direct", url="https://example.com/ltx.safetensors", filename="ltx.safetensors")
    gguf = ParsedRemote(source="direct", url="https://example.com/ltx.gguf", filename="ltx.gguf")

    assert service.destination_for("ltx_checkpoint", safetensors.filename) == (
        tmp_path / "models" / "ltx" / "checkpoints" / "ltx.safetensors"
    )
    assert service.destination_for("ltx_upscaler", safetensors.filename) == (
        tmp_path / "models" / "ltx" / "upscalers" / "ltx.safetensors"
    )
    assert service.destination_for("ltx_lora", safetensors.filename) == (
        tmp_path / "models" / "ltx" / "loras" / "ltx.safetensors"
    )
    with pytest.raises(ValueError, match="LTX 2.3 checkpoint"):
        service.download_parsed(gguf, category="ltx_checkpoint")


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


@pytest.mark.parametrize(
    "category,repo_id,expected",
    [
        ("flux2_components", "black-forest-labs/FLUX.2-klein-4B", ("flux2", "Components", "FLUX.2-klein-4B")),
        ("z_image_components", "Tongyi-MAI/Z-Image-Turbo", ("z-image", "Components", "Z-Image-Turbo")),
    ],
)
def test_hf_snapshot_allowed_for_flux2_and_z_image_components(
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
    assert service.is_catalog_installed(entry) is True


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
    sdxl_dest.write_bytes(b"x")
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
    assert "flux-t5-q4km" in keys
    assert "flux-t5-fp16" in keys
    assert "flux-clip-l" in keys
    assert "flux-ae-vae" in keys
    assert "flux2-klein-4b-components" in keys
    assert "flux2-klein-9b-components" in keys
    assert "fluxtrait-klein9b-v2-q4km" in keys
    assert "fluxtrait-zimage-v2-q4" in keys
    assert "z-image-turbo-components" in keys
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


def test_flux_quick_start_uses_supported_runtime_assets(tmp_path: Path):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))

    assert QUICK_START_BUNDLES["flux"] == ["flux-fusion-v2-q4km", "flux-t5-fp16", "flux-clip-l", "flux-ae-vae"]
    entries = [service.find_catalog(key) for key in QUICK_START_BUNDLES["flux"]]
    assert all(entry is not None for entry in entries)
    assert [entry.source for entry in entries if entry is not None] == [
        "civitai",
        "huggingface",
        "huggingface",
        "huggingface",
    ]
    assert [entry.category for entry in entries if entry is not None] == [
        "flux_unet_gguf",
        "flux_text_encoder",
        "flux_text_encoder",
        "flux_vae",
    ]

    vae = service.find_catalog("flux-ae-vae")
    assert vae is not None
    remote = service._catalog_to_remote(vae)
    assert remote.filename == "ae.safetensors"
    assert remote.repo_filename == "split_files/vae/ae.safetensors"


def test_flux2_and_z_image_quick_start_use_separate_runtime_assets(tmp_path: Path):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))

    assert QUICK_START_BUNDLES["flux2"] == ["fluxtrait-klein9b-v2-q4km", "flux2-klein-9b-components"]
    flux2_entries = [service.find_catalog(key) for key in QUICK_START_BUNDLES["flux2"]]
    assert all(entry is not None for entry in flux2_entries)
    assert [entry.category for entry in flux2_entries if entry is not None] == [
        "flux2_unet_gguf",
        "flux2_components",
    ]
    assert service.destination_for("flux2_unet_gguf", "fluxtraitFLUX2KleinFLUXZ_klein9bV2Q4KM.gguf") == (
        tmp_path / "models" / "flux2" / "GGUF" / "fluxtraitFLUX2KleinFLUXZ_klein9bV2Q4KM.gguf"
    )
    assert service.snapshot_destination_for("flux2_components", "black-forest-labs/FLUX.2-klein-9B") == (
        tmp_path / "models" / "flux2" / "Components" / "FLUX.2-klein-9B"
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
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x" * (2 * 1024 * 1024))
        if on_progress:
            on_progress(dest.stat().st_size, dest.stat().st_size)
        return dest

    monkeypatch.setattr(model_download_module, "stream_download", fake_stream_download)

    path = service.download_catalog("flux-test-q4")

    assert path == stale
    assert path.stat().st_size == 2 * 1024 * 1024
    assert list(stale.parent.glob("flux-test.gguf.incomplete-*.bad"))


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
