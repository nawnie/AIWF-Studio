from __future__ import annotations

from pathlib import Path

import pytest

from aiwf.core.config.settings import RuntimeFlags
from aiwf.services.model_download import (
    ModelDownloadService,
    ParsedRemote,
    _parse_civitai_reference,
    _parse_hf_reference,
    detect_source,
    inspect_custom_input,
    split_hf_url,
)


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
    assert remote.repo_id == "stabilityai/sd-vae-ft-mse-original"


def test_parse_hf_repo_requires_filename():
    with pytest.raises(ValueError):
        _parse_hf_reference("runwayml/stable-diffusion-v1-5", "")


def test_split_hf_tree_url():
    url = "https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main/v1-5-pruned-emaonly.safetensors"
    repo, filename = split_hf_url(url)
    assert repo == "runwayml/stable-diffusion-v1-5"
    assert filename == "v1-5-pruned-emaonly.safetensors"


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


def test_catalog_lists_entries(tmp_path: Path):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    keys = {item.key for item in service.list_catalog()}
    assert "hf-sd15-pruned" in keys
    assert "hf-sd35-medium" in keys
    assert "hf-sd35-large-turbo" in keys
    assert "cn15-v11-full-suite" in keys
    assert "cn15-canny" in keys          # renamed from cn-canny-light
    assert "civit-dreamshaper-8" in keys
    assert "cnxl-union-full" in keys
    assert "cnxl-canny" in keys          # SDXL ControlNet
    assert "emb-easynegative" in keys    # embeddings
    assert "pre-annotators-full-suite" in keys
    assert "pre-dwpose" in keys          # preprocessors
    assert "gdino-swinb" in keys         # GroundingDINO


def test_direct_private_download_url_blocked(tmp_path: Path):
    service = ModelDownloadService(
        RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", block_private_download_urls=True)
    )
    remote = ParsedRemote(source="direct", url="http://127.0.0.1/model.safetensors", filename="model.safetensors")

    with pytest.raises(ValueError, match="Private"):
        service.download_parsed(remote, category="checkpoint")


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
