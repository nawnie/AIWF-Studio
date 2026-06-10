from __future__ import annotations

from pathlib import Path

import pytest

from aiwf.core.config.settings import RuntimeFlags
from aiwf.services.model_download import (
    ModelDownloadService,
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
    assert service.destination_dir("lora").name == "Lora"
    assert service.destination_dir("controlnet").name == "ControlNet"
    assert service.destination_dir("upscaler").name == "RealESRGAN"


def test_catalog_lists_entries(tmp_path: Path):
    service = ModelDownloadService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    keys = {item.key for item in service.list_catalog()}
    assert "hf-sd15-pruned" in keys
    assert "cn-canny-light" in keys
    assert "civit-dreamshaper-8" in keys


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