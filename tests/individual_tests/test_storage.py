from pathlib import Path
import json

from PIL import Image

from aiwf import __version__
from aiwf.core.config.settings import UserSettings
from aiwf.infrastructure.storage.filesystem import (
    FilesystemImageStore,
    make_grid,
    render_filename,
)
from aiwf.services.metadata import MetadataService


def test_store_save_preserves_png_metadata(tmp_path: Path):
    store = FilesystemImageStore(tmp_path)
    metadata = MetadataService()

    image = Image.new("RGB", (16, 16), color="blue")
    infotext = "Steps: 20, Tags: #portrait #wip"
    embedded = metadata.embed(image, infotext, tags=["portrait", "wip"])

    artifact = store.save(embedded, infotext, "txt2img-images")

    with Image.open(artifact.path) as saved:
        assert saved.text.get("parameters") == infotext
        assert json.loads(saved.text["aiwf"])["version"] == __version__
        assert "portrait" in metadata.read_tags(saved)
        assert "wip" in metadata.read_tags(saved)


def test_metadata_enrich_infotext_appends_selected_fields():
    metadata = MetadataService()

    text = metadata.enrich_infotext(
        "Prompt\nSteps: 20, Model: base",
        model_hash="abc123",
        vae_name="vae-ft-mse",
        vae_hash="def456",
        lora_hashes={"detail": "987xyz"},
        app_version="0.1.0",
    )

    assert "Model hash: abc123" in text
    assert "VAE: vae-ft-mse" in text
    assert "VAE hash: def456" in text
    assert "Lora hashes: detail: 987xyz" in text
    assert "AIWF Studio: 0.1.0" in text


def test_render_filename_tokens():
    name = render_filename("[seed]-[model_name]-[seq]", seed=1234, index=2, model_name="Real/Vis:XL")
    assert name == "1234-Real_Vis_XL-2"


def test_render_filename_default_is_timestamp():
    name = render_filename("[datetime]")
    assert len(name) == 15 and name[8] == "-"  # YYYYMMDD-HHMMSS


def test_save_never_overwrites_batch(tmp_path: Path):
    settings = UserSettings(filename_pattern="fixed")
    store = FilesystemImageStore(tmp_path, settings=settings)
    img = Image.new("RGB", (8, 8), "red")
    a = store.save(img, "", "txt2img-images", index=0)
    b = store.save(img, "", "txt2img-images", index=1)
    assert a.path != b.path
    assert Path(a.path).exists() and Path(b.path).exists()


def test_sidecar_txt_written_when_enabled(tmp_path: Path):
    settings = UserSettings(save_sidecar_txt=True)
    store = FilesystemImageStore(tmp_path, settings=settings)
    info = "Steps: 20, Seed: 42"
    artifact = store.save(Image.new("RGB", (8, 8), "green"), info, "txt2img-images", seed=42)
    sidecar = Path(artifact.path).with_suffix(".txt")
    assert sidecar.exists()
    assert sidecar.read_text(encoding="utf-8") == info


def test_no_sidecar_by_default(tmp_path: Path):
    store = FilesystemImageStore(tmp_path, settings=UserSettings())
    artifact = store.save(Image.new("RGB", (8, 8), "green"), "info", "txt2img-images")
    assert not Path(artifact.path).with_suffix(".txt").exists()


def test_make_grid_and_save_grid(tmp_path: Path):
    assert make_grid([Image.new("RGB", (8, 8), "red")]) is None
    imgs = [Image.new("RGB", (8, 8), c) for c in ("red", "blue", "green")]
    grid = make_grid(imgs)
    assert grid is not None and grid.width >= 16 and grid.height >= 16

    store = FilesystemImageStore(tmp_path, settings=UserSettings())
    artifact = store.save_grid(imgs, "txt2img-images")
    assert artifact is not None and Path(artifact.path).name.startswith("grid-")
    assert store.save_grid(imgs[:1], "txt2img-images") is None
