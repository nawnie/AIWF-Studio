from io import BytesIO
from pathlib import Path

from PIL import Image, PngImagePlugin

from aiwf.infrastructure.storage.filesystem import FilesystemImageStore
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
        assert "portrait" in metadata.read_tags(saved)
        assert "wip" in metadata.read_tags(saved)