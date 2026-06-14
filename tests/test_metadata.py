from pathlib import Path

from PIL import Image

from aiwf.core.config.settings import UserSettings
from aiwf.services.metadata import MetadataService


def test_metadata_settings_defaults():
    s = UserSettings()
    assert s.metadata_include_model_hash is True
    assert s.metadata_include_vae_hash is True
    assert s.metadata_include_lora_hashes is True
    assert s.metadata_include_app_version is True
    assert s.pnginfo_send_to_studio is True
    assert s.pnginfo_clear_after_apply is True


def test_metadata_settings_round_trip():
    s = UserSettings(
        metadata_include_model_hash=False,
        metadata_include_vae_hash=False,
        metadata_include_lora_hashes=False,
        metadata_include_app_version=False,
        pnginfo_send_to_studio=False,
        pnginfo_clear_after_apply=False,
    )
    r = UserSettings(**s.model_dump())
    assert r.metadata_include_model_hash is False
    assert r.pnginfo_send_to_studio is False
    assert r.pnginfo_clear_after_apply is False


def test_enrich_infotext_adds_requested_fields():
    md = MetadataService()
    base = "prompt\nSteps: 20, Seed: 1"
    out = md.enrich_infotext(
        base,
        model_hash="abc123",
        vae_name="kl-f8",
        vae_hash="def456",
        lora_hashes={"detail": "11aa22bb", "empty": ""},
        app_version="9.9.9",
    )
    assert "Model hash: abc123" in out
    assert "VAE: kl-f8" in out
    assert "VAE hash: def456" in out
    assert "Lora hashes: detail: 11aa22bb" in out
    assert "empty:" not in out  # blank hashes skipped
    assert "AIWF Studio: 9.9.9" in out
    assert out.startswith(base.rstrip())


def test_enrich_infotext_noop_when_nothing_requested():
    md = MetadataService()
    base = "prompt\nSteps: 20"
    assert md.enrich_infotext(base) == base


def test_file_fingerprint_stable_and_missing(tmp_path: Path):
    md = MetadataService()
    f = tmp_path / "model.safetensors"
    f.write_bytes(b"hello world" * 100)
    fp1 = md.file_fingerprint(f)
    fp2 = md.file_fingerprint(f)
    assert fp1 is not None and len(fp1) == 10 and fp1 == fp2
    assert md.file_fingerprint(tmp_path / "missing.safetensors") is None


def test_embed_and_read_infotext_round_trip():
    md = MetadataService()
    info = "a cat\nSteps: 20, Seed: 7"
    embedded = md.embed(Image.new("RGB", (8, 8)), info, tags=["x"])
    assert md.read_infotext(embedded) == info
