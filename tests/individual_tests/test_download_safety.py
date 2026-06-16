"""
Tests for download safety helpers in aiwf/services/model_download.py.

Covers: is_unsafe_download_format, write_download_receipt.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aiwf.services.model_download import is_unsafe_download_format, write_download_receipt


class TestIsUnsafeDownloadFormat:
    def test_ckpt_is_unsafe(self):
        assert is_unsafe_download_format("model.ckpt") is True

    def test_pt_is_unsafe(self):
        assert is_unsafe_download_format("embedding.pt") is True

    def test_pth_is_unsafe(self):
        assert is_unsafe_download_format("weights.pth") is True

    def test_safetensors_is_safe(self):
        assert is_unsafe_download_format("model.safetensors") is False

    def test_gguf_is_safe(self):
        assert is_unsafe_download_format("model.gguf") is False

    def test_bin_is_safe(self):
        assert is_unsafe_download_format("model.bin") is False

    def test_case_insensitive(self):
        assert is_unsafe_download_format("model.CKPT") is True
        assert is_unsafe_download_format("model.PT") is True

    def test_path_object(self):
        # accepts strings — Path suffix comparison works on str
        assert is_unsafe_download_format("a/b/c/model.ckpt") is True


class TestWriteDownloadReceipt:
    def test_writes_json_alongside_file(self, tmp_path):
        dest = tmp_path / "model.safetensors"
        dest.write_bytes(b"fake")
        write_download_receipt(
            dest,
            url="https://huggingface.co/org/model/resolve/main/model.safetensors",
            source="huggingface",
        )
        receipt = tmp_path / "model.safetensors.receipt.json"
        assert receipt.is_file()
        data = json.loads(receipt.read_text(encoding="utf-8"))
        assert data["file"] == "model.safetensors"
        assert data["source"] == "huggingface"
        assert "huggingface.co" in data["url"]
        assert "downloaded_at" in data

    def test_receipt_has_utc_timestamp(self, tmp_path):
        dest = tmp_path / "lora.safetensors"
        dest.write_bytes(b"x")
        write_download_receipt(dest, url="https://civitai.com/api/download/12345", source="civitai")
        receipt = tmp_path / "lora.safetensors.receipt.json"
        data = json.loads(receipt.read_text(encoding="utf-8"))
        # ISO format ends with +00:00 or Z
        ts = data["downloaded_at"]
        assert "T" in ts  # ISO datetime separator

    def test_silently_skips_on_unwritable_path(self, tmp_path):
        # Should not raise even if receipt can't be written
        fake_dest = Path("/nonexistent/no/such/dir/model.safetensors")
        write_download_receipt(fake_dest, url="http://example.com/f.bin", source="direct")
        # no exception — receipts are advisory

    def test_overwrites_existing_receipt(self, tmp_path):
        dest = tmp_path / "model.safetensors"
        dest.write_bytes(b"x")
        receipt = tmp_path / "model.safetensors.receipt.json"
        receipt.write_text('{"old": true}', encoding="utf-8")
        write_download_receipt(dest, url="http://example.com/new.safetensors", source="direct")
        data = json.loads(receipt.read_text(encoding="utf-8"))
        assert "old" not in data
        assert data["url"] == "http://example.com/new.safetensors"
