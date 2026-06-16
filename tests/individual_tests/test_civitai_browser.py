"""
Tests for aiwf/services/civitai_browser.py and aiwf/core/domain/civitai.py.

All CivitAI API calls are mocked — no live network access.
"""
from __future__ import annotations

import json
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from aiwf.core.domain.civitai import (
    CivitAIModel,
    CivitAIModelVersion,
    CivitAISearchResult,
)
from aiwf.services.civitai_browser import CivitAIBrowser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_version(id=1, name="v1.0", base="SD 1.5", url="https://cdn.example/model.safetensors"):
    return {
        "id": id,
        "name": name,
        "baseModel": base,
        "createdAt": "2024-01-01T00:00:00Z",
        "trainedWords": ["dreamshaper"],
        "files": [{"primary": True, "sizeKB": 2048.0, "downloadUrl": url}],
    }


def _fake_model(id=4384, name="DreamShaper", type="Checkpoint", nsfw=False, versions=None):
    return {
        "id": id,
        "name": name,
        "type": type,
        "nsfw": nsfw,
        "description": "<p>A great model.</p>",
        "tags": ["photorealistic", "portraits"],
        "stats": {"downloadCount": 50000, "rating": 4.8},
        "creator": {"username": "lykon"},
        "modelVersions": versions or [_fake_version()],
    }


def _mock_get(browser, response_body: dict | None):
    """Patch CivitAIBrowser._get to return response_body."""
    browser._get = lambda *a, **kw: response_body


# ---------------------------------------------------------------------------
# Domain model tests
# ---------------------------------------------------------------------------

class TestCivitAIModelVersion:
    def test_size_label_mb(self):
        v = CivitAIModelVersion(id=1, name="v1", base_model="SD 1.5",
                                download_url="", size_kb=2048)
        assert v.size_label() == "2 MB"

    def test_size_label_gb(self):
        v = CivitAIModelVersion(id=1, name="v1", base_model="SD 1.5",
                                download_url="", size_kb=4 * 1024 * 1024)
        assert "GB" in v.size_label()


class TestCivitAIModel:
    def test_url(self):
        m = CivitAIModel(id=4384, name="Test", type="Checkpoint", nsfw=False)
        assert m.url == "https://civitai.com/models/4384"

    def test_latest_version_none_when_empty(self):
        m = CivitAIModel(id=1, name="X", type="Checkpoint", nsfw=False)
        assert m.latest_version is None

    def test_summary_markdown_nsfw_hidden(self):
        m = CivitAIModel(id=1, name="Hidden", type="Checkpoint", nsfw=True)
        md = m.summary_markdown(show_nsfw=False)
        assert "NSFW" in md
        assert "Hidden" not in md

    def test_summary_markdown_nsfw_shown(self):
        v = CivitAIModelVersion(id=1, name="v1", base_model="SD 1.5",
                                download_url="", trigger_words=["trigger"])
        m = CivitAIModel(id=1, name="Shown", type="Checkpoint", nsfw=True,
                         versions=[v])
        md = m.summary_markdown(show_nsfw=True)
        assert "Shown" in md
        assert "trigger" in md

    def test_summary_markdown_safe(self):
        v = CivitAIModelVersion(id=1, name="v1", base_model="SDXL 1.0",
                                download_url="", size_kb=4096,
                                trigger_words=["portrait"])
        m = CivitAIModel(id=4384, name="DreamShaper", type="Checkpoint",
                         nsfw=False, tags=["photorealistic"], versions=[v])
        md = m.summary_markdown()
        assert "DreamShaper" in md
        assert "SDXL 1.0" in md
        assert "portrait" in md


class TestCivitAISearchResult:
    def test_ok_true_when_no_error(self):
        r = CivitAISearchResult(models=[], total_count=0)
        assert r.ok is True

    def test_ok_false_when_error(self):
        r = CivitAISearchResult(models=[], total_count=0, error="timeout")
        assert r.ok is False


# ---------------------------------------------------------------------------
# CivitAIBrowser tests
# ---------------------------------------------------------------------------

class TestCivitAIBrowser:
    def test_search_returns_empty_result_on_network_error(self):
        browser = CivitAIBrowser()
        _mock_get(browser, None)
        result = browser.search("dreamshaper")
        assert not result.ok
        assert result.models == []
        assert result.error is not None

    def test_search_parses_models(self):
        browser = CivitAIBrowser()
        body = {
            "items": [_fake_model()],
            "metadata": {"totalItems": 1, "nextCursor": None},
        }
        _mock_get(browser, body)
        result = browser.search("dreamshaper")
        assert result.ok
        assert len(result.models) == 1
        assert result.models[0].name == "DreamShaper"
        assert result.total_count == 1

    def test_search_parses_version_triggers(self):
        browser = CivitAIBrowser()
        ver = _fake_version()
        ver["trainedWords"] = ["portrait", "dreamy"]
        body = {"items": [_fake_model(versions=[ver])], "metadata": {"totalItems": 1}}
        _mock_get(browser, body)
        result = browser.search("")
        assert result.models[0].versions[0].trigger_words == ["portrait", "dreamy"]

    def test_search_nsfw_flag_excluded(self):
        browser = CivitAIBrowser()
        body = {
            "items": [_fake_model(nsfw=True)],
            "metadata": {"totalItems": 1},
        }
        _mock_get(browser, body)
        result = browser.search("test")
        # nsfw models are parsed (hiding is a UI decision, not service decision)
        assert result.models[0].nsfw is True

    def test_get_model_returns_none_on_error(self):
        browser = CivitAIBrowser()
        _mock_get(browser, None)
        assert browser.get_model(9999) is None

    def test_get_model_parses_response(self):
        browser = CivitAIBrowser()
        _mock_get(browser, _fake_model(id=4384, name="DreamShaper"))
        m = browser.get_model(4384)
        assert m is not None
        assert m.id == 4384
        assert m.name == "DreamShaper"
        assert m.creator == "lykon"
        assert m.stats_downloads == 50000

    def test_list_installed_empty_when_no_dir(self, tmp_path):
        flags = SimpleNamespace(resolved_models_dir=lambda: tmp_path / "nonexistent")
        browser = CivitAIBrowser()
        assert browser.list_installed(flags) == []

    def test_list_installed_finds_safetensors(self, tmp_path):
        models = tmp_path / "models"
        ckpt = models / "Checkpoint"
        ckpt.mkdir(parents=True)
        (ckpt / "dreamshaper.safetensors").write_bytes(b"fake" * 100)
        flags = SimpleNamespace(resolved_models_dir=lambda: models)
        browser = CivitAIBrowser()
        items = browser.list_installed(flags)
        assert len(items) == 1
        assert items[0]["name"] == "dreamshaper"
        assert items[0]["category"] == "Checkpoint"
        assert items[0]["extension"] == ".safetensors"

    def test_list_installed_skips_non_model_files(self, tmp_path):
        models = tmp_path / "models" / "Checkpoint"
        models.mkdir(parents=True)
        (models / "model.safetensors").write_bytes(b"x")
        (models / "readme.txt").write_text("ignore me")
        (models / "config.json").write_text("{}")
        flags = SimpleNamespace(resolved_models_dir=lambda: tmp_path / "models")
        browser = CivitAIBrowser()
        items = browser.list_installed(flags)
        assert len(items) == 1
        assert items[0]["name"] == "model"

    def test_installed_summary_empty(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        flags = SimpleNamespace(resolved_models_dir=lambda: empty)
        browser = CivitAIBrowser()
        md = browser.installed_summary(flags)
        assert "No model files" in md

    def test_installed_summary_groups_by_category(self, tmp_path):
        models = tmp_path / "models"
        (models / "Checkpoint").mkdir(parents=True)
        (models / "Lora").mkdir(parents=True)
        (models / "Checkpoint" / "model_a.safetensors").write_bytes(b"x")
        (models / "Lora" / "style.safetensors").write_bytes(b"x")
        flags = SimpleNamespace(resolved_models_dir=lambda: models)
        browser = CivitAIBrowser()
        md = browser.installed_summary(flags)
        assert "Checkpoint" in md
        assert "Lora" in md
        assert "model_a" in md
        assert "style" in md


# ---------------------------------------------------------------------------
# _civitai_render logic (extracted from model_manager, tested standalone)
# ---------------------------------------------------------------------------

class TestCivitAIRenderLogic:
    """Tests for the render/download-button logic added in the Stretch sprint.

    We duplicate the render function here so the test is independent of
    the Gradio tab import chain.
    """

    def _render(self, result, prefer_safe: bool) -> str:
        """Copy of the render logic from model_manager._civitai_render."""
        if not result.ok:
            return f"⚠ {result.error}"
        if not result.models:
            return "_No results found. Try a different keyword or type._"
        lines = [f"**{result.total_count} results** (showing {len(result.models)})\n"]
        for m in result.models:
            card = m.summary_markdown()
            ver = m.latest_version
            if ver and ver.download_url:
                url = ver.download_url
                ext = url.rsplit(".", 1)[-1].lower() if "." in url else ""
                safe_ext = ext in {"safetensors", "gguf", "bin"}
                if prefer_safe and not safe_ext:
                    card += (
                        f"\n\n> ⚠ Download format `.{ext}` blocked "
                        "(prefer_safetensors is on). Find a `.safetensors` version "
                        "or disable in Settings."
                    )
                else:
                    card += (
                        f"\n\n📥 **Download:** "
                        f"[{ver.name} ({ver.size_label()})]({url})"
                    )
            lines.append(card)
            lines.append("---")
        return "\n\n".join(lines)

    def _make_model(self, download_url: str) -> CivitAIModel:
        from aiwf.core.domain.civitai import CivitAIModelVersion
        ver = CivitAIModelVersion(
            id=1, name="v1.0", base_model="SD 1.5",
            download_url=download_url, size_kb=1024,
        )
        return CivitAIModel(
            id=100, name="Test Model", type="Checkpoint",
            nsfw=False, versions=[ver],
        )

    def _make_result(self, model) -> CivitAISearchResult:
        return CivitAISearchResult(models=[model], total_count=1)

    def test_safetensors_shows_download_link_always(self):
        m = self._make_model("https://civitai.com/api/download/models/1?token=x.safetensors")
        result = self._make_result(m)
        md = self._render(result, prefer_safe=True)
        assert "📥 **Download:**" in md
        assert "blocked" not in md

    def test_ckpt_blocked_when_prefer_safe(self):
        m = self._make_model("https://civitai.com/api/download/models/2/model.ckpt")
        result = self._make_result(m)
        md = self._render(result, prefer_safe=True)
        assert "blocked" in md
        assert "📥" not in md

    def test_ckpt_allowed_when_prefer_safe_off(self):
        m = self._make_model("https://civitai.com/api/download/models/2/model.ckpt")
        result = self._make_result(m)
        md = self._render(result, prefer_safe=False)
        assert "📥 **Download:**" in md
        assert "blocked" not in md

    def test_pt_blocked_when_prefer_safe(self):
        m = self._make_model("https://example.com/weight.pt")
        result = self._make_result(m)
        md = self._render(result, prefer_safe=True)
        assert "blocked" in md

    def test_gguf_allowed_when_prefer_safe(self):
        m = self._make_model("https://example.com/model.gguf")
        result = self._make_result(m)
        md = self._render(result, prefer_safe=True)
        assert "📥 **Download:**" in md

    def test_error_result_shows_warning(self):
        result = CivitAISearchResult(models=[], total_count=0, error="API timeout")
        md = self._render(result, prefer_safe=True)
        assert "⚠" in md
        assert "API timeout" in md

    def test_empty_result_shows_hint(self):
        result = CivitAISearchResult(models=[], total_count=0)
        md = self._render(result, prefer_safe=True)
        assert "No results found" in md

    def test_result_count_shown(self):
        m = self._make_model("https://example.com/m.safetensors")
        result = CivitAISearchResult(models=[m, m], total_count=42)
        md = self._render(result, prefer_safe=True)
        assert "42 results" in md
        assert "showing 2" in md

    def test_no_download_url_shows_no_link(self):
        from aiwf.core.domain.civitai import CivitAIModelVersion
        ver = CivitAIModelVersion(id=1, name="v1", base_model="SD 1.5", download_url="")
        m = CivitAIModel(id=1, name="Anon", type="LoRA", nsfw=False, versions=[ver])
        result = CivitAISearchResult(models=[m], total_count=1)
        md = self._render(result, prefer_safe=True)
        assert "📥" not in md
        assert "blocked" not in md


# ---------------------------------------------------------------------------
# Preview image parsing + gallery helpers
# ---------------------------------------------------------------------------

class TestPreviewImages:
    def _raw_version(self, images=None):
        return {
            "id": 1, "name": "v1", "baseModel": "SD 1.5",
            "files": [{"sizeKB": 2048, "downloadUrl": "https://x.com/m.safetensors", "primary": True}],
            "trainedWords": [],
            "images": images or [],
        }

    def test_safe_preview_images_parsed(self):
        raw = self._raw_version([
            {"url": "https://img.civitai.com/a.jpg", "nsfw": False},
            {"url": "https://img.civitai.com/b.jpg", "nsfw": False},
        ])
        ver = CivitAIBrowser._parse_version(raw)
        assert len(ver.preview_images) == 2
        assert ver.preview_images[0] == "https://img.civitai.com/a.jpg"

    def test_nsfw_preview_images_filtered(self):
        raw = self._raw_version([
            {"url": "https://img.civitai.com/safe.jpg", "nsfw": False},
            {"url": "https://img.civitai.com/nsfw.jpg", "nsfw": True},
        ])
        ver = CivitAIBrowser._parse_version(raw)
        assert len(ver.preview_images) == 1
        assert "safe" in ver.preview_images[0]

    def test_no_images_gives_empty_list(self):
        raw = self._raw_version([])
        ver = CivitAIBrowser._parse_version(raw)
        assert ver.preview_images == []

    def test_preview_image_url_returns_first_safe(self):
        from aiwf.core.domain.civitai import CivitAIModelVersion
        ver = CivitAIModelVersion(
            id=1, name="v1", base_model="SD 1.5", download_url="",
            preview_images=["https://a.com/img.jpg"],
        )
        m = CivitAIModel(id=1, name="X", type="Checkpoint", nsfw=False, versions=[ver])
        assert m.preview_image_url() == "https://a.com/img.jpg"

    def test_preview_image_url_none_when_no_images(self):
        from aiwf.core.domain.civitai import CivitAIModelVersion
        ver = CivitAIModelVersion(id=1, name="v1", base_model="SD 1.5", download_url="")
        m = CivitAIModel(id=1, name="X", type="Checkpoint", nsfw=False, versions=[ver])
        assert m.preview_image_url() is None

    def test_gallery_index_map_skips_models_without_preview(self):
        from aiwf.core.domain.civitai import CivitAIModelVersion, CivitAISearchResult
        def _ver(has_img):
            return CivitAIModelVersion(
                id=1, name="v", base_model="SD 1.5", download_url="",
                preview_images=["https://x.com/img.jpg"] if has_img else [],
            )
        models = [
            CivitAIModel(id=1, name="A", type="Checkpoint", nsfw=False, versions=[_ver(True)]),
            CivitAIModel(id=2, name="B", type="Checkpoint", nsfw=False, versions=[_ver(False)]),
            CivitAIModel(id=3, name="C", type="Checkpoint", nsfw=False, versions=[_ver(True)]),
        ]
        result = CivitAISearchResult(models=models, total_count=3)
        browser = CivitAIBrowser()
        idx_map = browser.gallery_index_map(result)
        assert idx_map == [0, 2]  # models 0 and 2 have previews

    def test_gallery_images_returns_url_caption_pairs(self):
        from aiwf.core.domain.civitai import CivitAIModelVersion, CivitAISearchResult
        ver = CivitAIModelVersion(
            id=1, name="v1", base_model="SD 1.5", download_url="",
            preview_images=["https://x.com/img.jpg"],
        )
        m = CivitAIModel(id=1, name="MyModel", type="LoRA", nsfw=False, versions=[ver])
        result = CivitAISearchResult(models=[m], total_count=1)
        browser = CivitAIBrowser()
        pairs = browser.gallery_images(result)
        assert len(pairs) == 1
        url, caption = pairs[0]
        assert url == "https://x.com/img.jpg"
        assert "MyModel" in caption
