"""Tests for aiwf/services/model_info_lookup.py.

All network calls are patched via unittest.mock so no real HTTP requests
are made.  Tests verify parsing logic and graceful error handling.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from aiwf.services.model_info_lookup import (
    ModelInfoLookup,
    RemoteModelInfo,
    _coerce_list,
    _parse_civitai_ref,
    _strip_html,
    get_model_info_lookup,
)

lookup = ModelInfoLookup()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestCoerceList:
    def test_list_passthrough(self):
        assert _coerce_list(["a", "b"]) == ["a", "b"]

    def test_comma_string(self):
        assert _coerce_list("cat, dog") == ["cat", "dog"]

    def test_single_string(self):
        assert _coerce_list("solo") == ["solo"]

    def test_empty_string(self):
        assert _coerce_list("") == []

    def test_none(self):
        assert _coerce_list(None) == []

    def test_strips_whitespace(self):
        assert _coerce_list(["  hello  ", " world "]) == ["hello", "world"]


class TestStripHtml:
    def test_removes_tags(self):
        assert _strip_html("<p>Hello <b>world</b></p>") == "Hello world"

    def test_decodes_entities(self):
        assert "&amp;" not in _strip_html("foo &amp; bar")
        assert "&" in _strip_html("foo &amp; bar")

    def test_empty_string(self):
        assert _strip_html("") == ""

    def test_no_tags(self):
        assert _strip_html("plain text") == "plain text"


class TestParseCivitaiRef:
    def test_bare_integer(self):
        model_id, version_id = _parse_civitai_ref("4384")
        assert model_id == 4384
        assert version_id is None

    def test_url_model_id(self):
        model_id, version_id = _parse_civitai_ref("https://civitai.com/models/4384")
        assert model_id == 4384
        assert version_id is None

    def test_url_with_version(self):
        model_id, version_id = _parse_civitai_ref(
            "https://civitai.com/models/4384?modelVersionId=128713"
        )
        assert model_id == 4384
        assert version_id == 128713

    def test_non_civitai_url_returns_none(self):
        model_id, _ = _parse_civitai_ref("https://example.com/models/99")
        assert model_id is None

    def test_garbage_returns_none(self):
        model_id, _ = _parse_civitai_ref("not a ref")
        assert model_id is None

    def test_empty_returns_none(self):
        model_id, _ = _parse_civitai_ref("")
        assert model_id is None


# ---------------------------------------------------------------------------
# RemoteModelInfo.summary_markdown
# ---------------------------------------------------------------------------

class TestRemoteModelInfoSummary:
    def _make(self, **kwargs) -> RemoteModelInfo:
        defaults = dict(source="huggingface", name="TestModel")
        defaults.update(kwargs)
        return RemoteModelInfo(**defaults)

    def test_name_in_summary(self):
        md = self._make(name="MyModel").summary_markdown()
        assert "MyModel" in md

    def test_source_in_summary(self):
        md = self._make(source="civitai").summary_markdown()
        assert "civitai" in md

    def test_trigger_words_shown(self):
        md = self._make(trigger_words=["fluffy cat", "detailed"]).summary_markdown()
        assert "fluffy cat" in md

    def test_tags_shown(self):
        md = self._make(tags=["anime", "lora"]).summary_markdown()
        assert "anime" in md

    def test_base_model_shown(self):
        md = self._make(base_model="SDXL 1.0").summary_markdown()
        assert "SDXL" in md

    def test_url_shown(self):
        md = self._make(url="https://example.com/model").summary_markdown()
        assert "https://example.com/model" in md

    def test_downloads_shown(self):
        md = self._make(downloads=12345).summary_markdown()
        assert "12,345" in md


# ---------------------------------------------------------------------------
# HuggingFace lookup (mocked)
# ---------------------------------------------------------------------------

_HF_API_RESPONSE = {
    "modelId": "stabilityai/stable-diffusion-xl-base-1.0",
    "tags": [
        "diffusers", "text-to-image",
        "license:openrail++",
        "base_model:stability-ai/sdxl",
    ],
    "downloads": 987654,
    "cardData": {
        "description": "SDXL base model.",
        "trigger_words": ["masterpiece"],
        "base_model": ["stability-ai/sdxl"],
    },
}


def _fake_http_get_json(url, **kwargs):
    if "huggingface.co/api/models" in url:
        return _HF_API_RESPONSE
    raise RuntimeError(f"Unexpected URL in test: {url}")


class TestLookupHF:
    def test_returns_remote_model_info(self):
        with patch("aiwf.services.model_info_lookup._http_get_json", side_effect=_fake_http_get_json):
            result = lookup.lookup_hf("stabilityai/stable-diffusion-xl-base-1.0")
        assert isinstance(result, RemoteModelInfo)
        assert result.source == "huggingface"

    def test_name_parsed(self):
        with patch("aiwf.services.model_info_lookup._http_get_json", side_effect=_fake_http_get_json):
            result = lookup.lookup_hf("stabilityai/stable-diffusion-xl-base-1.0")
        assert "stabilityai" in result.name

    def test_license_extracted(self):
        with patch("aiwf.services.model_info_lookup._http_get_json", side_effect=_fake_http_get_json):
            result = lookup.lookup_hf("stabilityai/stable-diffusion-xl-base-1.0")
        assert "openrail" in result.license

    def test_downloads_parsed(self):
        with patch("aiwf.services.model_info_lookup._http_get_json", side_effect=_fake_http_get_json):
            result = lookup.lookup_hf("stabilityai/stable-diffusion-xl-base-1.0")
        assert result.downloads == 987654

    def test_trigger_words_extracted(self):
        with patch("aiwf.services.model_info_lookup._http_get_json", side_effect=_fake_http_get_json):
            result = lookup.lookup_hf("stabilityai/stable-diffusion-xl-base-1.0")
        assert "masterpiece" in result.trigger_words

    def test_description_extracted(self):
        with patch("aiwf.services.model_info_lookup._http_get_json", side_effect=_fake_http_get_json):
            result = lookup.lookup_hf("stabilityai/stable-diffusion-xl-base-1.0")
        assert "SDXL" in result.description

    def test_tags_exclude_license_prefix(self):
        with patch("aiwf.services.model_info_lookup._http_get_json", side_effect=_fake_http_get_json):
            result = lookup.lookup_hf("stabilityai/stable-diffusion-xl-base-1.0")
        assert not any(t.startswith("license:") for t in result.tags)

    def test_network_error_returns_none(self):
        with patch("aiwf.services.model_info_lookup._http_get_json", side_effect=OSError("timeout")):
            result = lookup.lookup_hf("org/repo")
        assert result is None

    def test_empty_repo_id_returns_none(self):
        result = lookup.lookup_hf("")
        assert result is None

    def test_repo_id_without_slash_returns_none(self):
        result = lookup.lookup_hf("noslash")
        assert result is None


# ---------------------------------------------------------------------------
# CivitAI lookup (mocked)
# ---------------------------------------------------------------------------

_CIVITAI_MODEL_RESPONSE = {
    "id": 4384,
    "name": "Deliberate",
    "description": "<p>A <b>great</b> model.</p>",
    "tags": ["photorealistic", "portrait"],
    "stats": {"downloadCount": 50000},
    "allowCommercialUse": "Sell",
    "modelVersions": [
        {
            "id": 128713,
            "baseModel": "SD 1.5",
            "trainedWords": ["dslr", "sharp focus"],
        }
    ],
}


def _fake_civitai_get(url, **kwargs):
    if "civitai.com/api/v1/models" in url:
        return _CIVITAI_MODEL_RESPONSE
    raise RuntimeError(f"Unexpected URL in test: {url}")


class TestLookupCivitAI:
    def test_returns_remote_model_info(self):
        with patch("aiwf.services.model_info_lookup._http_get_json", side_effect=_fake_civitai_get):
            result = lookup.lookup_civitai("4384")
        assert isinstance(result, RemoteModelInfo)
        assert result.source == "civitai"

    def test_name_parsed(self):
        with patch("aiwf.services.model_info_lookup._http_get_json", side_effect=_fake_civitai_get):
            result = lookup.lookup_civitai("4384")
        assert result.name == "Deliberate"

    def test_html_stripped_from_description(self):
        with patch("aiwf.services.model_info_lookup._http_get_json", side_effect=_fake_civitai_get):
            result = lookup.lookup_civitai("4384")
        assert "<" not in result.description
        assert "great" in result.description

    def test_trigger_words_extracted(self):
        with patch("aiwf.services.model_info_lookup._http_get_json", side_effect=_fake_civitai_get):
            result = lookup.lookup_civitai("4384")
        assert "dslr" in result.trigger_words

    def test_base_model_extracted(self):
        with patch("aiwf.services.model_info_lookup._http_get_json", side_effect=_fake_civitai_get):
            result = lookup.lookup_civitai("4384")
        assert result.base_model == "SD 1.5"

    def test_downloads_extracted(self):
        with patch("aiwf.services.model_info_lookup._http_get_json", side_effect=_fake_civitai_get):
            result = lookup.lookup_civitai("4384")
        assert result.downloads == 50000

    def test_tags_extracted(self):
        with patch("aiwf.services.model_info_lookup._http_get_json", side_effect=_fake_civitai_get):
            result = lookup.lookup_civitai("4384")
        assert "photorealistic" in result.tags

    def test_network_error_returns_none(self):
        with patch("aiwf.services.model_info_lookup._http_get_json", side_effect=OSError("timeout")):
            result = lookup.lookup_civitai("4384")
        assert result is None

    def test_url_input_parsed(self):
        with patch("aiwf.services.model_info_lookup._http_get_json", side_effect=_fake_civitai_get):
            result = lookup.lookup_civitai("https://civitai.com/models/4384")
        assert result is not None
        assert result.name == "Deliberate"

    def test_invalid_ref_returns_none(self):
        result = lookup.lookup_civitai("not-a-ref")
        assert result is None


# ---------------------------------------------------------------------------
# Ollama lookup (mocked)
# ---------------------------------------------------------------------------

_OLLAMA_SHOW_RESPONSE = {
    "details": {
        "family": "llama",
        "parameter_size": "8B",
        "quantization_level": "Q4_K_M",
        "families": ["llama"],
    },
    "parameters": "temperature 0.7\nstop \"<|eot_id|>\"",
    "system": "You are a helpful assistant.",
    "modelfile": "FROM llama3:8b\nSYSTEM You are helpful.",
}


def _fake_ollama_post(url, body, **kwargs):
    if "/api/show" in url:
        return _OLLAMA_SHOW_RESPONSE
    raise RuntimeError(f"Unexpected URL: {url}")


class TestLookupOllama:
    def test_returns_remote_model_info(self):
        with patch("aiwf.services.model_info_lookup._http_post_json", side_effect=_fake_ollama_post):
            result = lookup.lookup_ollama("llama3:8b")
        assert isinstance(result, RemoteModelInfo)
        assert result.source == "ollama"

    def test_name_includes_param_size(self):
        with patch("aiwf.services.model_info_lookup._http_post_json", side_effect=_fake_ollama_post):
            result = lookup.lookup_ollama("llama3:8b")
        assert "8B" in result.name

    def test_family_in_tags(self):
        with patch("aiwf.services.model_info_lookup._http_post_json", side_effect=_fake_ollama_post):
            result = lookup.lookup_ollama("llama3:8b")
        assert "llama" in result.tags

    def test_quant_in_tags(self):
        with patch("aiwf.services.model_info_lookup._http_post_json", side_effect=_fake_ollama_post):
            result = lookup.lookup_ollama("llama3:8b")
        assert "Q4_K_M" in result.tags

    def test_network_error_returns_none(self):
        with patch("aiwf.services.model_info_lookup._http_post_json", side_effect=OSError("refused")):
            result = lookup.lookup_ollama("llama3:8b")
        assert result is None

    def test_empty_name_returns_none(self):
        result = lookup.lookup_ollama("")
        assert result is None


# ---------------------------------------------------------------------------
# Auto-detect routing
# ---------------------------------------------------------------------------

class TestLookupAuto:
    def test_civitai_url_routes_to_civitai(self):
        with patch.object(lookup, "lookup_civitai", return_value=None) as mock_cv:
            lookup.lookup_auto("https://civitai.com/models/4384")
        mock_cv.assert_called_once()

    def test_bare_integer_routes_to_civitai(self):
        with patch.object(lookup, "lookup_civitai", return_value=None) as mock_cv:
            lookup.lookup_auto("4384")
        mock_cv.assert_called_once()

    def test_hf_repo_id_routes_to_hf(self):
        with patch.object(lookup, "lookup_hf", return_value=None) as mock_hf:
            lookup.lookup_auto("org/repo")
        mock_hf.assert_called_once()

    def test_plain_name_tries_ollama_first(self):
        with patch.object(lookup, "lookup_ollama", return_value=None) as mock_ol:
            lookup.lookup_auto("llama3:8b")
        mock_ol.assert_called_once()

    def test_empty_query_returns_none(self):
        assert lookup.lookup_auto("") is None

    def test_ollama_hit_short_circuits_hf(self):
        sentinel = RemoteModelInfo(source="ollama", name="test")
        with patch.object(lookup, "lookup_ollama", return_value=sentinel):
            with patch.object(lookup, "lookup_hf") as mock_hf:
                result = lookup.lookup_auto("llama3:8b")
        mock_hf.assert_not_called()
        assert result is sentinel


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

class TestSingleton:
    def test_returns_same_instance(self):
        a = get_model_info_lookup()
        b = get_model_info_lookup()
        assert a is b

    def test_is_lookup_instance(self):
        assert isinstance(get_model_info_lookup(), ModelInfoLookup)
