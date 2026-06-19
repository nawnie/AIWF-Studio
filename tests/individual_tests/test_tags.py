from pathlib import Path

from aiwf.core.domain.generation import GenerationMode, GenerationRequest
from aiwf.core.domain.models import Checkpoint
from aiwf.core.config.settings import UserSettings
from aiwf.core.infotext import format_infotext, infotext_to_request_updates, parse_infotext
from aiwf.core.tags import format_tags_display, format_tags_infotext, normalize_tag, parse_tags, tags_match_filter
from aiwf.services.tags import TagService


def test_normalize_tag():
    assert normalize_tag("#Client Work") == "client-work"
    assert normalize_tag("WIP!!!") == "wip"
    assert normalize_tag("") is None


def test_parse_tags_dedupes_and_limits():
    raw = "#portrait #Portrait portrait, #landscape #extra"
    tags = parse_tags(raw)
    assert tags == ["portrait", "landscape", "extra"]


def test_format_tags_display():
    assert format_tags_display(["portrait", "wip"]) == "#portrait #wip"


def test_infotext_tags_round_trip():
    request = GenerationRequest(prompt="sunset", tags=["portrait", "client-work"])
    checkpoint = Checkpoint(
        id="test",
        title="test",
        filename="test.safetensors",
        path="/tmp/test.safetensors",
    )
    text = format_infotext(request, 7, checkpoint)
    assert "Tags: #portrait #client-work" in text
    params = parse_infotext(text)
    updates = infotext_to_request_updates(params, GenerationMode.TXT2IMG)
    assert updates["tags"] == ["portrait", "client-work"]


def test_tags_match_filter():
    assert tags_match_filter(["portrait", "wip"], "portrait")
    assert tags_match_filter(["client-work"], "#client")
    assert not tags_match_filter(["landscape"], "portrait")


def test_tag_service_filter_entries_accepts_none_query():
    service = TagService(UserSettings(), Path("."))
    assert service.filter_entries([], None) == []
