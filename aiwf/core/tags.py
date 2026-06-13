from __future__ import annotations

import re

MAX_TAGS = 12
MAX_TAG_LENGTH = 32

_TAG_TOKEN = re.compile(r"#?([\w][\w\-]*)", re.UNICODE)


def normalize_tag(raw: str) -> str | None:
    """Normalize a single tag: lowercase, hyphenated, alphanumeric."""
    value = raw.strip().lstrip("#").lower().replace(" ", "-")
    value = re.sub(r"[^a-z0-9\-_]", "", value)
    value = re.sub(r"-{2,}", "-", value).strip("-_")
    if not value or len(value) > MAX_TAG_LENGTH:
        return None
    return value


def parse_tags(text: str) -> list[str]:
    """Parse user or infotext input into deduplicated normalized tags."""
    if not text or not str(text).strip():
        return []

    seen: set[str] = set()
    tags: list[str] = []
    for match in _TAG_TOKEN.finditer(str(text)):
        tag = normalize_tag(match.group(1))
        if tag and tag not in seen:
            seen.add(tag)
            tags.append(tag)
            if len(tags) >= MAX_TAGS:
                break
    return tags


def format_tags_display(tags: list[str]) -> str:
    """Space-separated hashtags for the Studio input field."""
    return " ".join(f"#{tag}" for tag in parse_tags(" ".join(tags)))


def format_tags_infotext(tags: list[str]) -> str:
    """Space-separated hashtags for the infotext Tags field (comma-safe)."""
    normalized = parse_tags(" ".join(tags))
    return " ".join(f"#{tag}" for tag in normalized)


def parse_tags_from_params(params: dict) -> list[str]:
    """Extract tags from parsed infotext parameter dict."""
    raw = params.get("Tags") or params.get("tags") or ""
    return parse_tags(str(raw))


def tags_match_filter(tags: list[str], query: str) -> bool:
    """Return True if any tag matches the search query (with or without #)."""
    needle = normalize_tag(query)
    if not needle:
        return True
    return any(needle in tag or tag.startswith(needle) for tag in tags)