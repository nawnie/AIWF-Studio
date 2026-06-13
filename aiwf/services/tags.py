from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image

from aiwf.core.config.settings import UserSettings
from aiwf.core.tags import parse_tags, tags_match_filter
from aiwf.services.metadata import MetadataService

MAX_RECENT_TAGS = 24


@dataclass(frozen=True)
class LibraryEntry:
    path: Path
    tags: list[str]
    infotext: str
    modified_at: datetime


class TagService:
    def __init__(self, settings: UserSettings, output_root: Path) -> None:
        self.settings = settings
        self.output_root = output_root
        self._metadata = MetadataService()

    def remember_tags(self, tags: list[str], *, save: Callable[[], None]) -> list[str]:
        """Update recent-tag history and persist settings."""
        normalized = parse_tags(" ".join(tags))
        if not normalized:
            return self.recent_tag_choices()

        for tag in reversed(normalized):
            if tag in self.settings.recent_tags:
                self.settings.recent_tags.remove(tag)
            self.settings.recent_tags.insert(0, tag)
        self.settings.recent_tags = self.settings.recent_tags[:MAX_RECENT_TAGS]
        save()
        return self.recent_tag_choices()

    def recent_tag_choices(self) -> list[str]:
        return list(self.settings.recent_tags)

    def scan_library(self) -> list[LibraryEntry]:
        if not self.output_root.exists():
            return []

        entries: list[LibraryEntry] = []
        for path in self.output_root.rglob("*.png"):
            try:
                stat = path.stat()
                with Image.open(path) as image:
                    infotext = self._metadata.read_infotext(image) or ""
                    tags = self._metadata.read_tags(image)
                entries.append(
                    LibraryEntry(
                        path=path,
                        tags=tags,
                        infotext=infotext,
                        modified_at=datetime.fromtimestamp(stat.st_mtime),
                    )
                )
            except OSError:
                continue

        entries.sort(key=lambda item: item.modified_at, reverse=True)
        return entries

    def filter_entries(self, entries: list[LibraryEntry], query: str | None) -> list[LibraryEntry]:
        text = query or ""
        tags = parse_tags(text)
        if not tags:
            needle = text.strip().lstrip("#").lower()
            if not needle:
                return entries
            return [
                entry
                for entry in entries
                if any(needle in tag for tag in entry.tags)
                or needle in entry.path.name.lower()
            ]
        return [
            entry
            for entry in entries
            if all(tags_match_filter(entry.tags, tag) for tag in tags)
        ]

    def collect_tag_counts(self, entries: list[LibraryEntry]) -> list[tuple[str, int]]:
        counts: dict[str, int] = {}
        for entry in entries:
            for tag in entry.tags:
                counts[tag] = counts.get(tag, 0) + 1
        return sorted(counts.items(), key=lambda item: (-item[1], item[0]))
