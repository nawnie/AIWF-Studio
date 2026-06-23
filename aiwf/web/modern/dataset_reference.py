from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


DEFAULT_LATEST_POINTER = (
    Path.home()
    / "Desktop"
    / "MoK-Project"
    / "datasets"
    / "gradio6_textbook_illustrations_aiwf_latest.json"
)


@dataclass(frozen=True)
class GradioReferenceRecord:
    id: str
    image_id: str
    chapter_title: str
    asset_type: str
    caption: str
    image_path: Path
    caption_path: Path
    split: str
    status: str
    notes: str


def resolve_dataset_dir(pointer: Path = DEFAULT_LATEST_POINTER) -> Path | None:
    if not pointer.exists():
        return None
    try:
        data = json.loads(pointer.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    raw = data.get("dataset_dir")
    if not raw:
        return None
    path = Path(raw)
    return path if path.exists() else None


def load_manifest(dataset_dir: Path | None = None) -> list[GradioReferenceRecord]:
    root = dataset_dir or resolve_dataset_dir()
    if root is None:
        return []
    manifest = root / "manifest.csv"
    if not manifest.exists():
        return []
    rows: list[GradioReferenceRecord] = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            image_rel = row.get("file_name", "")
            caption_rel = row.get("caption_file", "")
            rows.append(
                GradioReferenceRecord(
                    id=row.get("id", ""),
                    image_id=row.get("image_id", ""),
                    chapter_title=row.get("chapter_title", ""),
                    asset_type=row.get("type", ""),
                    caption=row.get("caption", ""),
                    image_path=root / image_rel,
                    caption_path=root / caption_rel,
                    split=row.get("split", ""),
                    status=row.get("status", ""),
                    notes=row.get("notes", ""),
                )
            )
    return rows


def filter_records(
    records: list[GradioReferenceRecord],
    *,
    query: str = "",
    asset_type: str = "All",
) -> list[GradioReferenceRecord]:
    needle = (query or "").strip().lower()
    selected_type = (asset_type or "All").strip().lower()
    out: list[GradioReferenceRecord] = []
    for record in records:
        if selected_type not in {"", "all"} and record.asset_type.lower() != selected_type:
            continue
        if needle:
            haystack = " ".join(
                [
                    record.id,
                    record.image_id,
                    record.chapter_title,
                    record.asset_type,
                    record.caption,
                    record.notes,
                ]
            ).lower()
            if needle not in haystack:
                continue
        out.append(record)
    return out


def dataset_summary_markdown(records: list[GradioReferenceRecord], dataset_dir: Path | None = None) -> str:
    root = dataset_dir or resolve_dataset_dir()
    if root is None:
        return "**Dataset unavailable** - expected the MoK Gradio dataset pointer on the Desktop."
    counts: dict[str, int] = {}
    for record in records:
        counts[record.asset_type] = counts.get(record.asset_type, 0) + 1
    type_summary = ", ".join(f"{name}: {count}" for name, count in sorted(counts.items())) or "no records"
    return (
        f"**MoK Gradio reference dataset**  \n"
        f"`{root}`  \n"
        f"Records: **{len(records)}**. Types: {type_summary}."
    )


def gallery_items(records: list[GradioReferenceRecord], limit: int = 18) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for record in records:
        if record.image_path.exists():
            caption = f"{record.image_id} - {record.chapter_title}"
            items.append((str(record.image_path), caption))
        if len(items) >= limit:
            break
    return items


def table_rows(records: list[GradioReferenceRecord], limit: int = 80) -> list[list[str]]:
    return [
        [
            record.id,
            record.asset_type,
            record.split,
            record.chapter_title,
            record.caption[:220],
            record.status,
        ]
        for record in records[:limit]
    ]
