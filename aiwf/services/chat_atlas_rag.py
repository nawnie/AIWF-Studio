"""Atlas-style card and lane retrieval helpers for the Chat tab.

This module is pure stdlib. It builds retrieval packets and prepares
chat-style JSONL for a QLoRA Atlas adapter without importing any model
training or embedding dependencies.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


SOURCE_EXTENSIONS = {".py", ".md", ".txt", ".json", ".jsonl", ".toml", ".yaml", ".yml"}
SKIP_PARTS = {".git", ".pytest_cache", "__pycache__", "venv", ".venv", "models", "outputs", "_local"}


@dataclass(frozen=True)
class AtlasRagBuildResult:
    output_dir: Path
    manifest_path: Path
    cards_path: Path
    training_data_path: Path
    train_path: Path
    card_count: int
    source_count: int
    lane_count: int
    warnings: tuple[str, ...]

    def markdown(self) -> str:
        warnings = ""
        if self.warnings:
            warnings = "\n\nWarnings:\n" + "\n".join(f"- {warning}" for warning in self.warnings)
        return (
            f"**Atlas RAG packet built:** `{self.output_dir}`\n\n"
            f"- Cards: `{self.card_count}`\n"
            f"- Sources: `{self.source_count}`\n"
            f"- Lanes: `{self.lane_count}`\n"
            f"- Cards JSONL: `{self.cards_path}`\n"
            f"- QLoRA data: `{self.training_data_path}`\n"
            f"- Train split: `{self.train_path}`\n"
            f"- Manifest: `{self.manifest_path}`"
            f"{warnings}"
        )


def build_atlas_rag_packet(
    source_paths: str | Iterable[str],
    *,
    packet_name: str = "atlas_rag_packet",
    output_root: str | Path = Path("datasets") / "atlas_rag",
    max_files: int = 200,
    max_chars_per_source: int = 6000,
) -> AtlasRagBuildResult:
    """Build an Atlas card/lane packet plus QLoRA training JSONL."""
    roots = _parse_paths(source_paths)
    if not roots:
        raise ValueError("At least one source file or folder is required.")

    files, warnings = _collect_files(roots, max_files=max_files)
    if not files:
        raise ValueError("No supported source files were found.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_root) / f"{_slug(packet_name)}_{stamp}"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing Atlas packet: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    sources: list[dict[str, Any]] = []
    cards: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    cartographer: list[dict[str, Any]] = []
    for index, file in enumerate(files, start=1):
        source = _source_row(file, index)
        lane = _lane_for_source(file)
        card = _card_for_source(file, source, lane, index, max_chars_per_source=max_chars_per_source)
        record = _training_record_for_card(card, source)
        sources.append(source)
        cards.append(card)
        records.append(record)
        cartographer.append(
            {
                "source_id": source["id"],
                "card_id": card["id"],
                "lane": lane,
                "path": source["path"],
                "retrieval_terms": sorted(_tokens(card["claim"]) | _tokens(card["title"])),
            }
        )

    _assign_splits(records)
    splits = {
        "train": [row for row in records if row["metadata"]["split"] == "train"],
        "validation": [row for row in records if row["metadata"]["split"] == "validation"],
        "test": [row for row in records if row["metadata"]["split"] == "test"],
    }
    lane_names = sorted({card["lane"] for card in cards})

    cards_path = output_dir / "atlas_cards.jsonl"
    training_data_path = output_dir / "training_data.jsonl"
    train_path = output_dir / "splits" / "train.jsonl"
    _write_jsonl(output_dir / "source_registry.jsonl", sources)
    _write_jsonl(cards_path, cards)
    _write_jsonl(training_data_path, records)
    for split, rows in splits.items():
        _write_jsonl(output_dir / "splits" / f"{split}.jsonl", rows)
    for lane in lane_names:
        _write_jsonl(output_dir / "lanes" / f"{lane}.jsonl", [card for card in cards if card["lane"] == lane])
    _write_json(output_dir / "cartographer_map.json", {"lanes": lane_names, "entries": cartographer})

    manifest = {
        "schema_version": "aiwf-atlas-rag-v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "packet_name": packet_name,
        "output_dir": str(output_dir.resolve()),
        "card_count": len(cards),
        "source_count": len(sources),
        "lane_count": len(lane_names),
        "lane_counts": dict(sorted(Counter(card["lane"] for card in cards).items())),
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "cards_path": str(cards_path.resolve()),
        "training_data_path": str(training_data_path.resolve()),
        "train_path": str(train_path.resolve()),
        "activation_policy": "Activating Atlas RAG trains a QLoRA adapter from this chat-style training data.",
        "warnings": warnings,
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    _write_readme(output_dir, manifest)
    _validate_packet(output_dir)

    return AtlasRagBuildResult(
        output_dir=output_dir.resolve(),
        manifest_path=manifest_path.resolve(),
        cards_path=cards_path.resolve(),
        training_data_path=training_data_path.resolve(),
        train_path=train_path.resolve(),
        card_count=len(cards),
        source_count=len(sources),
        lane_count=len(lane_names),
        warnings=tuple(warnings),
    )


def retrieve_atlas_cards(packet_path: str | Path, query: str, *, top_k: int = 4) -> list[dict[str, Any]]:
    """Return top lexical card matches from an Atlas packet directory or cards file."""
    cards_path = _cards_path(packet_path)
    cards = _load_jsonl(cards_path)
    query_tokens = _tokens(query)
    if not query_tokens:
        return cards[:top_k]
    scored: list[tuple[int, dict[str, Any]]] = []
    for card in cards:
        haystack = " ".join(
            str(card.get(key, ""))
            for key in ("title", "lane", "retrieval_intent", "claim")
        )
        haystack += " " + " ".join(str(tag) for tag in card.get("tags", []))
        score = len(query_tokens & _tokens(haystack))
        if score:
            scored.append((score, card))
    scored.sort(key=lambda item: (-item[0], item[1].get("id", "")))
    return [card for _score, card in scored[:top_k]]


def format_cards_for_prompt(cards: list[dict[str, Any]]) -> str:
    if not cards:
        return ""
    lines = ["Atlas retrieved cards:"]
    for card in cards:
        evidence = card.get("evidence") or []
        path = evidence[0].get("path", "") if evidence and isinstance(evidence[0], dict) else ""
        lines.append(
            f"- [{card.get('lane', 'general')}] {card.get('title', card.get('id', 'card'))}: "
            f"{card.get('claim', '')} Evidence: {path}"
        )
    return "\n".join(lines)


def _cards_path(packet_path: str | Path) -> Path:
    path = Path(packet_path)
    if path.is_dir():
        path = path / "atlas_cards.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Atlas cards not found: {path}")
    return path


def _parse_paths(source_paths: str | Iterable[str]) -> list[Path]:
    if isinstance(source_paths, str):
        raw = re.split(r"[\r\n]+", source_paths)
    else:
        raw = list(source_paths)
    return [Path(str(value).strip().strip('"')) for value in raw if str(value).strip()]


def _collect_files(roots: list[Path], *, max_files: int) -> tuple[list[Path], list[str]]:
    files: list[Path] = []
    warnings: list[str] = []
    missing: list[str] = []
    for root in roots:
        if not root.exists():
            missing.append(str(root))
            continue
        if root.is_file():
            if _supported(root):
                files.append(root.resolve())
            continue
        for path in root.rglob("*"):
            if len(files) >= max_files:
                warnings.append(f"Stopped at max_files={max_files}; narrow the source folder for more control.")
                break
            if not path.is_file() or not _supported(path):
                continue
            try:
                rel = path.relative_to(root)
            except ValueError:
                rel = path
            if any(part in SKIP_PARTS for part in rel.parts):
                continue
            files.append(path.resolve())
    if missing:
        warnings.append("Missing source paths skipped: " + ", ".join(missing[:5]))
    return sorted(dict.fromkeys(files), key=lambda item: str(item).lower()), warnings


def _supported(path: Path) -> bool:
    return path.suffix.lower() in SOURCE_EXTENSIONS and path.stat().st_size <= 2_000_000


def _source_row(path: Path, index: int) -> dict[str, Any]:
    text = _read_text(path)
    return {
        "id": f"src-{index:04d}-{_slug(path.stem)}",
        "path": str(path.resolve()),
        "name": path.name,
        "suffix": path.suffix.lower(),
        "sha256": _sha256(path),
        "line_count": text.count("\n") + 1 if text else 0,
        "allowed_use": "local Atlas RAG packet with provenance; review licenses before external sharing",
    }


def _lane_for_source(path: Path) -> str:
    lowered = str(path).lower()
    suffix = path.suffix.lower()
    if "test" in lowered:
        return "tests"
    if suffix == ".py":
        if "web" in lowered or "tab" in lowered:
            return "ui"
        if "service" in lowered:
            return "services"
        return "code"
    if suffix in {".json", ".toml", ".yaml", ".yml"}:
        return "config"
    if any(word in lowered for word in ("rag", "retrieval", "atlas", "cartographer")):
        return "retrieval"
    if suffix == ".md":
        return "docs"
    return "general"


def _card_for_source(path: Path, source: dict[str, Any], lane: str, index: int, *, max_chars_per_source: int) -> dict[str, Any]:
    text = _read_text(path)[:max_chars_per_source]
    title = path.name
    claim = _summarize_source(path, text)
    tags = sorted({lane, path.suffix.lower().lstrip(".") or "source", *_tokens(path.stem)})
    return {
        "id": f"atlas-card-{index:04d}-{_slug(path.stem)}",
        "lane": lane,
        "title": title,
        "retrieval_intent": f"Use when the user asks about {lane} behavior related to {title}.",
        "claim": claim,
        "evidence": [{"source_id": source["id"], "path": source["path"], "sha256": source["sha256"]}],
        "source_class": "local",
        "verification_state": "source_hash_recorded",
        "tags": tags[:20],
    }


def _summarize_source(path: Path, text: str) -> str:
    headings = [
        line.strip().lstrip("#").strip()
        for line in text.splitlines()
        if line.strip().startswith("#")
    ][:8]
    if headings:
        return f"`{path.name}` covers: {', '.join(headings)}."
    if path.suffix.lower() == ".py":
        classes = re.findall(r"^class\s+([A-Za-z_][A-Za-z0-9_]*)", text, re.MULTILINE)[:12]
        funcs = re.findall(r"^(?:async\s+def|def)\s+([A-Za-z_][A-Za-z0-9_]*)", text, re.MULTILINE)[:18]
        parts = [f"`{path.name}` is Python source."]
        if classes:
            parts.append("Classes: " + ", ".join(classes) + ".")
        if funcs:
            parts.append("Functions: " + ", ".join(funcs) + ".")
        return " ".join(parts)
    excerpt = " ".join(line.strip() for line in text.splitlines()[:8] if line.strip())[:700]
    return f"`{path.name}` contains local source material. Excerpt: {excerpt}"


def _training_record_for_card(card: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"train-{card['id']}",
        "messages": [
            {
                "role": "system",
                "content": "You are an Atlas RAG adapter. Use lane, card, and evidence metadata before answering.",
            },
            {
                "role": "user",
                "content": f"Retrieve the relevant Atlas card for lane `{card['lane']}` and answer from it.",
            },
            {
                "role": "assistant",
                "content": (
                    f"Lane: {card['lane']}\nCard: {card['title']}\n"
                    f"Answer: {card['claim']}\nEvidence: {source['path']}"
                ),
            },
        ],
        "metadata": {
            "record_type": "atlas_card_retrieval",
            "card_id": card["id"],
            "lane": card["lane"],
            "source_id": source["id"],
            "source_path": source["path"],
            "split": "train",
        },
    }


def _assign_splits(records: list[dict[str, Any]]) -> None:
    for index, record in enumerate(records):
        bucket = index % 10
        record["metadata"]["split"] = "validation" if bucket == 8 else "test" if bucket == 9 else "train"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_readme(output_dir: Path, manifest: dict[str, Any]) -> None:
    (output_dir / "README.md").write_text(
        "# Atlas RAG Packet\n\n"
        "This packet contains source-backed Atlas cards, lane files, a cartographer map, "
        "and chat-style QLoRA training rows.\n\n"
        f"- Cards: {manifest['card_count']}\n"
        f"- Lanes: {manifest['lane_count']}\n"
        "- Activation policy: train a QLoRA Atlas adapter from `training_data.jsonl`.\n",
        encoding="utf-8",
    )


def _validate_packet(output_dir: Path) -> None:
    for path in output_dir.rglob("*.jsonl"):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    json.loads(line)
                elif path.name in {"atlas_cards.jsonl", "training_data.jsonl"}:
                    raise ValueError(f"{path}:{line_number} contains a blank row.")
    for path in output_dir.rglob("*.json"):
        json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
    return rows


def _tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z0-9_]{3,}", str(value).lower())}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()[:80] or "item"
