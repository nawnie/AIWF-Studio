"""Build small source-backed datasets for local AI model training.

This is intentionally pure stdlib. It prepares chat-style JSONL records and
provenance files, but it does not install model-training dependencies or touch
the GPU.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


DEFAULT_OUTPUT_ROOT = Path("datasets") / "ai_model_training"
SOURCE_EXTENSIONS = {".py", ".md", ".txt", ".json", ".jsonl", ".toml", ".yaml", ".yml"}
SKIP_PARTS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "venv",
    ".venv",
    "models",
    "outputs",
    "_local",
}
SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"AIza[A-Za-z0-9_-]{20,}"),
    re.compile(
        r"(?i)(api[_-]?key|secret|password|token)\s*[:=]\s*[\"']?[A-Za-z0-9_./+=-]{16,}"
    ),
)


@dataclass(frozen=True)
class DatasetBuildResult:
    output_dir: Path
    manifest_path: Path
    training_data_path: Path
    train_path: Path
    validation_path: Path
    test_path: Path
    record_count: int
    source_count: int
    warnings: tuple[str, ...]

    def markdown(self) -> str:
        return (
            f"**Dataset built:** `{self.output_dir}`\n\n"
            f"- Records: `{self.record_count}`\n"
            f"- Sources: `{self.source_count}`\n"
            f"- Master JSONL: `{self.training_data_path}`\n"
            f"- Train split: `{self.train_path}`\n"
            f"- Validation split: `{self.validation_path}`\n"
            f"- Test split: `{self.test_path}`\n"
            f"- Manifest: `{self.manifest_path}`"
            + (f"\n\nWarnings:\n" + "\n".join(f"- {warning}" for warning in self.warnings) if self.warnings else "")
        )


def build_ai_model_dataset(
    source_paths: str | Iterable[str],
    *,
    dataset_name: str = "ai_bot_dataset",
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    max_files: int = 200,
    max_chars_per_file: int = 6000,
) -> DatasetBuildResult:
    """Build a chat-style SFT dataset from local source files/folders."""
    source_roots = _parse_source_paths(source_paths)
    if not source_roots:
        raise ValueError("At least one source file or folder is required.")

    files, warnings = _collect_files(source_roots, max_files=max_files)
    if not files:
        raise ValueError("No supported source files were found.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_root) / f"{_slugify(dataset_name)}_{stamp}"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset folder: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    source_rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for index, path in enumerate(files, start=1):
        source_row = _source_row(path, index)
        source_rows.append(source_row)
        records.extend(
            _records_for_source(
                path,
                source_row,
                max_chars_per_file=max_chars_per_file,
            )
        )

    if not records:
        raise ValueError("No trainable records could be produced from the selected sources.")

    _assign_splits(records)
    splits = {
        "train": [row for row in records if row["metadata"]["split"] == "train"],
        "validation": [row for row in records if row["metadata"]["split"] == "validation"],
        "test": [row for row in records if row["metadata"]["split"] == "test"],
    }

    training_data_path = output_dir / "training_data.jsonl"
    train_path = output_dir / "splits" / "train.jsonl"
    validation_path = output_dir / "splits" / "validation.jsonl"
    test_path = output_dir / "splits" / "test.jsonl"
    _write_jsonl(output_dir / "source_registry.jsonl", source_rows)
    _write_jsonl(training_data_path, records)
    _write_jsonl(train_path, splits["train"])
    _write_jsonl(validation_path, splits["validation"])
    _write_jsonl(test_path, splits["test"])

    manifest = {
        "schema_version": "aiwf-ai-model-dataset-v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset_name": dataset_name,
        "output_dir": str(output_dir.resolve()),
        "record_count": len(records),
        "source_count": len(source_rows),
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "record_type_counts": dict(sorted(Counter(row["metadata"]["record_type"] for row in records).items())),
        "source_paths": [str(path.resolve()) for path in files],
        "privacy_policy": "Secret-like tokens are redacted. Large binaries, model folders, outputs, and venvs are skipped.",
        "training_format": "messages JSONL",
        "warnings": warnings,
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    _write_readme(output_dir, manifest)
    _validate_dataset_files(output_dir)

    return DatasetBuildResult(
        output_dir=output_dir.resolve(),
        manifest_path=manifest_path.resolve(),
        training_data_path=training_data_path.resolve(),
        train_path=train_path.resolve(),
        validation_path=validation_path.resolve(),
        test_path=test_path.resolve(),
        record_count=len(records),
        source_count=len(source_rows),
        warnings=tuple(warnings),
    )


def _parse_source_paths(source_paths: str | Iterable[str]) -> list[Path]:
    if isinstance(source_paths, str):
        raw_values = re.split(r"[\r\n]+", source_paths)
    else:
        raw_values = list(source_paths)
    paths: list[Path] = []
    for value in raw_values:
        text = str(value).strip().strip('"')
        if not text:
            continue
        paths.append(Path(text))
    return paths


def _collect_files(source_roots: list[Path], *, max_files: int) -> tuple[list[Path], list[str]]:
    files: list[Path] = []
    warnings: list[str] = []
    missing: list[str] = []
    for root in source_roots:
        if not root.exists():
            missing.append(str(root))
            continue
        if root.is_file():
            if _supported_file(root):
                files.append(root.resolve())
            continue
        for path in root.rglob("*"):
            if len(files) >= max_files:
                warnings.append(f"Stopped at max_files={max_files}; add narrower source folders for a focused dataset.")
                break
            if not path.is_file() or not _supported_file(path):
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
    deduped = sorted(dict.fromkeys(files), key=lambda item: str(item).lower())
    return deduped[:max_files], warnings


def _supported_file(path: Path) -> bool:
    if path.suffix.lower() not in SOURCE_EXTENSIONS:
        return False
    if path.stat().st_size > 2_000_000:
        return False
    return True


def _source_row(path: Path, index: int) -> dict[str, Any]:
    text = _read_text(path)
    return {
        "id": f"src-{index:04d}-{_slugify(path.stem)}",
        "path": str(path.resolve()),
        "name": path.name,
        "suffix": path.suffix.lower(),
        "sha256": _sha256_file(path),
        "byte_count": path.stat().st_size,
        "line_count": text.count("\n") + 1 if text else 0,
        "allowed_use": "local training dataset with provenance; review licenses before sharing externally",
    }


def _records_for_source(path: Path, source: dict[str, Any], *, max_chars_per_file: int) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = _records_from_existing_jsonl(path, source)
        if rows:
            return rows
    if suffix == ".json":
        rows = _records_from_existing_json(path, source)
        if rows:
            return rows
    if suffix == ".py":
        return [_code_summary_record(path, source), _excerpt_record(path, source, max_chars_per_file=max_chars_per_file)]
    return [_document_summary_record(path, source), _excerpt_record(path, source, max_chars_per_file=max_chars_per_file)]


def _records_from_existing_jsonl(path: Path, source: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            record = _normalise_training_row(row, source, suffix=f"line-{line_number}")
            if record:
                records.append(record)
    return records


def _records_from_existing_json(path: Path, source: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return []
    rows: list[dict[str, Any]]
    if isinstance(raw, list):
        rows = [row for row in raw if isinstance(row, dict)]
    elif isinstance(raw, dict):
        rows = []
        for key in ("train", "data", "examples", "rows"):
            value = raw.get(key)
            if isinstance(value, list):
                rows = [row for row in value if isinstance(row, dict)]
                break
        if not rows:
            rows = [raw]
    else:
        rows = []
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        record = _normalise_training_row(row, source, suffix=f"item-{index}")
        if record:
            records.append(record)
    return records


def _normalise_training_row(row: dict[str, Any], source: dict[str, Any], *, suffix: str) -> dict[str, Any] | None:
    messages = row.get("messages")
    if _valid_messages(messages):
        clean_messages = [
            {"role": str(message["role"]), "content": _redact(str(message["content"]))}
            for message in messages
        ]
    else:
        prompt = row.get("prompt") or row.get("instruction") or row.get("input")
        completion = row.get("completion") or row.get("response") or row.get("output")
        if isinstance(prompt, str) and isinstance(completion, str):
            clean_messages = [
                {"role": "user", "content": _redact(prompt)},
                {"role": "assistant", "content": _redact(completion)},
            ]
        elif isinstance(row.get("text"), str):
            clean_messages = [
                {"role": "user", "content": f"Study this local source excerpt from `{source['name']}`."},
                {"role": "assistant", "content": _redact(str(row["text"]))},
            ]
        else:
            return None
    return _training_record(
        record_id=f"rec-{source['id']}-{suffix}",
        messages=clean_messages,
        source=source,
        record_type="imported_training_row",
    )


def _code_summary_record(path: Path, source: dict[str, Any]) -> dict[str, Any]:
    text = _read_text(path)
    try:
        tree = ast.parse(text)
        docstring = (ast.get_docstring(tree) or "").strip().splitlines()[0:1]
        classes = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
        functions = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        summary = [
            f"`{path.name}` is Python source.",
            f"Module note: {docstring[0]}" if docstring else "",
            f"Classes: {', '.join(classes[:20])}." if classes else "",
            f"Functions: {', '.join(functions[:30])}." if functions else "",
            "Use the source path and hash in metadata as provenance before making exact claims.",
        ]
        answer = " ".join(part for part in summary if part)
    except SyntaxError as exc:
        answer = f"`{path.name}` is Python source but failed AST parsing: {exc}."
    return _training_record(
        record_id=f"rec-{source['id']}-code-summary",
        messages=[
            {"role": "user", "content": f"What should an AI coding assistant learn from `{path.name}`?"},
            {"role": "assistant", "content": answer},
        ],
        source=source,
        record_type="code_summary",
    )


def _document_summary_record(path: Path, source: dict[str, Any]) -> dict[str, Any]:
    text = _read_text(path)
    headings = [
        line.strip().lstrip("#").strip()
        for line in text.splitlines()
        if line.strip().startswith("#")
    ][:20]
    if headings:
        answer = f"`{path.name}` contains these main headings: {', '.join(headings)}."
    else:
        first_lines = " ".join(line.strip() for line in text.splitlines()[:8] if line.strip())
        answer = f"`{path.name}` is local text/source material. Opening excerpt: {_redact(first_lines[:800])}"
    answer += " Keep provenance from the source registry when training or reviewing this material."
    return _training_record(
        record_id=f"rec-{source['id']}-document-summary",
        messages=[
            {"role": "user", "content": f"What does `{path.name}` cover?"},
            {"role": "assistant", "content": answer},
        ],
        source=source,
        record_type="document_summary",
    )


def _excerpt_record(path: Path, source: dict[str, Any], *, max_chars_per_file: int) -> dict[str, Any]:
    text = _redact(_read_text(path)[:max_chars_per_file])
    return _training_record(
        record_id=f"rec-{source['id']}-bounded-excerpt",
        messages=[
            {"role": "user", "content": f"Preserve a bounded local training excerpt from `{path.name}`."},
            {"role": "assistant", "content": text},
        ],
        source=source,
        record_type="bounded_source_excerpt",
    )


def _training_record(
    *,
    record_id: str,
    messages: list[dict[str, str]],
    source: dict[str, Any],
    record_type: str,
) -> dict[str, Any]:
    return {
        "id": record_id,
        "messages": messages,
        "metadata": {
            "record_type": record_type,
            "source_id": source["id"],
            "source_path": source["path"],
            "source_sha256": source["sha256"],
            "split": "train",
        },
    }


def _assign_splits(records: list[dict[str, Any]]) -> None:
    for index, record in enumerate(records):
        bucket = index % 10
        if bucket == 8:
            split = "validation"
        elif bucket == 9:
            split = "test"
        else:
            split = "train"
        record["metadata"]["split"] = split


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
        "# AI Model Training Dataset\n\n"
        "Generated by AIWF Studio's pure-stdlib dataset builder.\n\n"
        f"- Records: {manifest['record_count']}\n"
        f"- Sources: {manifest['source_count']}\n"
        "- Format: chat-style `messages` JSONL.\n"
        "- Use `training_data.jsonl` or `splits/train.jsonl` in the AI Bot Trainer.\n",
        encoding="utf-8",
    )


def _validate_dataset_files(output_dir: Path) -> None:
    for path in output_dir.rglob("*.jsonl"):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if not raw.strip():
                    continue
                row = json.loads(raw)
                if path.name != "source_registry.jsonl":
                    messages = row.get("messages")
                    if not _valid_messages(messages):
                        raise ValueError(f"{path}:{line_number} is missing valid messages.")
    for path in output_dir.rglob("*.json"):
        json.loads(path.read_text(encoding="utf-8"))


def _valid_messages(messages: Any) -> bool:
    if not isinstance(messages, list) or not messages:
        return False
    return all(
        isinstance(message, dict)
        and isinstance(message.get("role"), str)
        and isinstance(message.get("content"), str)
        and bool(message.get("content", "").strip())
        for message in messages
    )


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _redact(value: str) -> str:
    text = value.replace("\r\n", "\n")
    for pattern in SECRET_PATTERNS:
        text = pattern.sub("[REDACTED_SECRET]", text)
    return text


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return slug[:80] or "dataset"
