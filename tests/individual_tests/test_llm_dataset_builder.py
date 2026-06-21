from __future__ import annotations

import json
from pathlib import Path

from aiwf.services.training.llm_dataset_builder import build_ai_model_dataset


def test_build_ai_model_dataset_from_code_and_markdown(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "module.py").write_text(
        '"""Example module."""\n\nclass Tool:\n    pass\n\ndef run():\n    return 1\n',
        encoding="utf-8",
    )
    (source / "README.md").write_text("# Demo\n\nThis is a demo dataset source.\n", encoding="utf-8")

    result = build_ai_model_dataset(
        str(source),
        dataset_name="demo",
        output_root=tmp_path / "out",
        max_files=10,
        max_chars_per_file=1000,
    )

    assert result.record_count >= 4
    assert result.source_count == 2
    assert result.training_data_path.is_file()
    rows = [
        json.loads(line)
        for line in result.training_data_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows
    assert all("messages" in row for row in rows)
    assert all(row["metadata"]["source_sha256"] for row in rows)


def test_build_ai_model_dataset_imports_existing_messages_jsonl(tmp_path: Path):
    source = tmp_path / "training.jsonl"
    source.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "Answer"},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = build_ai_model_dataset(str(source), dataset_name="existing", output_root=tmp_path / "out")

    rows = [
        json.loads(line)
        for line in result.training_data_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert result.record_count == 1
    assert rows[0]["messages"][0]["content"] == "Question"
