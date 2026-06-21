from __future__ import annotations

import json
from pathlib import Path

from aiwf.services.chat_atlas_rag import (
    build_atlas_rag_packet,
    format_cards_for_prompt,
    retrieve_atlas_cards,
)


def test_build_atlas_rag_packet_creates_cards_lanes_and_training_data(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("# Atlas Demo\n\nRetrieval card notes.\n", encoding="utf-8")
    (source / "service.py").write_text("def retrieve_cards():\n    return []\n", encoding="utf-8")

    result = build_atlas_rag_packet(
        str(source),
        packet_name="demo",
        output_root=tmp_path / "atlas",
        max_files=10,
    )

    assert result.card_count == 2
    assert result.lane_count >= 1
    assert result.cards_path.is_file()
    assert result.training_data_path.is_file()
    assert (result.output_dir / "cartographer_map.json").is_file()
    assert any((result.output_dir / "lanes").glob("*.jsonl"))

    records = [
        json.loads(line)
        for line in result.training_data_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert records
    assert records[0]["messages"][0]["role"] == "system"
    assert records[0]["metadata"]["record_type"] == "atlas_card_retrieval"


def test_retrieve_atlas_cards_scores_query_terms(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "rag_notes.md").write_text("# Retrieval\n\nAtlas lane retrieval and cards.\n", encoding="utf-8")
    (source / "ui_notes.md").write_text("# Interface\n\nTabs and buttons.\n", encoding="utf-8")
    result = build_atlas_rag_packet(str(source), packet_name="demo", output_root=tmp_path / "atlas")

    cards = retrieve_atlas_cards(result.output_dir, "How does Atlas retrieval use cards?", top_k=1)
    prompt = format_cards_for_prompt(cards)

    assert len(cards) == 1
    assert "Atlas retrieved cards" in prompt
    assert "retrieval" in prompt.lower()
