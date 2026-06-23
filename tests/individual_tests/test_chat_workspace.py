from __future__ import annotations

import json
from pathlib import Path

from aiwf.services.chat_atlas_rag import build_atlas_rag_packet
from aiwf.web.tabs.chat_workspace import (
    _atlas_context,
    _atlas_training_path,
    _build_atlas_qlora_request,
    _format_chat_signals,
)


def test_atlas_training_path_resolves_packet_training_data(tmp_path: Path):
    packet = tmp_path / "packet"
    packet.mkdir()
    data = packet / "training_data.jsonl"
    data.write_text("{}\n", encoding="utf-8")

    assert _atlas_training_path(str(packet)) == str(data)
    assert _atlas_training_path(str(data)) == str(data)


def test_format_chat_signals_summarizes_backend_models_and_trainer():
    summary = _format_chat_signals(alive=True, models=["qwen", "smol"], trainer_ready=False)

    assert "Backend:** online" in summary
    assert "Models:** 2" in summary
    assert "Trainer:** not ready" in summary


def test_build_atlas_qlora_request_is_for_qlora_messages_dataset(tmp_path: Path):
    data = tmp_path / "training_data.jsonl"
    data.write_text(json.dumps({"messages": [{"role": "user", "content": "x"}]}) + "\n", encoding="utf-8")

    req = _build_atlas_qlora_request(
        base_model_path="Qwen/Qwen3.5-2B",
        packet_path=str(data),
        adapter_name="atlas_adapter",
        output_dir="outputs/training/atlas",
        max_steps=50,
        batch_size=1,
        grad_accum=8,
        learning_rate=2e-5,
        max_seq_length=1024,
        lora_rank=16,
        lora_alpha=32,
    )

    assert req["method"] == "qlora"
    assert req["dataset_format"] == "messages"
    assert req["dataset_path"] == str(data)
    assert req["job_name"] == "atlas_adapter"


def test_atlas_context_returns_retrieved_cards(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "atlas.md").write_text("# Atlas Retrieval\n\nCard lane retrieval.\n", encoding="utf-8")
    result = build_atlas_rag_packet(str(source), output_root=tmp_path / "atlas")

    context = _atlas_context(str(result.output_dir), "retrieval cards", 2)

    assert "Atlas retrieved cards" in context
    assert "retrieval" in context.lower()
