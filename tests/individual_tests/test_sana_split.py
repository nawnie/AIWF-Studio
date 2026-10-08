from __future__ import annotations

import json
import http.client
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from aiwf.infrastructure.distributed.sana_encoder_server import create_sana_encoder_app
from aiwf.infrastructure.distributed.sana_split import (
    SANA_SPLIT_MEDIA_TYPE,
    SANA_SPLIT_PROTOCOL_VERSION,
    SanaEncoding,
    SanaEncoderClient,
    SanaSplitError,
    pack_sana_encoding,
    sana_model_fingerprint,
    unpack_sana_encoding,
)


TOKEN = "test-token-with-at-least-24-characters"


class DisconnectingOpener:
    def open(self, request, timeout):  # noqa: ANN001, ANN201
        raise http.client.RemoteDisconnected("startup race")


class FakeRuntime:
    model_fingerprint = "a" * 64

    def encode(self, prompt: str, *, num_images_per_prompt: int, max_sequence_length: int) -> SanaEncoding:
        assert prompt == "small test"
        assert num_images_per_prompt == 1
        assert max_sequence_length == 16
        return SanaEncoding(
            prompt_embeds=torch.ones((1, 16, 8), dtype=torch.bfloat16),
            prompt_attention_mask=torch.ones((1, 16), dtype=torch.int64),
            model_fingerprint=self.model_fingerprint,
        )

    def health(self) -> dict[str, object]:
        return {
            "ready": True,
            "protocol_version": SANA_SPLIT_PROTOCOL_VERSION,
            "model_fingerprint": self.model_fingerprint,
            "disk_offload": False,
        }


def test_sana_encoding_round_trip_stays_in_memory():
    embeds = torch.randn((1, 12, 16), dtype=torch.bfloat16)
    mask = torch.ones((1, 12), dtype=torch.int64)
    payload = pack_sana_encoding(embeds, mask, model_fingerprint="b" * 64)
    result = unpack_sana_encoding(payload, model_fingerprint="b" * 64)
    assert torch.equal(result.prompt_embeds, embeds)
    assert torch.equal(result.prompt_attention_mask, mask)


def test_sana_encoding_rejects_wrong_tensor_contract():
    from safetensors.torch import save

    with pytest.raises(SanaSplitError, match="unexpected tensor contract"):
        unpack_sana_encoding(save({"wrong": torch.ones(1)}), model_fingerprint="c" * 64)


def test_encoder_client_wraps_transient_disconnect_for_retry():
    client = SanaEncoderClient("http://127.0.0.1:18794", TOKEN, opener=DisconnectingOpener())
    with pytest.raises(SanaSplitError, match="startup race"):
        client.health()


def test_encoder_api_requires_token_and_returns_safetensors():
    encoded = []
    client = TestClient(
        create_sana_encoder_app(FakeRuntime(), token=TOKEN, on_encode=lambda: encoded.append(True))
    )
    assert client.get("/healthz").status_code == 401
    auth = {"Authorization": f"Bearer {TOKEN}"}
    health = client.get("/healthz", headers=auth)
    assert health.status_code == 200
    assert health.json()["disk_offload"] is False

    response = client.post(
        "/api/v1/encode/sana-sprint",
        headers=auth,
        json={"prompt": "small test", "max_sequence_length": 16},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith(SANA_SPLIT_MEDIA_TYPE)
    assert response.headers["x-aiwf-protocol-version"] == SANA_SPLIT_PROTOCOL_VERSION
    result = unpack_sana_encoding(
        response.content,
        model_fingerprint=response.headers["x-aiwf-model-fingerprint"],
    )
    assert result.prompt_embeds.shape == (1, 16, 8)
    assert result.prompt_attention_mask.shape == (1, 16)
    assert encoded == [True]


def test_model_fingerprint_uses_snapshot_contract(tmp_path: Path):
    root = tmp_path / "sana"
    (root / "text_encoder").mkdir(parents=True)
    (root / "tokenizer").mkdir()
    (root / "model_index.json").write_text(
        json.dumps({"_class_name": "SanaSprintPipeline"}),
        encoding="utf-8",
    )
    (root / "text_encoder" / "config.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer" / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (root / "text_encoder" / "model-00001-of-00001.safetensors").write_bytes(b"weights")

    first = sana_model_fingerprint(root)
    second = sana_model_fingerprint(root)
    assert first == second
    assert len(first) == 64
