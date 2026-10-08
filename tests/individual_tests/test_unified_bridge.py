"""Unified Studio bridge: capability status, shared project ledger and the four cross-app flows.

Dataset Studio, ReTrain and Qwen Chat are replaced by one in-memory fake behind
httpx.MockTransport, so these tests need no running services, models or GPU.
The live end-to-end run against the real services is recorded separately.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image, PngImagePlugin

from aiwf.services.unified_bridge import BridgeConfig, BridgeError, UnifiedBridge
from aiwf.web.unified_api import build_unified_router


DATASET_URL = "http://127.0.0.1:8796"
RETRAIN_URL = "http://127.12.6.3:8787"
QWEN_URL = "http://127.0.0.1:8080"
COMFY_URL = "http://127.0.0.1:8188"


# --- fixture package that matches Dataset Studio's export format --------------------------
def _package(name: str = "fixture text pack") -> tuple[bytes, str]:
    train = b'{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]}\n'
    manifest = {
        "schema": "retrain-gui-recipe-dataset-v1",
        "package_name": name,
        "provenance": {"source_type": "synthetic", "purpose": "fixture-only UI and bridge verification"},
        "contract": {"format": "messages", "modality": "text-only", "image_training_supported": False},
        "files": [{"path": "train.jsonl", "sha256": hashlib.sha256(train).hexdigest(), "row_count": 1}],
        "counts": {"assets": 1, "rows": 1, "train": 1, "validation": 0},
        "assets": [{"asset_id": 7, "modality": "text"}],
        "row_assignments": [{"split": "train", "row_sha256": "0" * 64}],
    }
    digest = hashlib.sha256(json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest["revision"] = {"version": 1, "id": f"sha256:{digest}", "manifest_sha256": digest, "immutable": True}
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("train.jsonl", train)
    return stream.getvalue(), digest


# --- one fake for all three sibling apps ---------------------------------------------------
class FakeApps:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root
        self.calls: list[tuple[str, str, str]] = []
        self.down: set[str] = set()
        self.retrain_capability_status = 200
        self.package_bytes, self.package_hash = _package()
        self.retrain_reports_hash: str | None = None
        self.preflight_start_enabled = False
        self.assets: dict[int, dict] = {}
        self.labels: list[tuple[int, str]] = []
        self.collections: dict[int, list[int]] = {}
        self.qwen_payloads: list[dict] = []
        self.retrain_upload: bytes | None = None
        self.preflight_bodies: list[dict] = []
        self.weights_has_key = False
        self.hf_tokens_seen: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        host = f"{request.url.scheme}://{request.url.host}:{request.url.port}"
        path = request.url.path
        self.calls.append((host, request.method, path))
        if host in self.down:
            raise httpx.ConnectError("refused", request=request)
        if host == DATASET_URL:
            return self._dataset(request, path)
        if host == RETRAIN_URL:
            return self._retrain(request, path)
        if host == QWEN_URL:
            return self._qwen(request, path)
        return httpx.Response(404)

    # this section imitates Dataset Studio's authenticated API
    def _dataset(self, request: httpx.Request, path: str) -> httpx.Response:
        if request.headers.get("authorization") != "Bearer ds-token":
            return httpx.Response(401, json={"detail": "Local API authentication required."})
        if path == "/api/health":
            return httpx.Response(200, json={"status": "ok", "allowed_roots": [str(self.output_root)]})
        if path == "/api/retrain/exports" and request.method == "GET":
            return httpx.Response(200, json={"schema_version": "1", "packages": [
                {"package_name": "fixture text pack", "status": "ready", "manifest_sha256": self.package_hash, "revision_id": f"sha256:{self.package_hash}",
                 "created_at": "2026-10-06T00:00:00Z", "counts": {"train": 1, "validation": 0}, "modality": "text-only",
                 "download_url": "/api/retrain/exports/fixture%20text%20pack/download"},
                {"package_name": "broken", "status": "invalid", "reason": "Manifest revision hash does not match its contents."},
            ]})
        if path == "/api/retrain/exports/fixture text pack/download":
            return httpx.Response(200, content=self.package_bytes, headers={"content-type": "application/zip"})
        if path == "/api/assets/import":
            ids = []
            for raw in json.loads(request.content)["paths"]:
                asset_id = len(self.assets) + 1
                self.assets[asset_id] = {"id": asset_id, "path": raw, "caption": None, "revision": 1}
                ids.append(asset_id)
            return httpx.Response(200, json={"asset_ids": ids})
        if path == "/api/assets/labels":
            body = json.loads(request.content)
            self.labels.extend((asset_id, body["value"]) for asset_id in body["asset_ids"])
            return httpx.Response(200, json=body)
        if path.startswith("/api/assets/") and path.endswith("/labels"):
            asset_id = int(path.split("/")[3])
            self.labels.append((asset_id, json.loads(request.content)["value"]))
            return httpx.Response(200, json=self.assets[asset_id])
        if path.startswith("/api/assets/") and path.endswith("/caption"):
            asset_id = int(path.split("/")[3])
            body = json.loads(request.content)
            assert body["source"] == "import" and body["expected_revision"] == self.assets[asset_id]["revision"]
            self.assets[asset_id]["caption"] = {"text": body["text"], "source": "import"}
            return httpx.Response(200, json=self.assets[asset_id])
        if path.startswith("/api/assets/"):
            return httpx.Response(200, json=self.assets[int(path.split("/")[3])])
        if path == "/api/collections":
            self.collections.setdefault(1, [])
            return httpx.Response(200, json={"id": 1, "name": json.loads(request.content)["name"]})
        if path == "/api/collections/1/assets":
            self.collections[1].extend(json.loads(request.content)["asset_ids"])
            return httpx.Response(200, json={"id": 1})
        return httpx.Response(404, json={"detail": "Not Found"})

    # this section imitates ReTrain's import/preflight routes
    def _retrain(self, request: httpx.Request, path: str) -> httpx.Response:
        # ReTrain's model-weights manager (catalog, Hugging Face key, downloads)
        if path.startswith("/api/retrain/models/weights"):
            sub = path[len("/api/retrain/models/weights"):]
            body = json.loads(request.content) if request.content else {}
            if sub == "" and request.method == "GET":
                return httpx.Response(200, json={"schema_version": "1", "has_key": self.weights_has_key, "models": [
                    {"model_id": "qwen2.5-coder-1.5b", "label": "Qwen2.5-Coder 1.5B", "family": "qwen", "size_b": 1.5, "hf_repo": "Qwen/Qwen2.5-Coder-1.5B-Instruct", "present": True, "status": "present", "text_capable": True, "download": None},
                    {"model_id": "qwen2.5-7b", "label": "Qwen2.5 7B", "family": "qwen", "size_b": 7.0, "hf_repo": "Qwen/Qwen2.5-7B-Instruct", "present": False, "status": "missing", "text_capable": True, "download": None,
                     "access": {"state": "open", "gated": False, "has_key": self.weights_has_key, "download_bytes": 15242788168, "file_count": 11, "message": "", "folder": "C:/secret"}},
                    {"model_id": "gated-3b", "label": "Gated 3B", "family": "llama", "size_b": 3.0, "hf_repo": "Meta/Gated-3B", "present": False, "status": "missing", "text_capable": True, "download": None,
                     "access": {"state": "gated_ok" if self.weights_has_key else "gated_no_key", "gated": True, "has_key": self.weights_has_key, "download_bytes": 6000000000, "file_count": 5, "message": "This model is gated."}},
                ]})
            if sub == "/hf-token":
                self.hf_tokens_seen.append(body.get("token"))
                if body.get("token") != "hf_" + "a" * 30:
                    return httpx.Response(422, json={"detail": "Hugging Face rejected that token. Check it and try again."})
                self.weights_has_key = True
                return httpx.Response(200, json={"ok": True, "user": "tester"})
            if sub == "/hf-token/clear":
                self.weights_has_key = False
                return httpx.Response(200, json={"ok": True})
            if sub == "/download":
                if body.get("modelId") == "gated-3b" and not self.weights_has_key:
                    return httpx.Response(403, json={"detail": "This model is gated: accept its license on Hugging Face and save your key."})
                if body.get("modelId") == "qwen2.5-coder-1.5b":
                    return httpx.Response(409, json={"detail": "These weights are already on this PC."})
                return httpx.Response(200, json={"job_id": "0123456789ab", "model_id": body["modelId"], "repo": "Qwen/Qwen2.5-7B-Instruct", "status": "running", "total_bytes": 100, "done_bytes": 10, "files_total": 3, "files_done": 0, "message": "", "destination_name": "Qwen--Qwen2.5-7B-Instruct", "process": "secret"})
            if sub.startswith("/download/"):
                cancelled = sub.endswith("/cancel")
                return httpx.Response(200, json={"job_id": "0123456789ab", "model_id": "qwen2.5-7b", "repo": "Qwen/Qwen2.5-7B-Instruct", "status": "cancelled" if cancelled else "completed", "total_bytes": 100, "done_bytes": 100, "files_total": 3, "files_done": 3, "message": "", "destination_name": "Qwen--Qwen2.5-7B-Instruct"})
            return httpx.Response(404, json={"detail": "Not Found"})
        if path == "/api/retrain/datasets/import-capability":
            if self.retrain_capability_status != 200:
                return httpx.Response(self.retrain_capability_status, json={"detail": "nope"})
            return httpx.Response(200, json={"schema_version": "1", "package_schema": "retrain-gui-recipe-dataset-v1",
                                             "models": [{"model_id": "qwen2.5-coder-1.5b", "label": "Qwen2.5-Coder 1.5B", "family": "qwen", "text_capable": True, "path": "C:/secret"}]})
        if path == "/api/retrain/datasets/import-package":
            assert request.headers["content-type"] == "application/zip"
            self.retrain_upload = request.read()
            digest = self.retrain_reports_hash or self.package_hash
            return httpx.Response(200, json={"status": "imported", "dataset_id": f"sha256-{digest}", "manifest_sha256": digest,
                                             "package_name": "fixture text pack", "counts": {"train": 1, "validation": 0}, "modality": "text-only", "reused": False})
        if path.endswith("/preflight"):
            self.preflight_bodies.append(json.loads(request.content))
            # shaped like the real ReTrain plan observed live on 2026-10-06, including server paths
            secret_dir = "F:" + chr(92) + "_Projects" + chr(92) + "ReTrain" + chr(92) + "datasets"
            plan = {
                "status": "blocked",
                "summary": {"model": "Qwen2.5-Coder 1.5B", "fit_state": "safe"},
                "training_args": {"lora_rank": 16, "tensorboard_logdir": secret_dir},
                "validation": {
                    "config": {"dataset_path": secret_dir, "output_root": secret_dir},
                    "gates": [{"gate": "Base model", "state": "warning", "detail": "pick or download a local model"},
                              {"gate": "Dataset path", "state": "ready", "detail": secret_dir}],
                    "estimate": {"fit_state": "safe", "estimated_gb": 2.3, "limit_gb": 16.0, "warnings": []},
                    "dependencies": [{"package": "torch", "label": "PyTorch", "available": True}],
                    "notes": ["Keep dry run enabled."],
                },
            }
            return httpx.Response(200, json={"status": "preflight", "dataset": {"manifest_sha256": self.package_hash}, "plan": plan,
                                             "start_enabled": self.preflight_start_enabled, "execution_requested": False,
                                             "message": "Preflight only. This route cannot start training."})
        return httpx.Response(404, json={"detail": "Not Found"})

    # this section imitates Qwen Chat's OpenAI-compatible llama-server
    def _qwen(self, request: httpx.Request, path: str) -> httpx.Response:
        if request.headers.get("authorization") != "Bearer qwen-key":
            return httpx.Response(401, json={"error": "invalid api key"})
        if path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "qwen3-8b", "status": {"value": "loaded"}}, {"id": "bonsai-2-27b", "status": {"value": "unloaded"}}]})
        if path == "/v1/chat/completions":
            self.qwen_payloads.append(json.loads(request.content))
            return httpx.Response(200, json={"model": "qwen3-8b", "choices": [{"message": {"role": "assistant", "content": "Looks consistent."}}]})
        return httpx.Response(404)


@pytest.fixture()
def env(tmp_path: Path):
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    (tmp_path / "ds-token.txt").write_text("ds-token\n", encoding="utf-8")
    (tmp_path / "qwen-key.txt").write_text("# comment\n\nqwen-key\nsecond-line\n", encoding="utf-8")
    fake = FakeApps(output_root)
    config = BridgeConfig(
        dataset_studio_url=DATASET_URL,
        dataset_studio_token_file=tmp_path / "ds-token.txt",
        retrain_url=RETRAIN_URL,
        qwen_chat_url=QWEN_URL,
        qwen_chat_key_file=tmp_path / "qwen-key.txt",
        projects_dir=tmp_path / "data" / "_local" / "unified-projects",
    )
    bridge = UnifiedBridge(config, transport=httpx.MockTransport(fake.handler))
    return SimpleNamespace(bridge=bridge, fake=fake, output_root=output_root, tmp=tmp_path)


def _studio_png(root: Path, name: str, prompt: str = "a lighthouse at dusk") -> Path:
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", f"{prompt}\nSteps: 20, Sampler: Euler, Seed: 42, Model: sdxl-base")
    path = root / name
    Image.new("RGB", (8, 8), (10, 20, 30)).save(path, pnginfo=info)
    return path


def _provenance(path: Path) -> dict:
    from aiwf.web.pro_api import _read_output_infotext, _settings_from_infotext

    return _settings_from_infotext(_read_output_infotext(path))


# --- capability status -------------------------------------------------------------------
def test_status_reports_every_app_not_running_without_guessing(env) -> None:
    env.fake.down = {DATASET_URL, RETRAIN_URL, QWEN_URL, COMFY_URL}
    caps = env.bridge.status(studio_output_root=env.output_root)["capabilities"]
    assert {name: cap["state"] for name, cap in caps.items()} == {"dataset_studio": "not_running", "retrain": "not_running", "qwen_chat": "not_running", "comfyui": "not_running"}
    assert all(cap["available"] is False and cap["reason"] for cap in caps.values())


def test_status_ready_when_all_apps_answer(env) -> None:
    caps = env.bridge.status(studio_output_root=env.output_root)["capabilities"]
    assert caps["dataset_studio"]["state"] == "ready" and caps["dataset_studio"]["catalog_studio_outputs"] is True
    assert caps["retrain"]["available"] is True and caps["retrain"]["training_start_available"] is False
    assert caps["retrain"]["models"][0]["model_id"] == "qwen2.5-coder-1.5b"
    assert "path" not in caps["retrain"]["models"][0]
    assert caps["qwen_chat"]["models"] == [{"model_id": "qwen3-8b", "loaded": True}, {"model_id": "bonsai-2-27b", "loaded": False}]


def test_status_distinguishes_protected_service_and_missing_route(env) -> None:
    env.fake.retrain_capability_status = 401
    assert env.bridge.status()["capabilities"]["retrain"]["state"] == "auth_rejected"
    env.fake.retrain_capability_status = 404
    assert env.bridge.status()["capabilities"]["retrain"]["state"] == "route_missing"


def test_status_partial_when_studio_outputs_are_outside_catalog_roots(env) -> None:
    caps = env.bridge.status(studio_output_root=env.tmp / "elsewhere")["capabilities"]
    assert caps["dataset_studio"]["available"] is True
    assert caps["dataset_studio"]["state"] == "partial" and caps["dataset_studio"]["catalog_studio_outputs"] is False


def test_missing_token_and_non_loopback_config_fail_closed(env) -> None:
    (env.tmp / "ds-token.txt").unlink()
    assert env.bridge.status()["capabilities"]["dataset_studio"]["state"] == "not_configured"
    remote = UnifiedBridge(BridgeConfig(**{**env.bridge.config.__dict__, "retrain_url": "http://192.168.1.20:8787"}), transport=httpx.MockTransport(env.fake.handler))
    assert remote.status()["capabilities"]["retrain"]["state"] == "not_configured"
    assert not any(call[0].startswith("http://192.168") for call in env.fake.calls)


# --- shared project ledger -------------------------------------------------------------------
def test_projects_create_list_get_and_reject_bad_ids(env) -> None:
    created = env.bridge.create_project("Lighthouse set")["project"]
    assert created["project_id"].startswith("aiwfp-")
    assert [item["name"] for item in env.bridge.list_projects()["projects"]] == ["Lighthouse set"]
    assert env.bridge.get_project(created["project_id"])["project"]["events"] == []
    with pytest.raises(BridgeError) as bad:
        env.bridge.get_project("../../etc")
    assert bad.value.status_code == 422
    with pytest.raises(BridgeError):
        env.bridge.create_project("bad\x00name")


# --- flow 1: Studio outputs -> Dataset Studio catalog -----------------------------------------------
def test_catalog_outputs_tags_project_and_keeps_provenance_without_absolute_paths(env) -> None:
    project = env.bridge.create_project("Lighthouse set")["project"]
    _studio_png(env.output_root, "one.png")
    result = env.bridge.catalog_outputs(project["project_id"], ["one.png"], output_root=env.output_root, read_provenance=_provenance)

    assert result["status"] == "cataloged"
    assert result["assets"][0]["caption_written"] is True
    assert (1, f"aiwf-project:{project['project_id']}") in env.fake.labels
    assert (1, "aiwf-model:sdxl-base") in env.fake.labels
    assert env.fake.collections[1] == [1]
    assert env.fake.assets[1]["caption"]["text"] == "a lighthouse at dusk"
    ledger = env.bridge.get_project(project["project_id"])["project"]["events"][0]
    assert ledger["kind"] == "dataset_catalog"
    assert ledger["outputs"][0]["seed"] == 42 and len(ledger["outputs"][0]["sha256"]) == 64
    assert str(env.output_root) not in json.dumps(ledger)


def test_catalog_outputs_rejects_paths_outside_the_output_folder(env) -> None:
    project = env.bridge.create_project("p")["project"]
    (env.tmp / "secret.png").write_bytes(b"x")
    with pytest.raises(BridgeError) as escaped:
        env.bridge.catalog_outputs(project["project_id"], ["../secret.png"], output_root=env.output_root, read_provenance=_provenance)
    assert escaped.value.status_code == 404
    assert not any(call[2] == "/api/assets/import" for call in env.fake.calls)


# --- flow 2: explicit package -> verified ReTrain import ------------------------------------------
def test_package_list_forwards_display_fields_only(env) -> None:
    packages = env.bridge.list_packages()["packages"]
    assert packages[0]["manifest_sha256"] == env.fake.package_hash
    assert "download_url" not in packages[0]
    assert packages[1] == {"package_name": "broken", "status": "invalid", "reason": "Manifest revision hash does not match its contents."}


def test_import_transfers_exact_selected_revision(env) -> None:
    project = env.bridge.create_project("p")["project"]
    result = env.bridge.import_package(project["project_id"], "fixture text pack", env.fake.package_hash)
    assert result["dataset"]["dataset_id"] == f"sha256-{env.fake.package_hash}"
    assert env.fake.retrain_upload == env.fake.package_bytes
    assert env.bridge.get_project(project["project_id"])["project"]["events"][-1]["kind"] == "retrain_import"


def test_import_refuses_a_stale_selection_before_calling_retrain(env) -> None:
    project = env.bridge.create_project("p")["project"]
    with pytest.raises(BridgeError) as stale:
        env.bridge.import_package(project["project_id"], "fixture text pack", "a" * 64)
    assert stale.value.status_code == 409 and stale.value.code == "stale_revision"
    assert env.fake.retrain_upload is None


def test_import_rejects_retrain_storing_a_different_revision(env) -> None:
    project = env.bridge.create_project("p")["project"]
    env.fake.retrain_reports_hash = "b" * 64
    with pytest.raises(BridgeError) as mismatch:
        env.bridge.import_package(project["project_id"], "fixture text pack", env.fake.package_hash)
    assert mismatch.value.code == "revision_mismatch"
    assert env.bridge.get_project(project["project_id"])["project"]["events"] == []


# --- flow 3: preflight dry run bound to the imported revision ---------------------------------------
def test_preflight_is_bound_to_revision_and_filters_settings(env) -> None:
    project = env.bridge.create_project("p")["project"]
    digest = env.fake.package_hash
    with pytest.raises(BridgeError) as unbound:
        env.bridge.preflight(project["project_id"], "sha256-" + "c" * 64, digest, "qwen2.5-coder-1.5b", {})
    assert unbound.value.code == "revision_binding"
    result = env.bridge.preflight(project["project_id"], f"sha256-{digest}", digest, "qwen2.5-coder-1.5b",
                                  {"method": "QLoRA", "confirmed": True, "payload": {"dataset_path": "C:/x"}})
    assert result["start_enabled"] is False and result["result"]["plan_status"] == "blocked"
    assert env.fake.preflight_bodies[-1] == {"modelId": "qwen2.5-coder-1.5b", "settings": {"method": "QLoRA"}}
    # absolute ReTrain paths never reach the browser
    serialized = json.dumps(result)
    assert "_Projects" not in serialized and "config" not in result["result"]
    assert {"gate": "Dataset path", "state": "ready", "detail": "<server path>"} in result["result"]["gates"]
    assert result["result"]["estimate"]["fit_state"] == "safe"
    assert "tensorboard_logdir" not in result["result"]["training_args"]


def test_preflight_discards_a_result_that_could_start_training(env) -> None:
    project = env.bridge.create_project("p")["project"]
    env.fake.preflight_start_enabled = True
    digest = env.fake.package_hash
    with pytest.raises(BridgeError) as unsafe:
        env.bridge.preflight(project["project_id"], f"sha256-{digest}", digest, "qwen2.5-coder-1.5b", {})
    assert unsafe.value.code == "unsafe_preflight"


def test_synthetic_demo_package_can_be_imported_and_planned_without_training(env) -> None:
    """The demo uses an in-memory Dataset Studio/ReTrain fixture, never the user's catalog."""
    project = env.bridge.create_project("Synthetic Train demo")["project"]
    digest = env.fake.package_hash
    manifest = json.loads(zipfile.ZipFile(io.BytesIO(env.fake.package_bytes)).read("manifest.json"))
    assert manifest["provenance"]["source_type"] == "synthetic"
    imported = env.bridge.import_package(project["project_id"], "fixture text pack", digest)["dataset"]
    plan = env.bridge.preflight(project["project_id"], imported["dataset_id"], digest, "qwen2.5-coder-1.5b", {"method": "QLoRA"})
    assert plan["start_enabled"] is False and plan["execution_requested"] is False
    assert plan["result"]["manifest_sha256"] == digest
    assert [event["kind"] for event in env.bridge.get_project(project["project_id"])["project"]["events"]] == ["retrain_import", "retrain_preflight"]


# --- flow 4: explicit project context -> Qwen Chat ------------------------------------------------
def test_qwen_receives_only_the_context_card_and_question(env) -> None:
    project = env.bridge.create_project("Lighthouse set")["project"]
    _studio_png(env.output_root, "one.png")
    env.bridge.catalog_outputs(project["project_id"], ["one.png"], output_root=env.output_root, read_provenance=_provenance)
    env.bridge.import_package(project["project_id"], "fixture text pack", env.fake.package_hash)
    preview = env.bridge.qwen_context(project["project_id"])["context"]
    answer = env.bridge.qwen_ask(project["project_id"], "qwen3-8b", "Is this project ready for a dry run?")

    sent = env.fake.qwen_payloads[0]
    assert answer["answer"] == "Looks consistent." and answer["context_sent"] == preview
    assert sent["messages"][0]["content"].endswith(preview)
    assert sent["messages"][1] == {"role": "user", "content": "Is this project ready for a dry run?"}
    assert str(env.tmp) not in json.dumps(sent)
    assert "Studio outputs cataloged in Dataset Studio: 1" in preview and env.fake.package_hash[:12] in preview
    event = env.bridge.get_project(project["project_id"])["project"]["events"][-1]
    assert event["kind"] == "qwen_context_sent"
    assert event["question"] == "Is this project ready for a dry run?"


def test_streamed_context_question_is_recorded_without_storing_the_answer(env) -> None:
    project = env.bridge.create_project("Chat history")["project"]
    context = env.bridge.qwen_context(project["project_id"])
    result = env.bridge.record_context_question(project["project_id"], "qwen3-8b", "Which package should I review?", context["context_sha256"])
    event = env.bridge.get_project(project["project_id"])["project"]["events"][-1]
    assert result["event_id"] == event["event_id"]
    assert event["kind"] == "qwen_context_sent" and event["question"] == "Which package should I review?"
    assert event["answer_chars"] is None and "answer" not in event
    assert event["context_sha256"] == context["context_sha256"] and event["stream_returned"] is True


def test_qwen_context_lists_each_revision_and_preflight_once(env) -> None:
    project = env.bridge.create_project("p")["project"]
    digest = env.fake.package_hash
    for _ in range(3):
        env.bridge.import_package(project["project_id"], "fixture text pack", digest)
        env.bridge.preflight(project["project_id"], f"sha256-{digest}", digest, "qwen2.5-coder-1.5b", {})
    card = env.bridge.qwen_context(project["project_id"])["context"]
    assert card.count("- package 'fixture text pack'") == 1
    assert card.count("- preflight with qwen2.5-coder-1.5b") == 1


# --- model weights: catalog, key, downloads -----------------------------------------------------------
def test_model_catalog_forwards_display_fields_and_hides_folders(env) -> None:
    catalog = env.bridge.list_models()
    by_id = {item["model_id"]: item for item in catalog["models"]}
    assert by_id["qwen2.5-coder-1.5b"]["present"] is True and "access" not in by_id["qwen2.5-coder-1.5b"]
    assert by_id["qwen2.5-7b"]["access"]["state"] == "open" and by_id["qwen2.5-7b"]["access"]["download_bytes"] == 15242788168
    assert by_id["gated-3b"]["access"]["state"] == "gated_no_key"
    assert "folder" not in json.dumps(catalog) and "secret" not in json.dumps(catalog)


def test_gated_download_needs_the_key_and_errors_read_as_plain_text(env) -> None:
    with pytest.raises(BridgeError) as gated:
        env.bridge.download_model("gated-3b")
    assert gated.value.status_code == 403 and gated.value.code == "needs_permission"
    assert gated.value.message.startswith("This model is gated")
    with pytest.raises(BridgeError) as present:
        env.bridge.download_model("qwen2.5-coder-1.5b")
    assert present.value.status_code == 409 and "already on this PC" in present.value.message
    assert env.bridge.save_hf_token("hf_" + "a" * 30)["ok"] is True
    assert env.bridge.list_models()["has_key"] is True
    assert env.bridge.download_model("gated-3b")["download"]["status"] == "running"


def test_open_download_status_and_cancel_hide_internal_fields(env) -> None:
    started = env.bridge.download_model("qwen2.5-7b")["download"]
    assert started["job_id"] == "0123456789ab" and "process" not in started
    assert env.bridge.download_status("0123456789ab")["download"]["status"] == "completed"
    assert env.bridge.cancel_download("0123456789ab")["download"]["status"] == "cancelled"


def test_bad_token_is_rejected_and_never_reaches_the_ledger_or_responses(env) -> None:
    project = env.bridge.create_project("tokens")["project"]
    with pytest.raises(BridgeError) as bad:
        env.bridge.save_hf_token("hf_" + "z" * 30)
    assert bad.value.status_code == 422 and "hf_" + "z" * 30 not in bad.value.message
    env.bridge.save_hf_token("hf_" + "a" * 30)
    ledger = json.dumps(env.bridge.get_project(project["project_id"]))
    assert "hf_" + "a" * 30 not in ledger
    assert env.bridge.clear_hf_token()["ok"] is True and env.fake.weights_has_key is False


# --- HTTP routes ---------------------------------------------------------------------------
def _client(env, host: str = "127.0.0.1") -> TestClient:
    ctx = SimpleNamespace(flags=SimpleNamespace(data_dir=env.tmp / "data", resolved_output_dir=lambda: env.output_root))
    app = FastAPI()
    app.include_router(build_unified_router(ctx, bridge=env.bridge))
    return TestClient(app, client=(host, 50000))


def test_routes_round_trip_and_block_remote_mutation(env) -> None:
    client = _client(env)
    status = client.get("/api/pro/unified/status").json()
    assert status["schema_version"] == "1" and set(status["capabilities"]) == {"dataset_studio", "retrain", "qwen_chat", "comfyui"}
    project = client.post("/api/pro/unified/projects", json={"name": "Route test"}).json()["project"]
    context_sha256 = client.get(f"/api/pro/unified/projects/{project['project_id']}/qwen-context").json()["context_sha256"]
    _studio_png(env.output_root, "sub dir.png")
    catalog = client.post(f"/api/pro/unified/projects/{project['project_id']}/catalog-outputs",
                          json={"output_paths": ["/api/pro/outputs/sub%20dir.png"]})
    assert catalog.status_code == 200, catalog.text
    bad = client.post("/api/pro/unified/retrain/import", json={"project_id": project["project_id"], "package_name": "x", "manifest_sha256": "z"})
    assert bad.status_code == 422
    extra = client.post("/api/pro/unified/projects", json={"name": "x", "owner": "y"})
    assert extra.status_code == 422
    stale = client.post("/api/pro/unified/retrain/import", json={"project_id": project["project_id"], "package_name": "fixture text pack", "manifest_sha256": "a" * 64})
    assert stale.status_code == 409 and stale.json()["detail"]["code"] == "stale_revision"
    recorded = client.post(f"/api/pro/unified/projects/{project['project_id']}/chat-question",
                           json={"model_id": "qwen3-8b", "question": "Which fixture should I review?", "context_sha256": context_sha256})
    assert recorded.status_code == 200
    assert client.get(f"/api/pro/unified/projects/{project['project_id']}").json()["project"]["events"][-1]["question"] == "Which fixture should I review?"

    remote = _client(env, host="192.0.2.5")
    assert remote.post("/api/pro/unified/projects", json={"name": "phone"}).status_code == 403
    assert remote.post(f"/api/pro/unified/projects/{project['project_id']}/chat-question",
                       json={"model_id": "qwen3-8b", "question": "Which fixture should I review?", "context_sha256": context_sha256}).status_code == 403
    assert remote.get("/api/pro/unified/model-families").status_code == 200


# --- loopback-only launch switch ----------------------------------------------------------------
@pytest.mark.parametrize(("value", "expected"), [("1", True), ("true", True), ("", False), ("0", False)])
def test_loopback_only_switch(monkeypatch, value: str, expected: bool) -> None:
    from aiwf.app_pro import _loopback_only_requested

    monkeypatch.setenv("AIWF_PRO_LOOPBACK_ONLY", value)
    assert _loopback_only_requested() is expected


def test_model_routes_over_http_and_remote_callers_are_blocked(env) -> None:
    client = _client(env)
    assert client.get("/api/pro/unified/models").json()["models"][1]["access"]["state"] == "open"
    saved = client.post("/api/pro/unified/models/hf-token", json={"token": "hf_" + "a" * 30})
    assert saved.status_code == 200 and "hf_" + "a" * 30 not in saved.text
    assert client.get("/api/pro/unified/models").json()["has_key"] is True
    started = client.post("/api/pro/unified/models/download", json={"model_id": "qwen2.5-7b"})
    assert started.status_code == 200 and started.json()["download"]["status"] == "running"
    assert client.get("/api/pro/unified/models/downloads/0123456789ab").json()["download"]["status"] == "completed"
    assert client.post("/api/pro/unified/models/downloads/0123456789ab/cancel").json()["download"]["status"] == "cancelled"
    assert client.post("/api/pro/unified/models/download", json={"model_id": "qwen2.5-coder-1.5b"}).status_code == 409
    assert client.post("/api/pro/unified/models/hf-token", json={"token": "x", "extra": 1}).status_code == 422
    assert client.post("/api/pro/unified/models/hf-token/clear").json()["ok"] is True

    remote = _client(env, host="192.0.2.5")
    assert remote.get("/api/pro/unified/models").status_code == 200
    for path, body in (("/models/hf-token", {"token": "hf_" + "a" * 30}), ("/models/hf-token/clear", {}), ("/models/download", {"model_id": "qwen2.5-7b"}), ("/models/downloads/0123456789ab/cancel", {})):
        assert remote.post("/api/pro/unified" + path, json=body).status_code == 403, path
    # only the one local save reached the sibling; the blocked remote attempt never did
    assert env.fake.hf_tokens_seen == ["hf_" + "a" * 30]
