"""Image jobs through the engine API: Qwen Image 2.1 on a fake ComfyUI.

ComfyUI is replaced by an in-memory fake behind httpx.MockTransport (the same
approach as test_unified_bridge.py), so no GPU, model or running ComfyUI is
needed. A live run against the real ComfyUI is recorded separately.
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from aiwf.services import unified_contract
from aiwf.services.image_jobs import ImageJobs
from aiwf.services.unified_bridge import BridgeConfig, BridgeError, UnifiedBridge
from aiwf.web.unified_api import build_unified_router

COMFY_URL = "http://127.0.0.1:8188"


# --- the fake image engine ---------------------------------------------------------------
class FakeComfy:
    """Queues prompts, runs them after a couple of polls, and serves the result PNG."""

    def __init__(self) -> None:
        self.workflows: list[dict] = []
        self.pending: list[str] = []
        self.running: str | None = None
        self.history: dict[str, dict] = {}
        self.polls_until_done = 2
        self.polls = 0
        self.reject_with: dict | None = None
        self.cancel_status = 200
        self.fail_execution = False
        self.forget_jobs = False
        self.deleted: list[str] = []
        self.interrupts: list[dict] = []
        self.down = False
        self.missing_nodes: set[str] = set()
        self.missing_models: set[str] = set()
        self.requests: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request.url.path)
        if self.down:
            raise httpx.ConnectError("refused", request=request)
        path = request.url.path
        if path == "/system_stats":
            return httpx.Response(200, json={"system": {"comfyui_version": "0.37.0"}, "devices": []})
        if path == "/object_info":
            nodes = {
                "ResolutionSelector", "SaveImageAdvanced", "UNETLoader", "TextEncodeQwenImage21",
                "CLIPLoader", "VAELoader", "EmptyLatentImage", "VAEDecode", "KSampler",
            } - self.missing_nodes
            return httpx.Response(200, json={name: {} for name in nodes})
        model_files = {
            "/models/diffusion_models": ["Qwen Image 2.1/qwen_image_2.1_int8_convrot.safetensors"],
            "/models/text_encoders": ["Qwen3-VL/5.7B int8/qwen3vl_8b_w4a8_heretic.safetensors"],
            "/models/vae": ["Qwen Image 2.1/qwen_image_2.1_vae_bf16.safetensors"],
        }
        if path in model_files:
            return httpx.Response(200, json=[name for name in model_files[path] if name not in self.missing_models])
        if path == "/prompt" and request.method == "POST":
            if self.reject_with is not None:
                return httpx.Response(400, json=self.reject_with)
            body = json.loads(request.content)
            self.workflows.append(body["prompt"])
            prompt_id = f"p{len(self.workflows):04d}"
            self.pending.append(prompt_id)
            return httpx.Response(200, json={"prompt_id": prompt_id, "number": len(self.workflows)})
        if path == "/queue" and request.method == "GET":
            self._advance()
            return httpx.Response(200, json={
                "queue_running": [[0, self.running, {}, {}, []]] if self.running else [],
                "queue_pending": [[i + 1, pid, {}, {}, []] for i, pid in enumerate(self.pending)],
            })
        if path == "/queue" and request.method == "POST":
            status = self.cancel_status
            if status >= 400:
                return httpx.Response(status, json={"error": "synthetic cancel failure"})
            for pid in json.loads(request.content).get("delete", []):
                self.deleted.append(pid)
                if pid in self.pending:
                    self.pending.remove(pid)
            return httpx.Response(200, json={})
        if path == "/interrupt":
            self.interrupts.append(json.loads(request.content or b"{}"))
            if self.cancel_status >= 400:
                return httpx.Response(self.cancel_status, json={"error": "synthetic cancel failure"})
            self.running = None
            return httpx.Response(200, json={})
        if path.startswith("/history/"):
            pid = path.rsplit("/", 1)[1]
            return httpx.Response(200, json={pid: self.history[pid]} if pid in self.history else {})
        if path == "/view":
            stream = io.BytesIO()
            Image.new("RGB", (64, 48), (200, 120, 40)).save(stream, format="PNG")
            return httpx.Response(200, content=stream.getvalue(), headers={"content-type": "image/png"})
        return httpx.Response(404)

    # this method moves the fake queue forward one step per /queue poll
    def _advance(self) -> None:
        if self.forget_jobs:
            self.pending.clear()
            self.running = None
            return
        if self.running is None and self.pending:
            self.running = self.pending.pop(0)
            self.polls = 0
            return
        if self.running is not None:
            self.polls += 1
            if self.polls >= self.polls_until_done:
                pid, self.running = self.running, None
                if self.fail_execution:
                    self.history[pid] = {"status": {"status_str": "error", "completed": False, "messages": [
                        ["execution_error", {"node_type": "UNETLoader", "exception_message": "CUDA out of memory.\nTried to allocate 2 GiB"}]]}, "outputs": {}}
                else:
                    self.history[pid] = {"status": {"status_str": "success", "completed": True, "messages": []},
                                         "outputs": {"461": {"images": [{"filename": "AIWF_Studio/qwen_image_2.1_00001_.png", "subfolder": "", "type": "output"}]}}}


@pytest.fixture()
def images_env(tmp_path: Path):
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    comfy = FakeComfy()
    config = BridgeConfig(
        dataset_studio_url="http://127.0.0.1:8796",
        dataset_studio_token_file=tmp_path / "missing-token.txt",
        retrain_url="http://127.12.6.3:8787",
        qwen_chat_url="http://127.0.0.1:8080",
        qwen_chat_key_file=tmp_path / "missing-key.txt",
        projects_dir=tmp_path / "data" / "_local" / "unified-projects",
        comfyui_url=COMFY_URL,
    )
    bridge = UnifiedBridge(config, transport=httpx.MockTransport(comfy.handler))
    # same runner the bridge would build, with fast polling so the tests take milliseconds
    bridge._images = ImageJobs(
        client=lambda slow: bridge._client(COMFY_URL, slow=slow),
        comfy_url=COMFY_URL,
        on_done=bridge._record_image_job,
        poll_seconds=0.01,
        timeout_seconds=5.0,
    )
    return SimpleNamespace(bridge=bridge, comfy=comfy, output_root=output_root, tmp=tmp_path)


def _wait(bridge: UnifiedBridge, job_id: str, *, until=("done", "failed", "cancelled"), seconds: float = 5.0) -> dict:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        job = bridge.image_status(job_id)["job"]
        if job["state"] in until:
            return job
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} never reached {until}")


def test_comfy_status_checks_workflow_prerequisites_and_blocks_unready_jobs(images_env):
    images_env.comfy.missing_models.add("Qwen Image 2.1/qwen_image_2.1_int8_convrot.safetensors")
    images_env.comfy.missing_nodes.add("TextEncodeQwenImage21")

    capability = images_env.bridge.status()["capabilities"]["comfyui"]

    assert capability["available"] is True
    assert capability["state"] == "partial"
    assert capability["workflow_prerequisites_ready"] is False
    assert capability["missing_nodes"] == ["TextEncodeQwenImage21"]
    assert capability["missing_models"] == ["diffusion_models/Qwen Image 2.1/qwen_image_2.1_int8_convrot.safetensors"]
    with pytest.raises(BridgeError, match="Missing"):
        images_env.bridge.start_image_job(
            prompt="a studio test image", aspect_ratio="1:1", quality="draft", seed=1,
            project_id=None, output_root=images_env.output_root,
        )
    assert images_env.comfy.workflows == []


@pytest.mark.parametrize("invalid_workflow", [
    [],
    {"1": {"class_type": "UNETLoader", "inputs": None}},
    {"1": {"class_type": "UNETLoader", "inputs": {}}},
])
def test_invalid_bundled_workflow_fails_closed_without_comfy_requests(images_env, invalid_workflow):
    workflow_path = images_env.tmp / "invalid-workflow.json"
    workflow_path.write_text(json.dumps(invalid_workflow), encoding="utf-8")
    images_env.bridge._images._workflow_path = workflow_path
    images_env.comfy.requests.clear()

    setup = images_env.bridge._images.readiness()
    assert setup["ready"] is False
    assert setup["errorCode"] == "comfyui_workflow_invalid"
    assert images_env.comfy.requests == []

    with pytest.raises(BridgeError) as caught:
        images_env.bridge.start_image_job(
            prompt="a studio test image", aspect_ratio="1:1", quality="draft", seed=1,
            project_id=None, output_root=images_env.output_root,
        )
    assert caught.value.code == "comfyui_workflow_invalid"
    assert images_env.comfy.workflows == []


def test_bundled_workflow_missing_required_loader_fails_closed(images_env):
    workflow = json.loads(images_env.bridge._images._workflow_path.read_text(encoding="utf-8"))
    workflow = {node_id: node for node_id, node in workflow.items() if node.get("class_type") != "VAELoader"}
    workflow_path = images_env.tmp / "workflow-without-vae.json"
    workflow_path.write_text(json.dumps(workflow), encoding="utf-8")
    images_env.bridge._images._workflow_path = workflow_path
    images_env.comfy.requests.clear()

    setup = images_env.bridge._images.readiness()

    assert setup["ready"] is False
    assert setup["errorCode"] == "comfyui_workflow_invalid"
    assert setup["missingNodes"] == ["473:VAELoader"]
    assert images_env.comfy.requests == []


def test_bundled_workflow_requires_canonical_fixed_node_ids(images_env):
    workflow = json.loads(images_env.bridge._images._workflow_path.read_text(encoding="utf-8"))
    workflow["other-size-node"] = workflow.pop("13")
    workflow_path = images_env.tmp / "workflow-with-renumbered-size-node.json"
    workflow_path.write_text(json.dumps(workflow), encoding="utf-8")
    images_env.bridge._images._workflow_path = workflow_path
    images_env.comfy.requests.clear()

    setup = images_env.bridge._images.readiness()

    assert setup["ready"] is False
    assert setup["errorCode"] == "comfyui_workflow_invalid"
    assert setup["missingNodes"] == ["13:ResolutionSelector"]
    assert images_env.comfy.requests == []


def test_comfy_disconnect_during_readiness_is_not_reported_as_partial(images_env):
    original_handler = images_env.comfy.handler

    def disconnect_after_system_stats(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/object_info":
            raise httpx.ConnectError("dropped", request=request)
        return original_handler(request)

    images_env.bridge.transport = httpx.MockTransport(disconnect_after_system_stats)

    capability = images_env.bridge.status()["capabilities"]["comfyui"]

    assert capability["available"] is False
    assert capability["state"] == "not_running"
    assert capability["workflow_prerequisites_ready"] is False


# --- the happy path ----------------------------------------------------------------------
def test_job_runs_saves_into_studio_outputs_and_records_the_project(images_env) -> None:
    bridge = images_env.bridge
    project = bridge.create_project("Image test")["project"]
    started = bridge.start_image_job(prompt="a red fox in fresh snow", aspect_ratio="16:9", quality="standard",
                                     seed=1234, project_id=project["project_id"], output_root=images_env.output_root)
    assert started["job"]["state"] == "queued" and started["job"]["seed"] == 1234

    job = _wait(bridge, started["job"]["job_id"])
    assert job["state"] == "done" and job["error"] is None
    image = job["images"][0]
    assert image["relative_path"].startswith("qwen-image/") and image["relative_path"].endswith(".png")
    assert (image["width"], image["height"]) == (64, 48)
    assert image["url"] == f"/api/pro/unified/images/{job['job_id']}/files/0"

    # the saved PNG carries A1111-style parameters, so Studio's /outputs listing reads it like any output
    saved = images_env.output_root / image["relative_path"]
    with Image.open(saved) as png:
        parameters = png.text["parameters"]
    assert parameters.startswith("a red fox in fresh snow\n") and "Seed: 1234" in parameters and "Size: 64x48" in parameters
    assert bridge.image_file(job["job_id"], 0) == saved.resolve()

    # the ledger keeps a hash of the prompt, never the prompt text
    events = bridge.get_project(project["project_id"])["project"]["events"]
    generated = [event for event in events if event["kind"] == "image_generated"]
    assert len(generated) == 1 and generated[0]["images"] == [image["relative_path"]]
    assert "a red fox" not in json.dumps(generated) and len(generated[0]["prompt_sha256"]) == 64
    assert "Images generated with Qwen Image 2.1: 1" in bridge.qwen_context(project["project_id"])["context"]


def test_workflow_is_filled_per_job_and_carries_no_lora(images_env) -> None:
    images_env.bridge.start_image_job(prompt="  harbor at noon  ", aspect_ratio="9:16", quality="draft", seed=7,
                                      project_id=None, output_root=images_env.output_root)
    workflow = images_env.comfy.workflows[0]
    assert workflow["471"]["inputs"]["prompt"] == "harbor at noon"
    assert workflow["476"]["inputs"]["seed"] == 7
    assert workflow["13"]["inputs"]["aspect_ratio"] == "9:16 (Portrait Widescreen)" and workflow["13"]["inputs"]["megapixels"] == 0.6
    assert workflow["476"]["inputs"]["model"] == ["470", 0]
    assert not any(node["class_type"].startswith("Lora") for node in workflow.values())
    assert not any("_meta" in node for node in workflow.values())


# --- refusals before the GPU is touched ----------------------------------------------------
@pytest.mark.parametrize("kwargs, code", [
    ({"prompt": "   "}, "invalid_prompt"),
    ({"prompt": "x" * 2001}, "invalid_prompt"),
    ({"prompt": "ok", "aspect_ratio": "5:4"}, "invalid_aspect_ratio"),
    ({"prompt": "ok", "quality": "ultra"}, "invalid_quality"),
    ({"prompt": "ok", "seed": -1}, "invalid_seed"),
    ({"prompt": "ok", "seed": True}, "invalid_seed"),
    ({"prompt": "ok", "project_id": "aiwfp-0000000000000000"}, "project_not_found"),
])
def test_invalid_requests_are_refused_without_queuing(images_env, kwargs, code) -> None:
    arguments = {"aspect_ratio": "1:1", "quality": "draft", "seed": None, "project_id": None, "output_root": images_env.output_root, **kwargs}
    with pytest.raises(BridgeError) as caught:
        images_env.bridge.start_image_job(**arguments)
    assert caught.value.code == code
    assert images_env.comfy.workflows == []


def test_missing_output_folder_and_engine_down_are_explained(images_env) -> None:
    with pytest.raises(BridgeError) as caught:
        images_env.bridge.start_image_job(prompt="ok", aspect_ratio="1:1", quality="draft", seed=1, project_id=None, output_root=None)
    assert caught.value.code == "no_output_root"
    images_env.comfy.down = True
    with pytest.raises(BridgeError) as caught:
        images_env.bridge.start_image_job(prompt="ok", aspect_ratio="1:1", quality="draft", seed=1, project_id=None, output_root=images_env.output_root)
    assert caught.value.status_code == 503 and caught.value.code == "comfyui_unreachable"
    assert images_env.bridge.status()["capabilities"]["comfyui"]["state"] == "not_running"


def test_comfyui_rejection_names_the_missing_model(images_env) -> None:
    images_env.comfy.reject_with = {
        "error": {"type": "prompt_outputs_failed_validation", "message": "Prompt outputs failed validation"},
        "node_errors": {"470": {"class_type": "UNETLoader", "errors": [{"message": "Value not in list",
                                "details": "unet_name: 'Qwen Image 2.1\\qwen_image_2.1_int8_convrot.safetensors' not in []"}]}},
    }
    with pytest.raises(BridgeError) as caught:
        images_env.bridge.start_image_job(prompt="ok", aspect_ratio="1:1", quality="draft", seed=1, project_id=None, output_root=images_env.output_root)
    assert caught.value.status_code == 422 and caught.value.code == "workflow_rejected"
    assert "UNETLoader" in caught.value.message and "qwen_image_2.1_int8_convrot" in caught.value.message


# --- failures and cancellation after queuing ------------------------------------------------
def test_execution_error_becomes_a_failed_job_with_the_reason(images_env) -> None:
    images_env.comfy.fail_execution = True
    job_id = images_env.bridge.start_image_job(prompt="ok", aspect_ratio="1:1", quality="draft", seed=1, project_id=None,
                                               output_root=images_env.output_root)["job"]["job_id"]
    job = _wait(images_env.bridge, job_id)
    assert job["state"] == "failed" and job["error"]["code"] == "generation_failed"
    assert "UNETLoader" in job["error"]["message"] and "CUDA out of memory" in job["error"]["message"]


def test_job_forgotten_by_comfyui_fails_instead_of_spinning(images_env) -> None:
    images_env.comfy.forget_jobs = True
    job_id = images_env.bridge.start_image_job(prompt="ok", aspect_ratio="1:1", quality="draft", seed=1, project_id=None,
                                               output_root=images_env.output_root)["job"]["job_id"]
    job = _wait(images_env.bridge, job_id)
    assert job["state"] == "failed" and job["error"]["code"] == "job_lost"


def test_cancel_dequeues_a_waiting_job_and_interrupts_only_our_running_job(images_env) -> None:
    comfy, bridge = images_env.comfy, images_env.bridge
    comfy.polls_until_done = 10_000           # nothing finishes during this test
    comfy.running = "someone-else"            # another client's job holds the GPU
    waiting = bridge.start_image_job(prompt="ok", aspect_ratio="1:1", quality="draft", seed=1, project_id=None,
                                     output_root=images_env.output_root)["job"]["job_id"]
    assert bridge.cancel_image(waiting)["job"]["state"] == "cancelled"
    assert comfy.deleted == ["p0001"] and comfy.interrupts == []

    comfy.running = None
    running = bridge.start_image_job(prompt="ok", aspect_ratio="1:1", quality="draft", seed=2, project_id=None,
                                     output_root=images_env.output_root)["job"]["job_id"]
    _wait(bridge, running, until=("running",))
    cancel_response = bridge.cancel_image(running)["job"]
    assert cancel_response["state"] == "running"
    assert cancel_response["cancel_requested"] is True
    assert _wait(bridge, running)["state"] == "cancelled"
    assert comfy.interrupts == [{"prompt_id": "p0002"}]


def test_unknown_and_malformed_job_ids(images_env) -> None:
    with pytest.raises(BridgeError) as caught:
        images_env.bridge.image_status("img-000000000000")
    assert caught.value.status_code == 404
    with pytest.raises(BridgeError) as caught:
        images_env.bridge.image_status("../etc")
    assert caught.value.status_code == 422


# --- HTTP routes ------------------------------------------------------------------------
def _client(env, host: str = "127.0.0.1") -> TestClient:
    ctx = SimpleNamespace(flags=SimpleNamespace(data_dir=env.tmp / "data", resolved_output_dir=lambda: env.output_root))
    app = FastAPI()
    app.state.ctx = ctx
    app.include_router(build_unified_router(ctx, bridge=env.bridge))
    return TestClient(app, client=(host, 50000))


def test_routes_generate_poll_and_serve_the_png(images_env) -> None:
    client = _client(images_env)
    response = client.post("/api/pro/unified/images", json={"prompt": "lighthouse at dusk", "aspect_ratio": "3:2"})
    assert response.status_code == 200
    job_id = response.json()["job"]["job_id"]
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = client.get(f"/api/pro/unified/images/{job_id}").json()["job"]
        if job["state"] == "done":
            break
        time.sleep(0.01)
    assert job["state"] == "done"
    png = client.get(job["images"][0]["url"])
    assert png.status_code == 200 and png.headers["content-type"] == "image/png" and png.content.startswith(b"\x89PNG")
    # unknown fields are refused, like every other unified route
    assert client.post("/api/pro/unified/images", json={"prompt": "x", "lora": "anything"}).status_code == 422


def test_unified_qwen_job_holds_shared_model_operation_lock_until_terminal(images_env) -> None:
    client = _client(images_env)
    images_env.comfy.polls_until_done = 100
    images_env.bridge._images._poll_seconds = 0.05
    from aiwf.services.model_startup import pro_model_load_lock

    lock = pro_model_load_lock(client.app.state.ctx)
    response = client.post("/api/pro/unified/images", json={"prompt": "hold the GPU lease"})
    assert response.status_code == 200
    job_id = response.json()["job"]["job_id"]
    assert lock.acquire(blocking=False) is False

    cancelled = client.post(f"/api/pro/unified/images/{job_id}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["job"]["cancel_requested"] is True
    assert _wait(images_env.bridge, job_id)["state"] == "cancelled"
    assert lock.acquire(blocking=False) is True
    lock.release()


def test_unexpected_image_watcher_failure_releases_gpu_lease_after_comfy_job_stops(images_env, monkeypatch) -> None:
    client = _client(images_env)
    from aiwf.services.model_startup import pro_model_load_lock

    def fail_history(*_args, **_kwargs):
        raise RuntimeError("synthetic watcher bug")

    monkeypatch.setattr(images_env.bridge._images, "_complete_from_history", fail_history)
    response = client.post("/api/pro/unified/images", json={"prompt": "watcher error recovery"})
    assert response.status_code == 200
    job_id = response.json()["job"]["job_id"]
    job = _wait(images_env.bridge, job_id, until=("failed",))
    assert job["error"]["code"] == "watcher_failed"

    lock = pro_model_load_lock(client.app.state.ctx)
    deadline = time.monotonic() + 1.0
    acquired = False
    while time.monotonic() < deadline:
        if lock.acquire(blocking=False):
            acquired = True
            break
        time.sleep(0.01)
    assert acquired is True
    lock.release()


def test_unified_qwen_job_rejects_when_another_model_operation_holds_shared_lock(images_env) -> None:
    client = _client(images_env)
    from aiwf.services.model_startup import pro_model_load_lock

    lock = pro_model_load_lock(client.app.state.ctx)
    assert lock.acquire(blocking=False) is True
    try:
        response = client.post("/api/pro/unified/images", json={"prompt": "do not overlap model load"})
    finally:
        lock.release()

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "gpu_busy"
    assert images_env.comfy.workflows == []


def test_failed_comfy_cancel_keeps_shared_gpu_lease_until_retry_confirms_removal(images_env) -> None:
    client = _client(images_env)
    images_env.comfy.running = "someone-else"
    images_env.comfy.polls_until_done = 10_000
    from aiwf.services.model_startup import pro_model_load_lock

    lock = pro_model_load_lock(client.app.state.ctx)
    response = client.post("/api/pro/unified/images", json={"prompt": "cancel status failure"})
    assert response.status_code == 200
    job_id = response.json()["job"]["job_id"]
    images_env.comfy.cancel_status = 500
    failed_cancel = client.post(f"/api/pro/unified/images/{job_id}/cancel")
    assert failed_cancel.status_code == 503
    assert lock.acquire(blocking=False) is False

    images_env.comfy.cancel_status = 200
    cancelled = client.post(f"/api/pro/unified/images/{job_id}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["job"]["state"] == "cancelled"
    assert lock.acquire(blocking=False) is True
    lock.release()


def test_timed_out_comfy_job_keeps_gpu_lease_until_stop_is_confirmed(images_env) -> None:
    client = _client(images_env)
    images_env.comfy.polls_until_done = 10_000
    images_env.bridge._images._poll_seconds = 0.01
    images_env.bridge._images._timeout_seconds = 0.03
    from aiwf.services.model_startup import pro_model_load_lock

    lock = pro_model_load_lock(client.app.state.ctx)
    response = client.post("/api/pro/unified/images", json={"prompt": "timeout lease check"})
    assert response.status_code == 200
    job_id = response.json()["job"]["job_id"]
    images_env.comfy.cancel_status = 500
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline and not images_env.comfy.interrupts:
        time.sleep(0.01)

    assert images_env.comfy.interrupts
    assert lock.acquire(blocking=False) is False

    images_env.comfy.cancel_status = 200
    job = _wait(images_env.bridge, job_id, until=("failed",))
    assert job["error"]["code"] == "timeout"
    deadline = time.monotonic() + 1
    acquired = False
    while time.monotonic() < deadline and not acquired:
        acquired = lock.acquire(blocking=False)
        if not acquired:
            time.sleep(0.01)
    assert acquired is True
    lock.release()


def test_lost_comfy_contact_does_not_release_gpu_lease_until_queue_is_checked(images_env) -> None:
    client = _client(images_env)
    images_env.bridge._images._poll_seconds = 0.01
    from aiwf.services.model_startup import pro_model_load_lock

    lock = pro_model_load_lock(client.app.state.ctx)
    response = client.post("/api/pro/unified/images", json={"prompt": "connection loss lease check"})
    assert response.status_code == 200
    job_id = response.json()["job"]["job_id"]
    images_env.comfy.down = True
    time.sleep(0.15)
    assert images_env.bridge.image_status(job_id)["job"]["state"] not in {"failed", "cancelled", "done"}
    assert lock.acquire(blocking=False) is False

    images_env.comfy.down = False
    images_env.comfy.forget_jobs = True
    job = _wait(images_env.bridge, job_id, until=("failed",))
    assert job["error"]["code"] == "job_lost"
    assert lock.acquire(blocking=False) is True
    lock.release()


def test_remote_clients_cannot_queue_or_cancel_images(images_env) -> None:
    client = _client(images_env, host="192.168.1.20")
    assert client.post("/api/pro/unified/images", json={"prompt": "x"}).status_code == 403
    assert client.post("/api/pro/unified/images/img-000000000000/cancel").status_code == 403
    assert images_env.comfy.workflows == []


def test_contract_offers_image_tools_but_not_the_png_route() -> None:
    names = {operation["name"] for operation in unified_contract.OPERATIONS}
    assert {"generate_image", "image_status", "cancel_image"} <= names
    assert ("GET", "/images/{job_id}/files/{index}") in unified_contract.MEDIA_ROUTES
