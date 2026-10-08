"""Agent access to the unified workspace: contract, HTTP describe/outputs, CLI and MCP server.

The CLI and MCP server are exercised for real: a uvicorn server runs the actual
unified router (sibling apps faked at the HTTP transport, as in
test_unified_bridge.py), the CLI runs as a subprocess, and the MCP server is
driven by the official MCP SDK client from the agent environment.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aiwf.services import unified_contract
from aiwf.web.unified_api import build_unified_router
from test_unified_bridge import _studio_png, env  # noqa: F401  (env is a pytest fixture)


STUDIO_ROOT = Path(__file__).resolve().parents[2]
CLI = STUDIO_ROOT / "integrations" / "agent_bridge" / "aiwf_studio_cli.py"
MCP_SERVER = STUDIO_ROOT / "integrations" / "agent_bridge" / "aiwf_studio_mcp.py"
AGENT_PYTHON = Path(r"F:\environments\aiwf-studio-agent-py312\Scripts\python.exe")


def _app(env) -> FastAPI:
    ctx = SimpleNamespace(flags=SimpleNamespace(data_dir=env.tmp / "data", resolved_output_dir=lambda: env.output_root))
    app = FastAPI()
    app.include_router(build_unified_router(ctx, bridge=env.bridge))
    return app


# --- a real HTTP server for the subprocess clients ------------------------------------------
@pytest.fixture()
def live(env):
    server = uvicorn.Server(uvicorn.Config(_app(env), host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    assert server.started, "test server did not start"
    port = server.servers[0].sockets[0].getsockname()[1]
    yield SimpleNamespace(url=f"http://127.0.0.1:{port}", env=env)
    server.should_exit = True
    thread.join(timeout=10)


def _cli(url: str, *args: str, stdin: str | None = None) -> tuple[int, dict]:
    completed = subprocess.run(
        [sys.executable, str(CLI), *args],
        input=stdin, capture_output=True, text=True, timeout=120,
        env={**os.environ, "AIWF_STUDIO_URL": url},
    )
    return completed.returncode, json.loads(completed.stdout)


# --- contract and HTTP self-description ------------------------------------------------------
def test_contract_matches_the_router_exactly(env) -> None:
    ctx = SimpleNamespace(flags=SimpleNamespace(data_dir=env.tmp / "data", resolved_output_dir=lambda: env.output_root))
    router = build_unified_router(ctx, bridge=env.bridge)
    routes = {
        (method, route.path.removeprefix(unified_contract.BASE_PATH))
        for route in router.routes
        for method in route.methods
    }
    declared = {(operation["method"], operation["path"]) for operation in unified_contract.OPERATIONS}
    ui_only = set(unified_contract.UI_ONLY_ROUTES)
    media = set(unified_contract.MEDIA_ROUTES)
    # These routes intentionally stay outside the agent tool contract; chat-question
    # records the Studio UI's streamed Qwen exchange and is not agent-verifiable.
    assert ("POST", "/projects/{project_id}/chat-question") in ui_only
    assert ("POST", "/projects/{project_id}/chat-question") not in declared
    # Media routes return image bytes that agents get as relative paths instead.
    assert routes == declared | ui_only | media
    assert not declared & ui_only and not declared & media
    # every declared path parameter appears in its path, and every path placeholder is declared
    for operation in unified_contract.OPERATIONS:
        for param in operation["params"]:
            assert (param["in"] == "path") == ("{" + param["name"] + "}" in operation["path"]), operation["name"]


def test_describe_and_outputs_routes(env) -> None:
    client = TestClient(_app(env), client=("127.0.0.1", 50000))
    described = client.get("/api/pro/unified/describe").json()
    assert described["schema_version"] == "1" and len(described["operations"]) == len(unified_contract.OPERATIONS)
    assert any("Training is never started" in rule for rule in described["rules"])
    (env.output_root / "nested").mkdir()
    _studio_png(env.output_root / "nested", "shot.png", prompt="harbor at noon")
    outputs = client.get("/api/pro/unified/outputs", params={"limit": 5}).json()["outputs"]
    assert outputs[0]["relative_path"] == "nested/shot.png"
    assert outputs[0]["prompt"] == "harbor at noon" and outputs[0]["seed"] == 42
    assert str(env.output_root) not in json.dumps(outputs)
    assert client.get("/api/pro/unified/outputs", params={"limit": 0}).status_code == 422


# --- CLI as a subprocess against a real server --------------------------------------------------
def test_cli_runs_the_full_flow_and_reports_errors_with_exit_codes(live) -> None:
    code, listed = _cli(live.url, "list")
    assert code == 0 and {op["name"] for op in listed["operations"]} >= {"status", "import_package", "qwen_ask"}

    code, status = _cli(live.url, "status")
    assert code == 0 and status["capabilities"]["retrain"]["available"] is True

    code, created = _cli(live.url, "create-project", "--name", "CLI project")
    project_id = created["project"]["project_id"]
    _studio_png(live.env.output_root, "one.png")
    code, outputs = _cli(live.url, "list-outputs", "--limit", "5")
    assert code == 0 and outputs["outputs"][0]["relative_path"] == "one.png"
    code, cataloged = _cli(live.url, "catalog-outputs", "--project-id", project_id, "--output-paths", "one.png")
    assert code == 0 and cataloged["status"] == "cataloged"

    digest = live.env.fake.package_hash
    request = {"project_id": project_id, "package_name": "fixture text pack", "manifest_sha256": digest}
    code, imported = _cli(live.url, "call", "import-package", "--args-json", "-", stdin=json.dumps(request))
    assert code == 0 and imported["dataset"]["dataset_id"] == f"sha256-{digest}"

    # stdin request form, used by aiwf-studio-json.cmd; free text never touches the command line
    question = 'Is "this" enough? & | < > %PATH%'
    code, answer = _cli(live.url, "request", stdin=json.dumps({"operation": "qwen_ask", "arguments": {"project_id": project_id, "model_id": "qwen3-8b", "question": question}}))
    assert code == 0 and answer["answer"] == "Looks consistent."
    assert live.env.fake.qwen_payloads[-1]["messages"][1]["content"] == question

    code, stale = _cli(live.url, "request", stdin=json.dumps({"operation": "import_package", "arguments": {**request, "manifest_sha256": "a" * 64}}))
    assert code == 4 and stale["code"] == "stale_revision" and stale["status"] == 409
    code, missing = _cli(live.url, "call", "import-package", "--args-json", "{}")
    assert code == 2 and missing["code"] == "invalid_arguments"
    code, unknown = _cli(live.url, "request", stdin=json.dumps({"operation": "start_training", "arguments": {}}))
    assert code == 2 and unknown["code"] == "unknown_operation"


def test_cli_distinguishes_pro_down_and_refuses_remote_urls() -> None:
    code, down = _cli("http://127.0.0.1:9", "status")
    assert code == 3 and down["code"] == "pro_not_running"
    code, remote = _cli("http://192.168.1.20:7860", "status")
    assert code == 2 and remote["code"] == "non_loopback_url"


# --- MCP server through the official SDK client ------------------------------------------------
MCP_PROBE = """
import asyncio, json, sys
from mcp import Client, StdioServerParameters

async def main():
    params = StdioServerParameters(command=sys.argv[1], args=[sys.argv[2]], env={"AIWF_STUDIO_URL": sys.argv[3]})
    out = {}
    async with Client(params) as client:
        listed = await client.list_tools()
        out["tools"] = {tool.name: {"read_only": tool.annotations.read_only_hint, "required": tool.input_schema.get("required", [])} for tool in listed.tools}
        status = await client.call_tool("studio_status", {})
        out["status"] = status.structured_content
        created = await client.call_tool("studio_create_project", {"name": "MCP project"})
        project_id = created.structured_content["project"]["project_id"]
        packages = await client.call_tool("studio_list_packages", {})
        package = packages.structured_content["packages"][0]
        imported = await client.call_tool("studio_import_package", {"project_id": project_id, "package_name": package["package_name"], "manifest_sha256": package["manifest_sha256"]})
        out["imported"] = imported.structured_content
        stale = await client.call_tool("studio_import_package", {"project_id": project_id, "package_name": package["package_name"], "manifest_sha256": "a" * 64})
        out["stale"] = {"is_error": stale.is_error, "content": stale.structured_content}
        context = await client.call_tool("studio_qwen_context", {"project_id": project_id})
        out["context"] = context.structured_content["context"]
    print(json.dumps(out))

asyncio.run(main())
"""


@pytest.mark.skipif(not AGENT_PYTHON.is_file(), reason="agent environment with the MCP SDK is not installed")
def test_mcp_server_works_with_the_official_sdk_client(live, tmp_path: Path) -> None:
    probe = tmp_path / "mcp_probe.py"
    probe.write_text(MCP_PROBE, encoding="utf-8")
    completed = subprocess.run(
        [str(AGENT_PYTHON), str(probe), str(AGENT_PYTHON), str(MCP_SERVER), live.url],
        capture_output=True, text=True, timeout=180,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    out = json.loads(completed.stdout.strip().splitlines()[-1])

    expected = {"studio_" + op["name"] for op in unified_contract.OPERATIONS} | {"studio_start_pro"}
    assert set(out["tools"]) == expected
    assert out["tools"]["studio_status"]["read_only"] is True
    assert out["tools"]["studio_import_package"]["read_only"] is False
    assert out["tools"]["studio_preflight"]["required"] == ["project_id", "dataset_id", "manifest_sha256", "model_id"]
    assert out["status"]["capabilities"]["qwen_chat"]["available"] is True
    assert out["imported"]["dataset"]["dataset_id"].startswith("sha256-")
    assert out["stale"]["is_error"] is True and out["stale"]["content"]["code"] == "stale_revision"
    assert "ReTrain imports:" in out["context"]
