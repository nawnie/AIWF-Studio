# Studio Flow: agent access (HTTP, CLI, MCP)

Three ways to drive the unified workspace (AIWF Studio -> Dataset Studio -> ReTrain -> Qwen Chat). All three are generated from one list, `aiwf/services/unified_contract.py`, and a test fails if they drift.

Everything is local: AIWF Studio Pro must be running on loopback (default `http://127.0.0.1:7860`), and write operations are accepted only from the local machine. Nothing here can start training.

The same API also runs on its own, without Pro's web UI or diffusion runtime: `venv\Scripts\python.exe -m aiwf.engine_api` serves exactly these routes on `http://127.0.0.1:7870` (loopback only, starts in about a second). AIWF Studio for Windows (`native/`) starts it automatically; point the CLI or MCP server at it with `AIWF_STUDIO_URL=http://127.0.0.1:7870`.

Images: `generate_image` queues a Qwen Image 2.1 job on the local ComfyUI (`AIWF_COMFYUI_URL`, default `http://127.0.0.1:8188`), `image_status` follows it, `cancel_image` stops only that job. Finished PNGs land in Studio's output folder under `qwen-image/<date>/` with their settings embedded, so `list_outputs` and `catalog_outputs` treat them like any Studio output. The PNG bytes route (`GET /images/{job_id}/files/{index}`) is for UIs and is not an agent tool.

## 1. HTTP (same origin as the Studio Flow screen)

`GET /api/pro/unified/describe` returns every operation, its parameters, the recommended workflow and the rules. Errors are `{"detail": {"code", "message"}}`.

## 2. CLI

```bash
C:\AI-Agent-Workspace\bin\aiwf-studio.cmd list
C:\AI-Agent-Workspace\bin\aiwf-studio.cmd start-pro            # loopback-only launch, waits until ready
C:\AI-Agent-Workspace\bin\aiwf-studio.cmd status
C:\AI-Agent-Workspace\bin\aiwf-studio.cmd create-project --name "Lighthouse set"
```

For model-written text, use the stdin form so nothing passes through command-line quoting:

```bash
echo {"operation":"qwen_ask","arguments":{"project_id":"aiwfp-...","model_id":"...","question":"..."}} | C:\AI-Agent-Workspace\bin\aiwf-studio-json.cmd
```

Output is one JSON document. Exit codes: 0 ok, 2 bad arguments, 3 Pro not running or not ready, 4 refused or failed (read `code`).
The CLI uses only the Python standard library.

## 3. MCP server (stdio)

Tools are named `studio_<operation>`, plus `studio_start_pro`. Read-only operations are annotated read-only.

Claude Code (already registered on this machine, user scope):

```bash
claude mcp add --scope user aiwf-studio -- C:\Users\Shawn\AppData\Local\Python\pythoncore-3.12-64\pythonw.exe C:\Users\Shawn\.codex\scripts\run_noconsole.py -- F:\environments\aiwf-studio-agent-py312\Scripts\python.exe -B F:\AIWF_Studio\integrations\agent_bridge\aiwf_studio_mcp.py
```

Codex (`config.toml`):

```toml
[mcp_servers.aiwf-studio]
command = "C:\\Users\\Shawn\\AppData\\Local\\Python\\pythoncore-3.12-64\\pythonw.exe"
args = ["C:\\Users\\Shawn\\.codex\\scripts\\run_noconsole.py", "--", "F:\\environments\\aiwf-studio-agent-py312\\Scripts\\python.exe", "-B", "F:\\AIWF_Studio\\integrations\\agent_bridge\\aiwf_studio_mcp.py"]
```

Grok CLI: `grok mcp add aiwf-studio -- <the same command and arguments>`.

The server needs the MCP SDK (mcp 2.x) from `F:\environments\aiwf-studio-agent-py312`. Recipe: `integrations/agent_bridge/requirements.txt` (exact versions in `requirements-lock.txt`). Under mcp 2.x, `FastMCP` is now `MCPServer`; this server uses the low-level `Server` API.

## Workflow

1. `status`: are Dataset Studio, ReTrain and Qwen Chat reachable, and which models do they offer?
2. `list_projects` / `create_project`: one project ID links everything below.
3. `list_outputs`, then `catalog_outputs`: put Studio images into Dataset Studio's catalog (needs Dataset Studio started with `-IncludeAiwfOutputs`).
4. `list_packages`: choose a ready package; keep `package_name` and `manifest_sha256` together.
5. `import_package` with that exact pair; the result's `dataset_id` is `sha256-<manifest_sha256>`.
6. `preflight` with `dataset_id`, `manifest_sha256` and a text-capable `model_id`: a dry-run plan only.
7. `qwen_context`, then `qwen_ask`: preview what is sent, then ask. A model that is not loaded is loaded first (uses the GPU).

## Rules

- A changed package returns `stale_revision` (HTTP 409): refresh `list_packages` and choose again. Never guess a revision.
- Image outputs are cataloged only; Dataset Studio's ReTrain packages are text-only.
- `pro_not_running` means call `start-pro` / `studio_start_pro` first.
- Configuration: `AIWF_STUDIO_URL` (loopback only) and `AIWF_STUDIO_TIMEOUT` (default 300 s).
