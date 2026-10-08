"""MCP server (stdio) for AIWF Studio's unified workspace.

Gives any MCP-capable agent (Claude Code, Codex, Grok, Qwen Chat tools) the same
operations as the HTTP API and the aiwf-studio CLI. Tools are generated from
aiwf/services/unified_contract.py and named studio_<operation>, plus
studio_start_pro to launch AIWF Studio Pro loopback-only when it is not running.

The server is a thin client: every tool call becomes one HTTP request to
AIWF Studio Pro (default http://127.0.0.1:7860, override with AIWF_STUDIO_URL).
Pro does the cross-app work and keeps the project ledger, so the CLI, the MCP
server and the Studio Flow screen all see the same state.

Runtime: the official MCP Python SDK (mcp 2.x) in
F:\\environments\\aiwf-studio-agent-py312 (recipe: requirements.txt here).
Run:  F:\\environments\\aiwf-studio-agent-py312\\Scripts\\python.exe aiwf_studio_mcp.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import anyio
import mcp_types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

sys.path.insert(0, str(Path(__file__).resolve().parent))
from studio_client import StudioClient, StudioError  # noqa: E402


SERVER_NAME = "aiwf-studio"
SERVER_VERSION = "1.0.0"
TOOL_PREFIX = "studio_"
START_TOOL = TOOL_PREFIX + "start_pro"


# --- tool definitions generated from the contract ---------------------------------------
def build_tools(client: StudioClient) -> list[types.Tool]:
    tools = []
    # this loop turns each contract operation into one MCP tool with the same schema
    for operation in client.operations():
        read_only = not operation["mutating"]
        tools.append(types.Tool(
            name=TOOL_PREFIX + operation["name"],
            title=operation["name"].replace("_", " ").capitalize(),
            description=f"{operation['summary']} [{operation['method']} {client.contract.BASE_PATH}{operation['path']}]",
            input_schema=client.contract.input_schema(operation),
            annotations=types.ToolAnnotations(
                read_only_hint=read_only,
                destructive_hint=False,
                # catalog/import are idempotent by design (re-runs reuse the same assets/revision)
                idempotent_hint=read_only or operation["name"] in {"catalog_outputs", "import_package"},
                # only Qwen Chat answers vary; everything else is a closed local system
                open_world_hint=operation["name"] == "qwen_ask",
            ),
        ))
    tools.append(types.Tool(
        name=START_TOOL,
        title="Start AIWF Studio Pro",
        description="Start AIWF Studio Pro with the loopback-only launcher if it is not already running, wait until it answers, and return the unified status. No app window opens.",
        input_schema={
            "type": "object",
            "properties": {"wait_seconds": {"type": "number", "minimum": 10, "maximum": 600, "default": 240, "description": "How long to wait for readiness."}},
            "additionalProperties": False,
        },
        annotations=types.ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=True, open_world_hint=False),
    ))
    return tools


def instructions(client: StudioClient) -> str:
    contract = client.contract
    lines = ["AIWF Studio unified workspace: AIWF Studio -> Dataset Studio -> ReTrain -> Qwen Chat, linked by one project ID.", "", "Workflow:"]
    lines += [f"{index}. {step}" for index, step in enumerate(contract.WORKFLOW, start=1)]
    lines += ["", "Rules:"] + [f"- {rule}" for rule in contract.RULES]
    lines += ["", f"If a tool returns pro_not_running, call {START_TOOL} first."]
    return "\n".join(lines)


# --- results ---------------------------------------------------------------------------------
def _ok(payload: dict[str, Any]) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(payload, indent=2, ensure_ascii=False))],
        structured_content=payload,
        is_error=False,
    )


def _fail(error: StudioError) -> types.CallToolResult:
    payload = error.as_dict()
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=f"{error.code}: {error.message}")],
        structured_content=payload,
        is_error=True,
    )


# --- server wiring -----------------------------------------------------------------------
def create_server(client: StudioClient | None = None) -> Server:
    client = client or StudioClient()
    tools = build_tools(client)
    names = {tool.name for tool in tools}

    async def on_list_tools(ctx: Any, params: types.PaginatedRequestParams | None) -> types.ListToolsResult:
        return types.ListToolsResult(tools=tools)

    async def on_call_tool(ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        if params.name not in names:
            return _fail(StudioError(0, "unknown_tool", f"Unknown tool: {params.name}"))
        arguments = dict(params.arguments or {})
        # HTTP calls block, so they run on a worker thread to keep the stdio loop responsive
        try:
            if params.name == START_TOOL:
                wait = float(arguments.get("wait_seconds", 240))
                result = await anyio.to_thread.run_sync(lambda: client.start_pro(wait_seconds=wait))
            else:
                operation = params.name[len(TOOL_PREFIX):]
                result = await anyio.to_thread.run_sync(lambda: client.call(operation, arguments))
        except StudioError as exc:
            return _fail(exc)
        return _ok(result)

    return Server(
        SERVER_NAME,
        version=SERVER_VERSION,
        title="AIWF Studio",
        description="Studio Flow bridge: Dataset Studio catalog and packages, ReTrain import and dry-run preflight, Qwen Chat project context.",
        instructions=instructions(client),
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )


async def _serve() -> None:
    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    anyio.run(_serve)


if __name__ == "__main__":
    main()
