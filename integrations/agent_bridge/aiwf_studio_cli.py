"""aiwf-studio: command-line access to AIWF Studio's unified workspace for agents and people.

Every subcommand is generated from aiwf/services/unified_contract.py, so it always
matches the HTTP API and the MCP server. Output is one JSON document on stdout.

Usage:
    aiwf-studio list                                  operations and their parameters
    aiwf-studio status
    aiwf-studio create-project --name "Lighthouse set"
    aiwf-studio list-outputs --limit 10
    aiwf-studio catalog-outputs --project-id aiwfp-... --output-paths a.png b.png
    aiwf-studio import-package --project-id aiwfp-... --package-name NAME --manifest-sha256 HEX
    aiwf-studio preflight --project-id ... --dataset-id sha256-HEX --manifest-sha256 HEX --model-id qwen2.5-coder-1.5b --settings "{\"method\": \"QLoRA\"}"
    aiwf-studio call qwen-ask --args-json -           JSON arguments on stdin (safest for free text)
    aiwf-studio start-pro                             launch Pro loopback-only and wait until ready

Exit codes: 0 ok, 2 bad arguments, 3 AIWF Studio Pro not running or not ready,
4 the operation was refused or failed (see "code" in the JSON output).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from studio_client import StudioClient, StudioError  # noqa: E402


# --- exit codes ------------------------------------------------------------------------
EXIT_OK, EXIT_USAGE, EXIT_NOT_RUNNING, EXIT_FAILED = 0, 2, 3, 4
_NOT_RUNNING_CODES = {"pro_not_running", "pro_starting", "timeout", "start_timeout", "cannot_start"}


def _emit(payload: Any) -> None:
    # ensure_ascii keeps the output safe on any Windows console code page
    sys.stdout.write(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")
    sys.stdout.flush()


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


# --- parser construction from the contract ------------------------------------------
def build_parser(client: StudioClient) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aiwf-studio", description="Agent CLI for AIWF Studio's unified workspace (Dataset Studio, ReTrain, Qwen Chat).")
    parser.add_argument("--url", help="AIWF Studio Pro base URL (default from AIWF_STUDIO_URL or http://127.0.0.1:7860; loopback only).")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List operations, parameters, workflow and rules.")
    start = commands.add_parser("start-pro", help="Start AIWF Studio Pro (loopback only) if it is not running, then print status.")
    start.add_argument("--wait", type=float, default=240.0, help="Seconds to wait for readiness (default 240).")
    call = commands.add_parser("call", help="Run any operation with JSON arguments.")
    call.add_argument("operation", help="Operation name, e.g. qwen-ask or qwen_ask.")
    call.add_argument("--args-json", default="{}", help="JSON object of arguments, or - to read it from stdin.")
    # used by aiwf-studio-json.cmd: the whole request arrives on stdin, nothing on the command line
    commands.add_parser("request", help='Read {"operation": "...", "arguments": {...}} from stdin and run it.')

    # this loop adds one subcommand per contract operation, with one flag per parameter
    for operation in client.operations():
        sub = commands.add_parser(operation["name"].replace("_", "-"), help=operation["summary"], description=operation["summary"])
        for param in operation["params"]:
            schema = param["schema"]
            kwargs: dict[str, Any] = {"dest": param["name"], "help": param["description"], "required": param["required"]}
            if schema["type"] == "boolean":
                # a switch: absent means "not given" (None) so the server default applies
                kwargs = {"dest": param["name"], "help": param["description"], "action": "store_true", "default": None}
            elif schema["type"] == "array":
                kwargs["nargs"] = "+"
            elif schema["type"] == "integer":
                kwargs["type"] = int
            elif schema["type"] == "object":
                kwargs["type"] = _json_object
                kwargs["metavar"] = "JSON"
            sub.add_argument(_flag(param["name"]), **kwargs)
    return parser


def _json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("must be a JSON object")
    return value


def _read_args_json(raw: str) -> dict[str, Any]:
    text = sys.stdin.read() if raw == "-" else raw
    value = json.loads(text or "{}")
    if not isinstance(value, dict):
        raise ValueError("--args-json must be a JSON object")
    return value


# --- entry point ----------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    try:
        client = StudioClient()
    except StudioError as exc:
        _emit(exc.as_dict())
        return EXIT_USAGE
    parser = build_parser(client)
    args = parser.parse_args(argv)
    if args.url:
        try:
            client = StudioClient(base_url=args.url)
        except StudioError as exc:
            _emit(exc.as_dict())
            return EXIT_USAGE

    try:
        if args.command == "list":
            _emit(client.contract.describe())
            return EXIT_OK
        if args.command == "start-pro":
            _emit(client.start_pro(wait_seconds=args.wait))
            return EXIT_OK
        if args.command == "call":
            try:
                arguments = _read_args_json(args.args_json)
            except (ValueError, json.JSONDecodeError) as exc:
                _emit({"ok": False, "status": 0, "code": "invalid_arguments", "message": str(exc)})
                return EXIT_USAGE
            _emit(client.call(args.operation, arguments))
            return EXIT_OK
        if args.command == "request":
            try:
                request = json.loads(sys.stdin.read() or "{}")
                if not isinstance(request, dict) or not isinstance(request.get("operation"), str):
                    raise ValueError('stdin must be {"operation": "...", "arguments": {...}}')
                arguments = request.get("arguments") or {}
                if not isinstance(arguments, dict):
                    raise ValueError("arguments must be a JSON object")
            except (ValueError, json.JSONDecodeError) as exc:
                _emit({"ok": False, "status": 0, "code": "invalid_arguments", "message": str(exc)})
                return EXIT_USAGE
            if request["operation"].replace("-", "_") == "start_pro":
                _emit(client.start_pro(wait_seconds=float(arguments.get("wait_seconds", 240))))
            else:
                _emit(client.call(request["operation"], arguments))
            return EXIT_OK
        operation = client.contract.find_operation(args.command)
        arguments = {param["name"]: getattr(args, param["name"]) for param in operation["params"] if getattr(args, param["name"]) is not None}
        _emit(client.call(operation["name"], arguments))
        return EXIT_OK
    except KeyError as exc:
        _emit({"ok": False, "status": 0, "code": "unknown_operation", "message": str(exc).strip("'\"")})
        return EXIT_USAGE
    except StudioError as exc:
        _emit(exc.as_dict())
        if exc.code == "invalid_arguments":
            return EXIT_USAGE
        return EXIT_NOT_RUNNING if exc.code in _NOT_RUNNING_CODES else EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
