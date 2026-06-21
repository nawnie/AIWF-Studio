from __future__ import annotations

import json
from pathlib import Path

from aiwf.services.chat_agent_tools import (
    ChatAgentToolService,
    parse_agent_tool_request,
    run_agentic_turn,
)


def _service(root: Path, *, allow_file_edits: bool = False, skill_root: Path | None = None) -> ChatAgentToolService:
    output = root / "outputs"
    output.mkdir(exist_ok=True)
    return ChatAgentToolService(
        allowed_roots=[root],
        output_dir=output,
        skill_roots=[skill_root or root],
        allow_file_edits=allow_file_edits,
    )


def test_parse_agent_tool_request_accepts_fenced_json() -> None:
    request = parse_agent_tool_request(
        'Use a tool:\n```json\n{"tool":"read_file","args":{"path":"README.md"}}\n```'
    )

    assert request is not None
    assert request.tool == "read_file"
    assert request.args == {"path": "README.md"}


def test_read_file_and_search_are_root_bound(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("Atlas agent tools\n", encoding="utf-8")
    tools = _service(tmp_path)

    read = tools.execute_tool("read_file", {"path": "README.md"})
    search = tools.execute_tool("search_text", {"query": "agent", "path": "."})
    escape = tools.execute_tool("read_file", {"path": "..\\secret.txt"})

    assert read.ok is True
    assert "Atlas agent tools" in read.content
    assert search.ok is True
    assert "README.md:1" in search.content
    assert escape.ok is False
    assert "outside allowed roots" in escape.summary


def test_edit_tools_require_explicit_permission(tmp_path: Path) -> None:
    target = tmp_path / "app.py"
    target.write_text("print('old')\n", encoding="utf-8")

    blocked = _service(tmp_path).execute_tool(
        "replace_text",
        {"path": "app.py", "old": "old", "new": "new"},
    )
    allowed = _service(tmp_path, allow_file_edits=True).execute_tool(
        "replace_text",
        {"path": "app.py", "old": "old", "new": "new"},
    )

    assert blocked.ok is False
    assert "disabled" in blocked.summary
    assert allowed.ok is True
    assert target.read_text(encoding="utf-8") == "print('new')\n"


def test_write_file_creates_text_file_when_edits_are_enabled(tmp_path: Path) -> None:
    tools = _service(tmp_path, allow_file_edits=True)

    result = tools.execute_tool(
        "write_file",
        {"path": "notes/plan.md", "content": "# Plan\n", "overwrite": False},
    )

    assert result.ok is True
    assert (tmp_path / "notes" / "plan.md").read_text(encoding="utf-8") == "# Plan\n"


def test_discovers_codex_or_claude_style_instruction_packs(tmp_path: Path) -> None:
    skill = tmp_path / "skills" / "python-helper"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: python-helper\ndescription: Helps with Python edits.\n---\n# Python Helper\n",
        encoding="utf-8",
    )
    claude = tmp_path / "skills" / "claude-style"
    claude.mkdir()
    (claude / "CLAUDE.md").write_text("# Claude Style\nLocal instructions.\n", encoding="utf-8")
    tools = _service(tmp_path, skill_root=tmp_path / "skills")

    listed = tools.execute_tool("list_skills", {})
    read = tools.execute_tool("read_skill", {"name": "python-helper"})

    assert listed.ok is True
    assert "python-helper" in listed.content
    assert "claude-style" in listed.content
    assert read.ok is True
    assert "Helps with Python edits" in read.content


def test_agentic_turn_executes_tool_then_returns_final_answer(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("Agent tools are local.\n", encoding="utf-8")
    tools = _service(tmp_path)
    responses = iter(
        [
            json.dumps({"tool": "read_file", "args": {"path": "README.md"}}),
            "The README says agent tools are local.",
        ]
    )

    def chat_once(messages: list[dict[str, str]]) -> str:
        return next(responses)

    updates = list(
        run_agentic_turn(
            chat_once,
            [{"role": "user", "content": "What does the README say?"}],
            tools,
            max_steps=2,
        )
    )

    assert updates[-1].done is True
    assert updates[-1].content == "The README says agent tools are local."
    assert "read_file" in updates[-1].trace


def test_agent_tool_calls_are_audited(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("audit me\n", encoding="utf-8")
    tools = _service(tmp_path)

    tools.execute_tool("read_file", {"path": "README.md"})

    log = tmp_path / "outputs" / ".chat_agent_tool_log.jsonl"
    entry = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
    assert entry["tool"] == "read_file"
    assert entry["gpu_tenant_at_call"] == "chat"
