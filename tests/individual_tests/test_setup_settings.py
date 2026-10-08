"""Guided setup folders: aiwf/services/setup_settings.py and the /setup routes.

The native app's setup wizard reads and saves where Studio saves images and finds
models. These tests pin the promises the wizard relies on: only the five folder keys
change, every other launch.json value survives byte-for-byte in meaning, bad folders are
refused with a code and the field name, and the engine API stays torch-free.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aiwf.services import setup_settings
from aiwf.web.unified_api import build_unified_router

REPO = Path(__file__).resolve().parents[2]


# --- helpers -------------------------------------------------------------------------------
def _client(data_dir: Path, host: str = "127.0.0.1") -> TestClient:
    """The unified router alone, as the engine API mounts it, seen from one client address."""
    ctx = SimpleNamespace(flags=SimpleNamespace(data_dir=data_dir, resolved_output_dir=lambda: data_dir / "outputs"))
    app = FastAPI()
    app.include_router(build_unified_router(ctx))
    return TestClient(app, client=(host, 50000))


def _launch(data_dir: Path) -> dict:
    return json.loads((data_dir / "launch.json").read_text(encoding="utf-8"))


# --- reading ---------------------------------------------------------------------------------
def test_describe_without_a_saved_profile_reports_the_defaults(tmp_path: Path) -> None:
    described = setup_settings.describe_setup(tmp_path)
    folders = described["folders"]
    assert described["launch_file_exists"] is False
    assert folders["output_dir"]["saved"] == "" and folders["output_dir"]["path"] == str(tmp_path / "outputs")
    assert folders["models_dir"]["default"] == str(tmp_path / "models")
    assert folders["ckpt_dir"]["default"] == str(tmp_path / "models" / "Stable-diffusion")
    assert folders["extra_model_dirs"]["saved"] == [] and folders["extra_ckpt_dirs"]["effective"] == []
    # a missing folder still reports its drive's free space and whether it could be created
    assert folders["output_dir"]["exists"] is False
    assert folders["output_dir"]["free_bytes"] and folders["output_dir"]["writable"] is True
    assert described["python"]["exists"] is False


# --- writing ---------------------------------------------------------------------------------
def test_saving_changes_only_folder_keys_and_keeps_everything_else(tmp_path: Path) -> None:
    (tmp_path / "launch.json").write_text(json.dumps({
        "port": 7861, "theme": "light", "vram_profile": "low", "future_option": {"kept": True}, "output_dir": "",
    }), encoding="utf-8")
    images = tmp_path / "Pictures" / "AIWF"
    images.mkdir(parents=True)
    result = setup_settings.save_folders(tmp_path, {"output_dir": str(images)})
    saved = _launch(tmp_path)
    assert saved["output_dir"] == str(images.resolve())
    assert saved["port"] == 7861 and saved["theme"] == "light" and saved["vram_profile"] == "low"
    assert saved["future_option"] == {"kept": True}          # unknown keys from newer versions survive
    assert result["saved_fields"] == ["output_dir"] and result["folders"]["output_dir"]["exists"] is True
    # no temporary files are left next to launch.json
    assert sorted(p.name for p in tmp_path.iterdir() if p.is_file()) == ["launch.json"]


def test_first_save_writes_a_complete_valid_profile(tmp_path: Path) -> None:
    models = tmp_path / "lib"
    models.mkdir()
    setup_settings.save_folders(tmp_path, {"models_dir": str(models)})
    saved = _launch(tmp_path)
    assert saved["models_dir"] == str(models.resolve()) and saved["port"] == 7860 and saved["listen"] is False


def test_empty_value_returns_a_folder_to_its_default(tmp_path: Path) -> None:
    custom = tmp_path / "custom"
    custom.mkdir()
    setup_settings.save_folders(tmp_path, {"ckpt_dir": str(custom)})
    result = setup_settings.save_folders(tmp_path, {"ckpt_dir": ""})
    assert _launch(tmp_path)["ckpt_dir"] == ""
    assert result["folders"]["ckpt_dir"]["path"] == str(tmp_path / "models" / "Stable-diffusion")


def test_folder_lists_are_cleaned_and_deduplicated(tmp_path: Path) -> None:
    first, second = tmp_path / "A", tmp_path / "B"
    first.mkdir()
    second.mkdir()
    result = setup_settings.save_folders(tmp_path, {"extra_model_dirs": [str(first), "  ", str(first).upper(), str(second)]})
    assert _launch(tmp_path)["extra_model_dirs"] == f"{first.resolve()}\n{second.resolve()}"
    assert [item["path"] for item in result["folders"]["extra_model_dirs"]["effective"]] == [str(first.resolve()), str(second.resolve())]


@pytest.mark.parametrize(
    ("changes", "code", "field"),
    [
        ({"output_dir": "relative\\images"}, "relative_folder", "output_dir"),
        ({"models_dir": "Z:\\definitely\\not\\here\\aiwf"}, "folder_missing", "models_dir"),
        ({"port": 9000}, "unknown_field", None),
        ({"extra_ckpt_dirs": "C:\\not-a-list"}, "invalid_folder", "extra_ckpt_dirs"),
    ],
)
def test_bad_folders_are_refused_without_writing(tmp_path: Path, changes: dict, code: str, field: str | None) -> None:
    with pytest.raises(setup_settings.SetupError) as raised:
        setup_settings.save_folders(tmp_path, changes)
    assert raised.value.code == code and raised.value.field == field
    assert not (tmp_path / "launch.json").exists()


def test_a_file_is_not_accepted_as_a_folder(tmp_path: Path) -> None:
    file = tmp_path / "notes.txt"
    file.write_text("x", encoding="utf-8")
    with pytest.raises(setup_settings.SetupError) as raised:
        setup_settings.save_folders(tmp_path, {"output_dir": str(file)})
    assert raised.value.code == "not_a_folder"


def test_missing_folders_are_created_only_when_asked(tmp_path: Path) -> None:
    wanted = tmp_path / "new" / "outputs"
    result = setup_settings.save_folders(tmp_path, {"output_dir": str(wanted)}, create_missing=True)
    assert wanted.is_dir() and result["folders"]["output_dir"]["exists"] is True


# --- the HTTP routes ----------------------------------------------------------------------------
def test_routes_read_for_everyone_and_save_only_locally(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    local = _client(tmp_path)
    assert local.get("/api/pro/unified/setup").json()["folders"]["output_dir"]["saved"] == ""
    saved = local.post("/api/pro/unified/setup/folders", json={"output_dir": str(images)})
    assert saved.status_code == 200 and saved.json()["saved_fields"] == ["output_dir"]

    remote = _client(tmp_path, host="192.168.1.20")
    assert remote.get("/api/pro/unified/setup").status_code == 200
    refused = remote.post("/api/pro/unified/setup/folders", json={"output_dir": str(images)})
    assert refused.status_code == 403 and refused.json()["detail"]["code"] == "local_only"


def test_route_errors_name_the_field_and_reject_unknown_keys(tmp_path: Path) -> None:
    client = _client(tmp_path)
    bad = client.post("/api/pro/unified/setup/folders", json={"models_dir": "models"})
    assert bad.status_code == 422 and bad.json()["detail"] == {
        "code": "relative_folder", "message": bad.json()["detail"]["message"], "field": "models_dir"}
    assert client.post("/api/pro/unified/setup/folders", json={}).json()["detail"]["code"] == "nothing_to_save"
    # pydantic refuses unknown body fields before any file is touched
    assert client.post("/api/pro/unified/setup/folders", json={"port": 1}).status_code == 422
    assert not (tmp_path / "launch.json").exists()


def test_setup_routes_do_not_load_torch(tmp_path: Path) -> None:
    script = textwrap.dedent(f"""
        import sys
        from pathlib import Path
        from types import SimpleNamespace
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from aiwf.web.unified_api import build_unified_router

        root = Path({str(tmp_path)!r})
        (root / "out").mkdir()
        ctx = SimpleNamespace(flags=SimpleNamespace(data_dir=root, resolved_output_dir=lambda: root / "outputs"))
        app = FastAPI()
        app.include_router(build_unified_router(ctx))
        client = TestClient(app, client=("127.0.0.1", 5000))
        assert client.get("/api/pro/unified/setup").status_code == 200
        assert client.post("/api/pro/unified/setup/folders", json={{"output_dir": str(root / "out")}}).status_code == 200
        print("TORCH_LOADED=" + str("torch" in sys.modules))
        print("PRO_API_LOADED=" + str("aiwf.web.pro_api" in sys.modules))
    """)
    result = subprocess.run([sys.executable, "-c", script], cwd=REPO, capture_output=True, text=True, timeout=240)
    assert result.returncode == 0, result.stderr[-2000:]
    assert "TORCH_LOADED=False" in result.stdout and "PRO_API_LOADED=False" in result.stdout, result.stdout
