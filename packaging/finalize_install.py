"""Finalize a fresh AES Studio installation without downloading dependencies."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from setup_settings import configure_backend, ensure_profile_launch, migrate_legacy_profile

def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

root = Path(sys.argv[1]).resolve(strict=True)
backend = sys.argv[2]
settings_policy = sys.argv[3]
if backend not in {"cpp", "both"} or settings_policy not in {"apply", "preserve"}:
    raise SystemExit("Invalid backend selection or settings policy")
if not (root / "aiwf" / "engine_api.py").is_file():
    raise SystemExit("Installed API source is missing")
profile = Path(os.environ.get("AES_STUDIO_DATA_DIR") or (Path(os.environ["LOCALAPPDATA"]) / "Programs" / "AES Studio Data"))
if not profile.is_absolute() or profile == Path(profile.anchor):
    raise SystemExit("AES Studio profile must be an absolute, non-root directory")
profile.mkdir(parents=True, exist_ok=True)
venv = root / "venv"
clean = os.environ.copy()
for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PIP_REQUIRE_VIRTUALENV", "PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL"):
    clean.pop(key, None)
clean["PYTHONNOUSERSITE"] = "1"
clean["PIP_NO_INPUT"] = "1"
subprocess.run([str(root / "runtime" / "python" / "python.exe"), "-m", "venv", str(venv)], check=True, env=clean)
python = venv / "Scripts" / "python.exe"
requirements = root / "native" / "AIWF.Desktop" / "engine-api-requirements.txt"
subprocess.run([str(python), "-m", "pip", "--isolated", "install", "--no-index", "--find-links", str(root / "wheels"), "-r", str(requirements)], check=True, env=clean)
preflight = subprocess.run([str(python), str(root / "native" / "AIWF.Desktop" / "engine-api-preflight.py"), str(requirements)], check=True, text=True, capture_output=True, env=clean)
if preflight.stdout.strip() != "READY":
    raise SystemExit("Offline engine runtime verification failed")
info = json.loads((root / "native" / "native-build-info.json").read_text(encoding="utf-8"))
binding = {"schema":"aiwf-studio-root-v1", "studio_root":str(root), "protocol_version":info["protocol_version"], "api_source_fingerprint":info["api_source_fingerprint"]}
(root / "native" / "studio-root.json").write_text(json.dumps(binding, indent=2) + "\n", encoding="utf-8")
subprocess.run([str(python), "-B", "-m", "aiwf.native_compat", "verify", "--source-root", str(root), "--build-info", str(root / "native" / "native-build-info.json"), "--exe", str(root / "native" / "AESStudio.exe")], cwd=root, check=True, env=clean)
print("AES legacy profile migration:", migrate_legacy_profile(root, profile))
print("AES profile folder settings:", ensure_profile_launch(profile))
print("AES profile backend settings:", configure_backend(profile, backend, settings_policy))
# Record hashes of installed files. Mutable user configuration and all work stay outside this list.
owned = {}
for path in root.rglob("*"):
    if path.is_symlink():
        raise SystemExit(f"Unexpected link in installation: {path}")
    relative = path.relative_to(root).as_posix()
    if path.is_file() and path.name not in {"launch.json", "owned-files.json", "owned-files.tsv"} and relative != "packaging/uninstall_worker.exe":
        owned[relative] = file_hash(path)
(root / "owned-files.json").write_text(json.dumps({"schema":"aes-studio-owned-v1", "files":owned}, sort_keys=True), encoding="utf-8")
lines = ["AES-STUDIO-OWNED-V1"] + [digest + "\t" + relative for relative, digest in sorted(owned.items())]
(root / "owned-files.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")
print("AES Studio offline runtime is ready. No model was downloaded.")
