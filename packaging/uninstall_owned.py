"""Remove unchanged installer-owned files only, preserving user data and edits."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys

def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
root = Path(sys.argv[1]).resolve(strict=True)
pending = Path(sys.argv[2])
manifest = json.loads((root / "owned-files.json").read_text(encoding="utf-8"))
if manifest.get("schema") != "aes-studio-owned-v1":
    raise SystemExit("Unrecognized install manifest; no files removed")
locked = []
for rel, expected in manifest["files"].items():
    relpath = Path(rel)
    if relpath.is_absolute() or ".." in relpath.parts:
        raise SystemExit("Unsafe manifest entry; no further files removed")
    p = root / relpath
    if any(part.is_symlink() or (part.exists() and bool(part.stat(follow_symlinks=False).st_file_attributes & 0x400)) for part in [root, *list(p.parents)[:-1]] if part != root.parent and (part == root or root in part.parents)):
        continue
    try:
        if not p.is_file() or p.is_symlink() or file_hash(p) != expected:
            continue
        p.unlink()
    except PermissionError:
        locked.append(str(p))
    except FileNotFoundError:
        pass
pending.write_text("\n".join(locked) + ("\n" if locked else ""), encoding="utf-16-le", newline="")
for directory in sorted((p for p in root.rglob("*") if p.is_dir() and not p.is_symlink()), key=lambda p: len(p.parts), reverse=True):
    try: directory.rmdir()
    except OSError: pass

