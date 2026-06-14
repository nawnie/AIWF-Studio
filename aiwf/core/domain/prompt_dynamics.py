from __future__ import annotations

import random
import re
from pathlib import Path

RE_WILDCARD = re.compile(r"__(?![_\s])([a-zA-Z0-9_./*-]+)__")
MAX_RESOLVE_PASSES = 64


def resolve_variants(text: str, rng: random.Random | None = None) -> str:
    """Replace `{a|b|c}` with one equal-probability option."""
    rng = rng or random.Random()
    result = text
    for _ in range(MAX_RESOLVE_PASSES):
        match = None
        depth = 0
        start = -1
        for index, char in enumerate(result):
            if char == "{" and (index == 0 or result[index - 1] != "\\"):
                if depth == 0:
                    start = index
                depth += 1
            elif char == "}" and (index == 0 or result[index - 1] != "\\"):
                depth -= 1
                if depth == 0 and start >= 0:
                    match = (start, index + 1)
                    break
                if depth < 0:
                    depth = 0
                    start = -1
        if match is None:
            break
        start, end = match
        inner = result[start + 1 : end - 1]
        options = [part.strip() for part in inner.split("|") if part.strip()]
        if not options:
            break
        choice = rng.choice(options)
        result = result[:start] + choice + result[end:]
    return re.sub(r"\s{2,}", " ", result).strip()


def _wildcard_files(token: str, wildcards_dir: Path) -> list[Path]:
    token = token.replace("\\", "/").strip("/")
    if "*" in token:
        pattern = token.replace("/", "\\")
        matches = sorted(wildcards_dir.rglob(pattern + ".txt"))
        if not matches:
            matches = sorted(wildcards_dir.rglob(pattern))
        return [path for path in matches if path.is_file()]
    direct = wildcards_dir / f"{token}.txt"
    if direct.is_file():
        return [direct]
    nested = wildcards_dir / token
    if nested.with_suffix(".txt").is_file():
        return [nested.with_suffix(".txt")]
    return []


def _pick_wildcard_line(path: Path, rng: random.Random) -> str:
    try:
        lines = [
            line.strip()
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    except OSError:
        return ""
    if not lines:
        return ""
    return rng.choice(lines)


def resolve_wildcards(text: str, wildcards_dir: Path, rng: random.Random | None = None) -> str:
    """Replace `__name__` with a random line from `wildcards/name.txt`."""
    rng = rng or random.Random()
    if not wildcards_dir.exists():
        return text
    result = text
    for _ in range(MAX_RESOLVE_PASSES):
        match = RE_WILDCARD.search(result)
        if not match:
            break
        token = match.group(1)
        files = _wildcard_files(token, wildcards_dir)
        if not files:
            break
        choice = _pick_wildcard_line(rng.choice(files), rng)
        result = result[: match.start()] + choice + result[match.end() :]
    return re.sub(r"\s{2,}", " ", result).strip()


def resolve_dynamic_prompt(text: str, wildcards_dir: Path, rng: random.Random | None = None) -> str:
    """Apply wildcard and variant resolution until stable."""
    rng = rng or random.Random()
    result = text
    for _ in range(MAX_RESOLVE_PASSES):
        updated = resolve_variants(resolve_wildcards(result, wildcards_dir, rng), rng)
        if updated == result:
            break
        result = updated
    return result