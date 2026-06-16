from __future__ import annotations

import random
from pathlib import Path

from aiwf.core.domain.prompt_dynamics import resolve_dynamic_prompt, resolve_variants, resolve_wildcards


def test_resolve_variants_picks_one_option():
    rng = random.Random(0)
    result = resolve_variants("a {red|blue|green} coat", rng=rng)
    assert result in {"a red coat", "a blue coat", "a green coat"}


def test_resolve_variants_nested_braces():
    rng = random.Random(1)
    result = resolve_variants("{a|{b|c}}", rng=rng)
    assert result in {"a", "b", "c"}


def test_resolve_wildcards_replaces_token(tmp_path: Path):
    wildcards = tmp_path / "wildcards"
    wildcards.mkdir()
    (wildcards / "color.txt").write_text("red\ngreen\nblue\n", encoding="utf-8")
    rng = random.Random(2)
    result = resolve_wildcards("a __color__ sky", wildcards, rng=rng)
    assert result in {"a red sky", "a green sky", "a blue sky"}


def test_resolve_dynamic_prompt_chains_variants_and_wildcards(tmp_path: Path):
    wildcards = tmp_path / "wildcards"
    wildcards.mkdir()
    (wildcards / "animal.txt").write_text("cat\ndog\n", encoding="utf-8")
    rng = random.Random(3)
    result = resolve_dynamic_prompt("{cute|happy} __animal__", wildcards, rng=rng)
    assert "cat" in result or "dog" in result
    assert result.startswith("cute ") or result.startswith("happy ")