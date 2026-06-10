from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.domain.generation import GenerationMode, GenerationRequest
from aiwf.services.generation import GenerationService
from aiwf.services.prompt_processor import PromptProcessorService


def test_generation_service_resolves_prompts(tmp_path: Path):
    flags = RuntimeFlags(data_dir=tmp_path)
    wildcards = tmp_path / "wildcards"
    wildcards.mkdir()
    (wildcards / "color.txt").write_text("blue\n", encoding="utf-8")

    models = MagicMock()
    models.expand_prompt_keywords.side_effect = lambda text: text
    prompts = PromptProcessorService(flags, UserSettings(), models)

    service = GenerationService(
        backend=MagicMock(),
        store=MagicMock(),
        metadata=MagicMock(),
        queue=MagicMock(),
        events=MagicMock(),
        settings=UserSettings(),
        prompts=prompts,
    )

    request = GenerationRequest(
        mode=GenerationMode.TXT2IMG,
        prompt="sky __color__",
        seed=7,
    )
    resolved = service._resolve_prompts(request)
    assert resolved.prompt == "sky blue"