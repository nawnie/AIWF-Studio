from __future__ import annotations

import re

EMPHASIS_PATTERN = re.compile(r"(?<!\\)[(\[]")


def prompt_uses_emphasis(text: str | None) -> bool:
    return bool(text and EMPHASIS_PATTERN.search(text))


def _is_sdxl_pipe(pipe) -> bool:
    return hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is not None


def build_prompt_kwargs(pipe, prompt: str, negative_prompt: str | None, clip_skip: int) -> dict:
    """Return kwargs for diffusers pipe calls — uses Compel when emphasis syntax is present."""
    negative = negative_prompt or ""
    if not prompt_uses_emphasis(prompt) and not prompt_uses_emphasis(negative):
        return {
            "prompt": prompt,
            "negative_prompt": negative_prompt or None,
        }

    try:
        from compel import Compel, ReturnedEmbeddingsType
    except ImportError:
        return {
            "prompt": prompt,
            "negative_prompt": negative_prompt or None,
        }

    if _is_sdxl_pipe(pipe):
        compel = Compel(
            tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
            text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
            truncate_long_prompts=False,
            clip_skip=clip_skip,
        )
        prompt_embeds, pooled_prompt_embeds = compel(prompt)
        if negative:
            negative_embeds, negative_pooled_embeds = compel(negative)
        else:
            negative_embeds, negative_pooled_embeds = compel("")
        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "negative_prompt_embeds": negative_embeds,
            "negative_pooled_prompt_embeds": negative_pooled_embeds,
        }

    compel = Compel(
        tokenizer=pipe.tokenizer,
        text_encoder=pipe.text_encoder,
        truncate_long_prompts=False,
        clip_skip=clip_skip,
    )
    prompt_embeds = compel(prompt)
    negative_embeds = compel(negative) if negative else compel("")
    return {
        "prompt_embeds": prompt_embeds,
        "negative_prompt_embeds": negative_embeds,
    }