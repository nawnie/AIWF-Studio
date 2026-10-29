from __future__ import annotations

import re
import inspect

EMPHASIS_PATTERN = re.compile(r"(?<!\\)[(\[]")


def prompt_uses_emphasis(text: str | None) -> bool:
    return bool(text and EMPHASIS_PATTERN.search(text))


def _is_sdxl_pipe(pipe) -> bool:
    return hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is not None


def _is_sd3_pipe(pipe) -> bool:
    return hasattr(pipe, "transformer") and hasattr(pipe, "text_encoder_3")


def _pipe_accepts_kwarg(pipe, name: str) -> bool:
    try:
        sig = inspect.signature(pipe.__call__)
    except (TypeError, ValueError):
        return False
    if name in sig.parameters:
        return True
    return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values())


def build_prompt_kwargs(pipe, prompt: str, negative_prompt: str | None, clip_skip: int) -> dict:
    """Return kwargs for diffusers pipe calls — uses Compel when emphasis syntax is present."""
    negative = negative_prompt or ""
    if not prompt_uses_emphasis(prompt) and not prompt_uses_emphasis(negative):
        kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or None,
        }
        if clip_skip and clip_skip > 1 and _pipe_accepts_kwarg(pipe, "clip_skip"):
            kwargs["clip_skip"] = int(clip_skip)
        return kwargs

    if _is_sd3_pipe(pipe):
        kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or None,
        }
        if clip_skip and clip_skip > 1 and _pipe_accepts_kwarg(pipe, "clip_skip"):
            kwargs["clip_skip"] = int(clip_skip)
        return kwargs

    try:
        from compel import Compel, ReturnedEmbeddingsType

        try:
            from compel import DiffusersTextualInversionManager

            ti_manager = DiffusersTextualInversionManager(pipe)
        except Exception:
            ti_manager = None
    except ImportError:
        return {
            "prompt": prompt,
            "negative_prompt": negative_prompt or None,
        }

    if _is_sdxl_pipe(pipe):
        # SDXL always uses penultimate hidden states (the model was trained
        # that way); compel has no clip_skip parameter.
        compel = Compel(
            tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
            text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
            truncate_long_prompts=False,
            textual_inversion_manager=ti_manager,
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

    # Compel exposes clip-skip via embedding type: penultimate hidden states
    # is equivalent to "Clip skip: 2" (deeper skips fall back to 2).
    embeddings_type = (
        ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NORMALIZED
        if clip_skip and clip_skip > 1
        else ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED
    )
    compel = Compel(
        tokenizer=pipe.tokenizer,
        text_encoder=pipe.text_encoder,
        truncate_long_prompts=False,
        returned_embeddings_type=embeddings_type,
        textual_inversion_manager=ti_manager,
    )
    prompt_embeds = compel(prompt)
    negative_embeds = compel(negative) if negative else compel("")
    return {
        "prompt_embeds": prompt_embeds,
        "negative_prompt_embeds": negative_embeds,
    }
