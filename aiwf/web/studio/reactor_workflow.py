from __future__ import annotations

from collections.abc import Callable

from aiwf.bootstrap import AppContext
from aiwf.core.domain.enhance import RestoreOptions
from aiwf.core.domain.faceswap import FaceSwapOptions
from aiwf.dev.diagnostics import trace_exception_safe

REACTOR_GENERATION_ARG_COUNT = 12


def build_reactor_generation_postprocess(
    ctx: AppContext,
    *,
    enabled: bool,
    source_image,
    source_face_index,
    target_face_index,
    restore_face: bool,
    restorer_id,
    restore_visibility,
    codeformer_weight,
    model_id,
    gender_source,
    gender_target,
    mask_face: bool,
) -> Callable | None:
    """Build the optional ReActor postprocess used by Studio generation."""

    if not enabled or source_image is None:
        return None

    def postprocess(image):
        try:
            options = FaceSwapOptions(
                source_face_index=max(0, int(source_face_index or 0)),
                target_face_index=int(target_face_index if target_face_index is not None else -1),
                model_id=model_id or "inswapper_128",
                gender_source=int(gender_source or 0),
                gender_target=int(gender_target or 0),
                mask_face=bool(mask_face),
                restore_face=bool(restore_face),
                restorer_id=restorer_id,
                restore_visibility=float(restore_visibility),
                codeformer_weight=float(codeformer_weight),
            )
            restore_fn = None
            if restore_face and restorer_id:
                def restore_fn(im):
                    return ctx.enhance.restore(
                        im,
                        RestoreOptions(
                            model_id=restorer_id,
                            visibility=float(restore_visibility),
                            codeformer_weight=float(codeformer_weight),
                        ),
                    )

            return ctx.faceswap.swap(image, source_image, options, restore_fn=restore_fn)
        except Exception as exc:
            trace_exception_safe("studio.reactor_at_gen", exc)
            return image

    return postprocess


def build_reactor_generation_postprocess_from_args(ctx: AppContext, reactor_args: tuple) -> Callable | None:
    if not reactor_args:
        return None
    if len(reactor_args) != REACTOR_GENERATION_ARG_COUNT:
        raise ValueError(
            f"Expected {REACTOR_GENERATION_ARG_COUNT} ReActor generation arguments, got {len(reactor_args)}."
        )
    (
        enabled,
        source_image,
        source_face_index,
        target_face_index,
        restore_face,
        restorer_id,
        restore_visibility,
        codeformer_weight,
        model_id,
        gender_source,
        gender_target,
        mask_face,
    ) = reactor_args
    return build_reactor_generation_postprocess(
        ctx,
        enabled=bool(enabled),
        source_image=source_image,
        source_face_index=source_face_index,
        target_face_index=target_face_index,
        restore_face=bool(restore_face),
        restorer_id=restorer_id,
        restore_visibility=restore_visibility,
        codeformer_weight=codeformer_weight,
        model_id=model_id,
        gender_source=gender_source,
        gender_target=gender_target,
        mask_face=bool(mask_face),
    )
