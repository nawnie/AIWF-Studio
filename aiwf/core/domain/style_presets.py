from __future__ import annotations

from aiwf.core.domain.prompt_style import PromptStyle

DEFAULT_PROMPT_STYLES: tuple[PromptStyle, ...] = (
    PromptStyle(
        name="Quality — Standard",
        prompt="a high quality photo of {prompt}, masterpiece, best quality, highly detailed, sharp focus",
        negative_prompt="{prompt}, worst quality, low quality, normal quality, lowres, blurry, watermark, text, signature",
    ),
    PromptStyle(
        name="Quality — Photoreal",
        prompt="a photorealistic photograph of {prompt}, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3",
        negative_prompt="{prompt}, cartoon, painting, illustration, anime, 3d render, lowres, blurry, bad anatomy",
    ),
    PromptStyle(
        name="Quality — Studio portrait",
        prompt="a professional studio portrait of {prompt}, soft natural lighting, shallow depth of field, high quality",
        negative_prompt="{prompt}, bad anatomy, bad hands, deformed face, cross-eyed, disfigured, extra fingers",
    ),
    PromptStyle(
        name="Detail — General",
        prompt="{prompt}, (intricate details:1.1), (highly detailed:1.1), fine texture, crisp details",
        negative_prompt="{prompt}, low detail, blurry, soft focus, muddy, jpeg artifacts",
    ),
    PromptStyle(
        name="Detail — Portrait",
        prompt="{prompt}, (detailed face:1.2), (beautiful eyes:1.1), (detailed skin texture:1.1), soft natural lighting",
        negative_prompt="{prompt}, bad anatomy, bad hands, deformed face, cross-eyed, disfigured, extra fingers",
    ),
    PromptStyle(
        name="Detail — Hands",
        prompt="{prompt}, (detailed hands:1.2), natural hand pose, correct finger count",
        negative_prompt="{prompt}, bad hands, malformed hands, extra fingers, missing fingers, fused fingers",
    ),
    PromptStyle(
        name="Style — Anime",
        prompt="an anime illustration of {prompt}, vibrant colors, clean lineart, cel shading, detailed background",
        negative_prompt="{prompt}, lowres, bad anatomy, bad hands, text, watermark, blurry, photorealistic",
    ),
    PromptStyle(
        name="Style — Cinematic",
        prompt="a cinematic movie still of {prompt}, dramatic lighting, color graded, volumetric light, film grain",
        negative_prompt="{prompt}, flat lighting, amateur photo, oversaturated, low contrast, snapshot",
    ),
    PromptStyle(
        name="Style — Fantasy art",
        prompt="a fantasy art painting of {prompt}, intricate details, ethereal lighting, digital painting, artstation",
        negative_prompt="{prompt}, photo, photograph, realistic, lowres, blurry, plain background",
    ),
    PromptStyle(
        name="Style — 3D render",
        prompt="{prompt}, 3d render, octane render, unreal engine, ray tracing, global illumination, highly detailed",
        negative_prompt="{prompt}, 2d, flat, sketch, painting, low poly, lowres, blurry",
    ),
    PromptStyle(
        name="Style — Illustration",
        prompt="a digital illustration of {prompt}, concept art, painterly, detailed background, artstation",
        negative_prompt="{prompt}, photo, photograph, lowres, blurry, watermark, text",
    ),
    PromptStyle(
        name="Negative — Standard cleanup",
        prompt="{prompt}",
        negative_prompt="{prompt}, worst quality, low quality, blurry, jpeg artifacts, watermark, text, logo, cropped",
    ),
    PromptStyle(
        name="Negative — Anatomy cleanup",
        prompt="{prompt}",
        negative_prompt="{prompt}, bad anatomy, bad proportions, deformed, disfigured, extra limbs, missing limbs, mutated hands",
    ),
)

BUILTIN_STYLE_NAMES: frozenset[str] = frozenset(style.name for style in DEFAULT_PROMPT_STYLES)
BUILTIN_STYLES_BY_NAME: dict[str, PromptStyle] = {style.name: style for style in DEFAULT_PROMPT_STYLES}


def is_builtin_style(name: str | None) -> bool:
    return bool(name) and name in BUILTIN_STYLE_NAMES


def get_builtin_style(name: str | None) -> PromptStyle | None:
    if not name:
        return None
    preset = BUILTIN_STYLES_BY_NAME.get(name)
    return preset.model_copy() if preset is not None else None


def ensure_default_prompt_styles(settings) -> bool:
    """Add missing built-in presets without overwriting user edits."""
    existing = {style.name for style in settings.prompt_styles}
    added = False
    for preset in DEFAULT_PROMPT_STYLES:
        if preset.name not in existing:
            settings.prompt_styles.append(preset.model_copy())
            added = True
    if added:
        settings.prompt_styles.sort(key=lambda item: item.name.lower())
    return added


def style_preview_text(style: PromptStyle, sample_prompt: str = "a woman in a garden") -> str:
    from aiwf.core.domain.prompt_style import apply_prompt_style

    positive, negative = apply_prompt_style(style, sample_prompt, "blurry")
    lines = [f"**Positive example**  \n`{positive}`"]
    if negative:
        lines.append(f"**Negative example**  \n`{negative}`")
    return "\n\n".join(lines)