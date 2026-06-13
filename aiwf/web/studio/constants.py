from __future__ import annotations

MODES = [
    ("txt2img", "Text"),
    ("img2img", "Image2Image"),
    ("inpaint", "Inpaint"),
]

MODE_TITLES = {
    "txt2img": '<span class="aiwf-mode-kicker">Mode</span> Text to image',
    "img2img": '<span class="aiwf-mode-kicker">Mode</span> Image to image',
    "inpaint": '<span class="aiwf-mode-kicker">Mode</span> Inpaint & edit',
}

TOOLBAR_HINTS = {
    "txt2img": "Prompt → generate",
    "img2img": "Upload → vary",
    "inpaint_edit": "Paint mask → generate",
    "inpaint_result": "Original or last result — Generate again",
}

EMPTY_CANVAS = {
    "txt2img": (
        '<div class="aiwf-empty-state">'
        '<div class="aiwf-empty-state-icon" aria-hidden="true">◇</div>'
        '<p class="aiwf-empty-state-title">Canvas ready</p>'
        '<p class="aiwf-empty-state-desc">Write a prompt and press Generate. '
        "Live previews appear here while the model runs.</p></div>"
    ),
    "img2img": (
        '<div class="aiwf-empty-state">'
        '<div class="aiwf-empty-state-icon" aria-hidden="true">◇</div>'
        '<p class="aiwf-empty-state-title">Source image required</p>'
        '<p class="aiwf-empty-state-desc">Upload a reference image, tune denoising strength, '
        "then generate your variation.</p></div>"
    ),
    "inpaint_edit": (
        '<div class="aiwf-empty-state">'
        '<div class="aiwf-empty-state-icon" aria-hidden="true">◇</div>'
        '<p class="aiwf-empty-state-title">Paint your mask</p>'
        '<p class="aiwf-empty-state-desc">Brush over the region to replace. White areas are inpainted; '
        "the rest of the image stays intact.</p></div>"
    ),
    "inpaint_result": (
        '<div class="aiwf-empty-state">'
        '<div class="aiwf-empty-state-icon" aria-hidden="true">◇</div>'
        '<p class="aiwf-empty-state-title">Result canvas</p>'
        '<p class="aiwf-empty-state-desc">Your output appears here. Pick <strong>Original image</strong> or '
        "<strong>Last result</strong> under Inpaint source, then Generate again with the same mask.</p></div>"
    ),
}