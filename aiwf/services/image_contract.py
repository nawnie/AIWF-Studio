"""Small request limits shared by image job adapters.

Keep these values separate from any particular engine so lightweight native
backend discovery does not import the legacy ComfyUI job adapter.
"""

ASPECT_RATIOS = {
    "1:1": "1:1 (Square)",
    "2:3": "2:3 (Portrait Photo)",
    "3:2": "3:2 (Photo)",
    "3:4": "3:4 (Portrait Standard)",
    "4:3": "4:3 (Standard)",
    "9:16": "9:16 (Portrait Widescreen)",
    "16:9": "16:9 (Widescreen)",
    "21:9": "21:9 (Ultrawide)",
}
QUALITY_MEGAPIXELS = {"draft": 0.6, "standard": 1.0}
MAX_PROMPT_CHARS = 2000
MAX_SEED = 2**53 - 1  # largest seed that survives a JSON round trip in every client
MAX_IMAGE_BYTES = 64 * 1024 * 1024
MAX_JOBS_KEPT = 50
