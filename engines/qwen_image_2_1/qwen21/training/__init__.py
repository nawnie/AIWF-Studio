"""LoRA training helpers for Qwen-Image 2.1 (config generation + subprocess runner).

Two trainers are wrapped:

* **ai-toolkit** (ostris) — arch ``qwen_image_2``. The recommended path for a
  16 GB card: int8 ConvRot quantization of the DiT and text encoder, low-VRAM
  mode, layer offloading, cached latents and text embeddings. A 12 GB RTX 4070
  run (31 images, 2000 steps, rank 16) is documented in the README sources.
* **DiffSynth-Studio** (ModelScope) — the Qwen team's official training
  script ``examples/qwen_image_21/model_training/train.py``. Reference recipe;
  bf16 weights without quantization need more than 16 GB unless offloaded.

Nothing here imports torch; the heavy work runs in the trainer's own venv.
"""
