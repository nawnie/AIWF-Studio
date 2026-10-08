import assert from 'node:assert/strict'
import test from 'node:test'

import { DOWNLOAD_CATEGORY_LABELS, formatDownloadCategoryLabel } from '../src/layouts/studio/downloadLabels.ts'

test('every backend download category has an explicit user-facing label', () => {
  const backendCategories = [
    'checkpoint', 'sd_singlefile_config', 'lora', 'vae', 'controlnet', 'preprocessor', 'upscaler', 'esrgan',
    'gfpgan', 'codeformer', 'faceswap', 'embedding', 'hypernetwork', 'wan_safetensor', 'wan_gguf', 'wan_diffusers',
    'wan_lora', 'wan_vae', 'wan_text_encoder', 'flux_unet_safetensor', 'flux_unet_gguf', 'flux_text_encoder',
    'flux_vae', 'flux_tokenizer', 'flux2_unet_safetensor', 'flux2_unet_gguf', 'flux2_components', 'flux2_diffusers',
    'flux_kontext_diffusers', 'z_image_unet_safetensor', 'z_image_unet_gguf', 'z_image_components',
    'krea2_unet_safetensor', 'krea2_text_encoder', 'krea2_vae', 'krea2_diffusers', 'anima_unet_safetensor',
    'anima_text_encoder', 'anima_vae', 'qwen_image_diffusers', 'qwen_image_nunchaku', 'sana_diffusers',
    'sana_video_diffusers', 'ltx_checkpoint', 'ltx_gguf', 'ltx_upscaler', 'ltx_lora', 'ltx_vae',
    'ltx_audio_vae', 'ltx_text_encoder', 'ltx_tokenizer', 'llm_gguf', 'llm_safetensor', 'rife', 'sam', 'other',
  ]

  assert.deepEqual(Object.keys(DOWNLOAD_CATEGORY_LABELS).sort(), backendCategories.sort())
  for (const category of backendCategories) {
    assert.equal(formatDownloadCategoryLabel(category), DOWNLOAD_CATEGORY_LABELS[category])
  }
})

test('category labels retain model family and asset meaning', () => {
  assert.equal(formatDownloadCategoryLabel('sd_singlefile_config'), 'Stable Diffusion support config')
  assert.equal(formatDownloadCategoryLabel('wan_safetensor'), 'Wan transformer (safetensors)')
  assert.equal(formatDownloadCategoryLabel('wan_text_encoder'), 'Wan UMT5-XXL text encoder')
  assert.equal(formatDownloadCategoryLabel('ltx_checkpoint'), 'LTX video checkpoint (version shown per model)')
  assert.equal(formatDownloadCategoryLabel('ltx_audio_vae'), 'LTX 2.3 audio VAE')
  assert.equal(formatDownloadCategoryLabel('llm_gguf'), 'LLM GGUF')
})
