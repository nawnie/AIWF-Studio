/** Clear model-family and route names for setup bundles in Model Setup. */
export const SETUP_BUNDLE_LABELS: Record<string, string> = {
  flux: 'Flux model and complete local conditioning setup',
  'flux-components': 'Flux conditioning assets (CLIP-L, T5-XXL, VAE + tokenizers)',
  'flux-kontext': 'Flux Kontext image pipeline',
  'flux2-4b-components': 'Flux.2 Klein 4B components',
  'flux2-4b-base': 'Flux.2 Klein Base 4B Diffusers pipeline',
  'flux2-9b-components': 'Flux.2 Klein 9B components',
  flux2: 'Flux.2 Klein 4B pipeline',
  zimage: 'Z-Image model and components',
  'qwen-image': 'Qwen Image 2512 Diffusers pipeline',
  'qwen-image-original': 'Qwen Image original Diffusers pipeline',
  'qwen-nunchaku': 'Qwen Image Nunchaku model + isolated runtime (generation smoke pending)',
  krea2: 'Krea 2 Diffusers pipeline',
  'krea2-raw': 'Krea 2 Raw Diffusers pipeline',
  'krea2-low': 'Krea 2 split assets (not runnable by current loader)',
  'krea2-mid': 'Krea 2 split assets (not runnable by current loader)',
  'krea2-high': 'Krea 2 split assets (not runnable by current loader)',
  sana: 'Sana Sprint model',
  'sana-sprint-16b': 'Sana Sprint 1.6B image model',
  'sana-16b': 'Sana 1.6B image model',
  'sana-video': 'Sana Video 2B 480p pipeline',
  'sana-video-720p': 'Sana Video 2B 720p pipeline',
  'wan-ti2v-diffusers': 'Wan 2.2 TI2V 5B Diffusers snapshot (large download)',
  'wan-ti2v-support': 'Wan TI2V 5B support assets (UMT5, tokenizer, scheduler + 48-channel VAE)',
  'wan-14b-components': 'Wan 2.2 A14B high/low support (UMT5, tokenizer, scheduler + Wan 2.1 VAE)',
  'ltx-2b': 'LTX Video 0.9.5 2B text-to-video + T5-XXL fp16 (~16 GB)',
  ltx23: 'LTX 2.3 distilled model assets (engine setup required)',
  'ltx23-one-stage': 'LTX 2.3 one-stage BF16 model assets (22B; blocked on Windows)',
  'ltx23-one-stage-fp8': 'LTX 2.3 one-stage FP8 model and Gemma text encoder (supported on Windows)',
  video: 'Wan 2.2 video setup (~29 GB; high/low GGUF, VAE, shared components)',
  sd: 'Stable Diffusion 1.5 setup',
  sdxl: 'Stable Diffusion XL setup',
  sd35: 'Stable Diffusion 3.5 Medium model',
  'sd-components': 'Stable Diffusion 1.5 support assets',
  'sdxl-components': 'SDXL support assets',
  'sd35-components': 'Stable Diffusion 3.5 support assets',
  rife: 'RIFE frame interpolation model',
  seg: 'Segmentation tools',
  faceswap: 'Face swap model',
  embeddings: 'Textual inversion embeddings',
}

const FALLBACK_TERMS: Record<string, string> = {
  controlnet: 'ControlNet',
  sd15: 'SD 1.5',
  sdxl: 'SDXL',
  sd35: 'SD 3.5',
  ltx23: 'LTX 2.3',
  vae: 'VAE',
  gguf: 'GGUF',
}

export function formatSetupBundleLabel(bundleKey: string): string {
  const explicit = SETUP_BUNDLE_LABELS[bundleKey]
  if (explicit) return explicit
  return bundleKey
    .split('-')
    .map((part) => FALLBACK_TERMS[part] ?? `${part.charAt(0).toUpperCase()}${part.slice(1)}`)
    .join(' ')
}
