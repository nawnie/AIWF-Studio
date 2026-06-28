import type { PromptInsight, ResourceTone } from '../types'

type TextClassificationPipeline = {
  (input: string): Promise<Array<{ label?: string; score?: number }>>
  dispose?: () => Promise<void> | void
}

type ProgressInfo = {
  status?: string
  file?: string
  progress?: number
}

const PROMPT_MODEL_ID = 'Xenova/distilbert-base-uncased-finetuned-sst-2-english'

let promptPipeline: TextClassificationPipeline | null = null

export async function analyzePromptWithTransformers(
  prompt: string,
  negativePrompt: string,
  onProgress: (message: string, progress: number) => void,
): Promise<PromptInsight> {
  const trimmedPrompt = prompt.trim()
  if (!trimmedPrompt) {
    return buildInsight({
      status: 'idle',
      prompt: trimmedPrompt,
      negativePrompt,
      label: 'EMPTY',
      score: 0,
      progress: 0,
      summary: 'Add a prompt before running browser-side analysis.',
    })
  }

  try {
    const pipe = await getPromptPipeline(onProgress)
    onProgress('Running browser-side prompt model...', 96)
    const result = await pipe(trimmedPrompt)
    const first = result[0] ?? {}
    return buildInsight({
      status: 'ready',
      prompt: trimmedPrompt,
      negativePrompt,
      label: String(first.label ?? 'UNKNOWN'),
      score: typeof first.score === 'number' ? first.score : 0,
      progress: 100,
      summary: 'Browser-side prompt analysis completed with Transformers.js.',
    })
  } catch (error) {
    return buildInsight({
      status: 'error',
      prompt: trimmedPrompt,
      negativePrompt,
      label: 'ERROR',
      score: 0,
      progress: 0,
      summary: error instanceof Error ? error.message : 'Prompt analysis failed.',
    })
  }
}

export async function disposePromptInsightModel(): Promise<void> {
  if (promptPipeline?.dispose) {
    await promptPipeline.dispose()
  }
  promptPipeline = null
}

async function getPromptPipeline(
  onProgress: (message: string, progress: number) => void,
): Promise<TextClassificationPipeline> {
  if (promptPipeline) {
    return promptPipeline
  }

  onProgress('Loading Transformers.js prompt model...', 5)
  const transformers = await import('@huggingface/transformers')
  transformers.env.allowRemoteModels = true
  transformers.env.allowLocalModels = false
  transformers.env.useBrowserCache = true

  promptPipeline = await transformers.pipeline('sentiment-analysis', PROMPT_MODEL_ID, {
    dtype: 'q8',
    progress_callback: (info: ProgressInfo) => {
      const progress = typeof info.progress === 'number' ? Math.round(info.progress) : 0
      const status = info.status ?? 'loading'
      const file = info.file ? ` ${info.file}` : ''
      onProgress(`${status}${file}`.trim(), clamp(progress, 5, 95))
    },
  }) as TextClassificationPipeline

  onProgress('Prompt model ready.', 95)
  return promptPipeline
}

function buildInsight({
  status,
  prompt,
  negativePrompt,
  label,
  score,
  progress,
  summary,
}: {
  status: PromptInsight['status']
  prompt: string
  negativePrompt: string
  label: string
  score: number
  progress: number
  summary: string
}): PromptInsight {
  const words = prompt.split(/\s+/).filter(Boolean)
  const commaPhrases = prompt.split(',').map((item) => item.trim()).filter(Boolean)
  const negativeWords = negativePrompt.split(/\s+/).filter(Boolean)
  const qualityTerms = countMatches(prompt, [
    'cinematic',
    'detailed',
    'lighting',
    'composition',
    'texture',
    'sharp',
    'photorealistic',
    'illustration',
    'style',
  ])
  const cameraTerms = countMatches(prompt, ['lens', 'camera', 'shot', 'angle', 'depth', 'bokeh', 'macro'])
  const suggestions = buildSuggestions({
    wordCount: words.length,
    phraseCount: commaPhrases.length,
    negativeCount: negativeWords.length,
    qualityTerms,
    cameraTerms,
  })

  return {
    status,
    summary,
    modelLabel: label,
    modelScore: score,
    modelId: PROMPT_MODEL_ID,
    progress,
    signals: [
      signal('Prompt words', `${words.length}`, words.length >= 8 ? 'mint' : 'amber'),
      signal('Prompt clauses', `${commaPhrases.length}`, commaPhrases.length >= 3 ? 'mint' : 'neutral'),
      signal('Quality terms', `${qualityTerms}`, qualityTerms >= 2 ? 'mint' : 'amber'),
      signal('Camera/style terms', `${cameraTerms}`, cameraTerms >= 1 ? 'blue' : 'neutral'),
      signal('Negative terms', `${negativeWords.length}`, negativeWords.length >= 3 ? 'mint' : 'amber'),
      signal('Model confidence', `${Math.round(score * 100)}%`, score >= 0.8 ? 'blue' : 'neutral'),
    ],
    suggestions,
  }
}

function signal(label: string, value: string, tone: ResourceTone): PromptInsight['signals'][number] {
  return { label, value, tone }
}

function countMatches(value: string, needles: string[]): number {
  const normalized = value.toLowerCase()
  return needles.filter((needle) => normalized.includes(needle)).length
}

function buildSuggestions({
  wordCount,
  phraseCount,
  negativeCount,
  qualityTerms,
  cameraTerms,
}: {
  wordCount: number
  phraseCount: number
  negativeCount: number
  qualityTerms: number
  cameraTerms: number
}): string[] {
  const suggestions: string[] = []
  if (wordCount < 8) {
    suggestions.push('Add a clearer subject, setting, and visual outcome before generating.')
  }
  if (phraseCount < 3) {
    suggestions.push('Separate subject, environment, style, and lighting into short comma clauses.')
  }
  if (qualityTerms < 2) {
    suggestions.push('Add concrete quality/style terms such as lighting, composition, texture, or medium.')
  }
  if (cameraTerms < 1) {
    suggestions.push('Add camera, framing, or perspective language when you need tighter control.')
  }
  if (negativeCount < 3) {
    suggestions.push('Add a short negative prompt to protect against blur, text, watermark, and anatomy issues.')
  }
  if (suggestions.length === 0) {
    suggestions.push('Prompt structure looks ready for a first pass. Generate, then iterate from the receipt.')
  }
  return suggestions
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value))
}
