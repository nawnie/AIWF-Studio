import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  AlertCircle,
  CheckCircle2,
  Download,
  DownloadCloud,
  FolderOpen,
  Loader2,
  Music,
  Play,
  Plus,
  Radio,
  Scissors,
  Search,
  Settings2,
  SlidersHorizontal,
  Sparkles,
  Volume2,
  Waves,
} from 'lucide-react'
import {
  fetchProAudioStatus,
  formatApiError,
  generateProAudio,
  fetchProAudioProjects,
  prepareProAudioModel,
  ProApiError,
  loadProAudioProject,
  saveProAudioProject,
  runProSetupAction,
  setProAudioResearchMode,
  type ProAudioProjectManifest,
  type ProAudioProjectOptions,
  type ProAudioProjectSummary,
  type ProAudioGenerateResult,
  type ProAudioStatus,
} from '../../api'
import type { LayoutProps } from './LayoutTypes'
import { audioProjectAttribution } from './audioProjectAttribution'
import { formatRouteLifecycleStatus } from './modelLabels'
import { API_BASE } from '../../apiBase'
import { AudioProjectControls, type AudioProjectSaveRequest } from './AudioProjectControls'
import './AudioProjectControls.css'
import './studioLayouts.css'

type AudioKind = 'music' | 'sfx'

function storedAudioModel(kind: AudioKind): string {
  try { return window.localStorage.getItem(`aiwf.audio-studio.${kind}-model`) || '' } catch { return '' }
}

function saveAudioModel(kind: AudioKind, modelId: string): void {
  try { window.localStorage.setItem(`aiwf.audio-studio.${kind}-model`, modelId) } catch { /* Storage may be disabled. */ }
}

function clearStoredAudioModel(kind: AudioKind): void {
  try { window.localStorage.removeItem(`aiwf.audio-studio.${kind}-model`) } catch { /* Storage may be disabled. */ }
}

const AUDIO_PRESETS: Array<{ label: string; kind: AudioKind | null; note: string; title: string }> = [
  { label: 'Music Bed', kind: 'music', note: 'MusicGen', title: 'Generate a text-guided music clip.' },
  { label: 'Sound Effects', kind: 'sfx', note: 'MMAudio', title: 'Generate text-guided sound effects.' },
  { label: 'Video Soundtrack', kind: null, note: 'Video Lab', title: 'Use Pro Video Lab for video-conditioned audio.' },
  { label: 'Voice Cleanup', kind: null, note: 'Gradio lab', title: 'Use Gradio Audio Lab for cleanup and mixing.' },
  { label: 'Loudness Master', kind: null, note: 'Gradio lab', title: 'Use Gradio Audio Lab for loudness processing.' },
]
const AUDIO_EFFECTS = ['Noise Gate', 'EQ', 'Compressor', 'Limiter', 'Stereo Width', 'Reverb Send']
const AUDIO_MODELS = [
  { id: 'music', label: 'MusicGen small' },
  { id: 'sfx', label: 'MMAudio small 16k' },
  { id: 'lab', label: 'Audio Lab DSP' },
  { id: 'mux', label: 'FFmpeg mux' },
] as const

const AUDIO_PREPARE_RETRY_DELAYS_MS = [500, 1000, 2000, 4000, 8000, 8000, 8000] as const

function isTransientAudioPrepareConflict(error: unknown): boolean {
  return error instanceof ProApiError && error.status === 409
}

export function AudioStudioLayout({
  settings,
  runtime,
  recentOutputs,
  selectedModelName,
  statusMessage,
  isGenerating,
  onSettingsChange,
  onSendToWorkflow,
  onOpenModelSorter,
  onOpenSettings,
}: LayoutProps) {
  const [dockMode, setDockMode] = useState<'tracks' | 'scenes' | 'mixer'>('tracks')
  const [activeEffect, setActiveEffect] = useState('EQ')
  const [setupStatus, setSetupStatus] = useState<ProAudioStatus | null>(null)
  const [setupBusy, setSetupBusy] = useState(false)
  const [audioPrepareBusy, setAudioPrepareBusy] = useState(false)
  const [audioBusy, setAudioBusy] = useState(false)
  const [audioKind, setAudioKind] = useState<AudioKind>('music')
  const [audioDuration, setAudioDuration] = useState(8)
  const [audioOptionState, setAudioOptionState] = useState<Pick<ProAudioProjectOptions, 'negative_prompt' | 'temperature' | 'top_k'>>({ negative_prompt: '', temperature: 1, top_k: 250 })
  const [audioResult, setAudioResult] = useState<ProAudioGenerateResult | null>(null)
  const [audioRevision, setAudioRevision] = useState(0)
  const [audioError, setAudioError] = useState('')
  const [audioProjects, setAudioProjects] = useState<ProAudioProjectSummary[]>([])
  const [currentProjectId, setCurrentProjectId] = useState<string | null>(null)
  const [projectBusy, setProjectBusy] = useState(false)
  const [audioModelOverride, setAudioModelOverride] = useState(() => storedAudioModel('music'))
  const audioPreparationKeyRef = useRef('')
  const [audioPreparedKey, setAudioPreparedKey] = useState('')
  const [audioPreparationErrorKey, setAudioPreparationErrorKey] = useState('')
  const [audioPreparationRetry, setAudioPreparationRetry] = useState(0)
  const sceneRows = useMemo(() => recentOutputs.slice(0, 6), [recentOutputs])
  const generationBusy = isGenerating || audioBusy
  const selectedAudioReady = (audioKind === 'music' ? setupStatus?.musicReady : setupStatus?.sfxReady)
  const selectedAudioModel = audioModelOverride || (audioKind === 'music' ? setupStatus?.defaults.music : setupStatus?.defaults.sfx)
  const audioModelChoices = audioKind === 'music' ? setupStatus?.models.music ?? [] : setupStatus?.models.sfx ?? []
  // no hardcoded fallback model: in commercial-safe mode the server offers only commercially
  // licensed models, and none until one is installed
  const effectiveAudioModel = selectedAudioModel || audioModelChoices[0]?.id || ''
  const noModelLabel = setupStatus && !setupStatus.researchMode ? 'No commercial-safe model installed yet' : 'Audio model'
  const selectedAudioModelLabel = audioModelChoices.find((choice) => choice.id === effectiveAudioModel)?.label || effectiveAudioModel || noModelLabel
  const summaryMusicModelId = audioKind === 'music'
    ? effectiveAudioModel
    : storedAudioModel('music') || setupStatus?.defaults.music || ''
  const summarySfxModelId = audioKind === 'sfx'
    ? effectiveAudioModel
    : storedAudioModel('sfx') || setupStatus?.defaults.sfx || ''
  const selectedMusicChoice = (setupStatus?.models.music ?? []).find((choice) => choice.id === summaryMusicModelId)
  const selectedSfxChoice = (setupStatus?.models.sfx ?? []).find((choice) => choice.id === summarySfxModelId)
  const selectedMusicEngineLabel = selectedMusicChoice?.label || noModelLabel
  const selectedSfxEngineLabel = selectedSfxChoice?.label || noModelLabel
  const [researchBusy, setResearchBusy] = useState(false)
  const selectedChoice = audioModelChoices.find((choice) => choice.id === effectiveAudioModel)
  const selectedAudioNeedsInstall = Boolean(selectedChoice?.installable && selectedChoice.installed === false)
  const selectedAudioSetupAction = selectedChoice?.setupRoute?.setupAction || ''
  const selectedAudioNeedsRuntimeSetup = Boolean(
    selectedChoice
      && selectedChoice.installable !== false
      && selectedChoice.available === false,
  )
  const currentAudioPreparationKey = `${audioKind}:${effectiveAudioModel}`
  const selectedRouteStatus = audioPreparationErrorKey === currentAudioPreparationKey
    ? 'failed'
    : selectedChoice?.routeStatus
  const selectedSharedAudioAssets = effectiveAudioModel.startsWith('mmaudio:')
    ? setupStatus?.components.find((component) => component.id === `mmaudio-${effectiveAudioModel.slice('mmaudio:'.length).replaceAll('_', '-')}`)?.sharedReady
    : undefined
  const selectedAudioCanGenerate = Boolean(
    selectedChoice?.available
      && (selectedChoice.installed ?? selectedAudioReady)
      && audioPreparedKey === currentAudioPreparationKey,
  )
  const resultChoices = audioResult?.kind === 'music' ? setupStatus?.models.music ?? [] : setupStatus?.models.sfx ?? []
  const resultModelLabel = audioResult?.modelId ? resultChoices.find((choice) => choice.id === audioResult.modelId)?.label || audioResult.modelId : ''
  const draftFingerprint = JSON.stringify({ prompt: settings.prompt, cfgScale: settings.cfgScale, steps: settings.steps, seed: settings.seed, audioKind, audioDuration, audioOptionState, modelId: effectiveAudioModel, audioUrl: audioResult?.url || null, audioPath: audioResult?.outputPath || null, sampleRate: audioResult?.sampleRate ?? null, audioRevision })
  const [savedFingerprint, setSavedFingerprint] = useState(draftFingerprint)
  const hasUnsavedAudioChanges = savedFingerprint !== draftFingerprint
  const audioActionLabel = audioKind === 'music' ? 'Music' : 'Sound Effects'

  useEffect(() => {
    // Preparing another audio route unloads the prior backend model. Invalidate
    // the previous route's cached UI readiness as soon as selection changes.
    if (audioPreparationKeyRef.current !== currentAudioPreparationKey) {
      audioPreparationKeyRef.current = ''
      setAudioPreparedKey('')
      setAudioPreparationErrorKey('')
    }
  }, [currentAudioPreparationKey])

  useEffect(() => {
    const controller = new AbortController()
    void fetchProAudioStatus(controller.signal)
      .then(setSetupStatus)
      .catch((error: unknown) => {
        if (!controller.signal.aborted) {
          setAudioError(`Audio setup check failed: ${formatApiError(error)}`)
        }
      })
    return () => controller.abort()
  }, [])

  const handleAudioSetup = useCallback(async () => {
    setSetupBusy(true)
    setAudioError('')
    try {
      if (selectedAudioNeedsRuntimeSetup || !selectedAudioNeedsInstall) {
        const minimumResult = await runProSetupAction('POST /api/pro/audio/setup/minimum')
        if (!('minimumReady' in minimumResult)) throw new Error('Audio runtime setup returned an unexpected result.')
        setSetupStatus(minimumResult)
      }
      if (selectedAudioNeedsInstall) {
        if (!selectedAudioSetupAction) throw new Error('The selected audio model has no setup action in its route manifest.')
        const modelResult = await runProSetupAction(selectedAudioSetupAction, effectiveAudioModel)
        if (!('minimumReady' in modelResult)) throw new Error('Audio model setup returned an unexpected result.')
        setSetupStatus(modelResult)
      }
    } catch (error: unknown) {
      setAudioError(`Audio setup failed: ${formatApiError(error)}`)
    } finally {
      setSetupBusy(false)
    }
  }, [effectiveAudioModel, selectedAudioNeedsInstall, selectedAudioNeedsRuntimeSetup, selectedAudioSetupAction])

  const handleSelectAudioModel = useCallback((modelId: string) => {
    setAudioModelOverride(modelId)
    saveAudioModel(audioKind, modelId)
    setAudioError('')
  }, [audioKind])

  // research mode on/off: the server saves it and answers with the re-filtered model lists
  const handleResearchMode = useCallback(async (enabled: boolean) => {
    setResearchBusy(true)
    setAudioError('')
    try {
      const nextStatus = await setProAudioResearchMode(enabled)
      setSetupStatus(nextStatus)
      if (!enabled) {
        for (const kind of ['music', 'sfx'] as const) {
          const savedModel = storedAudioModel(kind)
          const availableModels = nextStatus.models[kind] ?? []
          if (savedModel && !availableModels.some((choice) => choice.id === savedModel)) {
            clearStoredAudioModel(kind)
          }
        }
        setAudioModelOverride('')
      }
    } catch (error: unknown) {
      setAudioError(`Could not change research mode: ${formatApiError(error)}`)
    } finally {
      setResearchBusy(false)
    }
  }, [])

  useEffect(() => {
    if (!setupStatus || !selectedChoice?.available || selectedChoice.installed !== true) return
    const preparationKey = `${audioKind}:${effectiveAudioModel}`
    if (audioPreparationKeyRef.current === preparationKey) return
    audioPreparationKeyRef.current = preparationKey
    let cancelled = false
    setAudioPrepareBusy(true)
    const prepareWithConflictRetry = async () => {
      for (let attempt = 0; ; attempt += 1) {
        try {
          return await prepareProAudioModel(audioKind, effectiveAudioModel)
        } catch (error: unknown) {
          const delay = AUDIO_PREPARE_RETRY_DELAYS_MS[attempt]
          if (cancelled || !isTransientAudioPrepareConflict(error) || delay === undefined) throw error
          await new Promise((resolve) => window.setTimeout(resolve, delay))
        }
      }
    }
    void prepareWithConflictRetry()
      .then((result) => {
        if (cancelled) return
        if (!result.ready) throw new Error(result.detail || 'The selected audio route is not ready.')
        setAudioPreparedKey(preparationKey)
        setAudioPreparationErrorKey('')
        setSetupStatus((current) => current ? {
          ...current,
          models: {
            ...current.models,
            [audioKind]: current.models[audioKind].map((item) => item.id === effectiveAudioModel
              ? { ...item, routeStatus: result.routeStatus || item.routeStatus, resident: result.resident }
              : item),
          },
        } : current)
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          if (audioPreparationKeyRef.current === preparationKey) audioPreparationKeyRef.current = ''
          setAudioPreparedKey('')
          setAudioPreparationErrorKey(preparationKey)
          setAudioError(`Audio model preparation failed: ${formatApiError(error)}`)
        }
      })
      .finally(() => {
        if (!cancelled) setAudioPrepareBusy(false)
      })
    return () => { cancelled = true }
  }, [audioKind, audioPreparationRetry, effectiveAudioModel, selectedChoice, setupStatus])

  const handleGenerateAudio = useCallback(async () => {
    if (projectBusy) return
    if (!settings.prompt.trim()) {
      setAudioError('Enter an audio prompt before generating.')
      return
    }
    if (!selectedChoice || !selectedChoice.available) {
      if (!selectedChoice) {
        setAudioError(`The saved model “${effectiveAudioModel}” is no longer available for ${audioKind === 'music' ? 'music' : 'sound effects'}. Choose an available model.`)
        return
      }
      if (selectedChoice?.available === false) {
        setAudioError(selectedChoice.unavailableReason || `${selectedChoice.label} is unavailable.`)
        return
      }
      setAudioError(`Install the minimum Audio setup before generating ${audioActionLabel.toLowerCase()}.`)
      return
    }
    if (selectedChoice.installed === false && selectedChoice.installable) {
      setAudioError(`Install ${selectedChoice.label} before generating.`)
      return
    }
    if (!selectedAudioReady && selectedChoice.installed !== true) {
      setAudioError(`Install the minimum Audio setup before generating ${audioActionLabel.toLowerCase()}.`)
      return
    }
    setAudioBusy(true)
    setAudioError('')
    try {
      const result = await generateProAudio({
        prompt: settings.prompt,
        negativePrompt: audioOptionState.negative_prompt,
        kind: audioKind,
        modelId: effectiveAudioModel,
        durationSeconds: audioDuration,
        temperature: audioOptionState.temperature,
        cfgCoef: settings.cfgScale,
        topK: audioOptionState.top_k,
        steps: settings.steps,
        seed: settings.seed,
      })
      setAudioResult(result)
      setAudioRevision((revision) => revision + 1)
    } catch (error: unknown) {
      setAudioError(`Audio generation failed: ${formatApiError(error)}`)
    } finally {
      // Audio generation may park MusicGen after rendering. Refresh backend
      // residency so the selected route label reflects the actual runtime.
      try {
        setSetupStatus(await fetchProAudioStatus())
      } catch (error: unknown) {
        const detail = `Audio status refresh failed: ${formatApiError(error)}`
        setAudioError((current) => current ? `${current} ${detail}` : detail)
      }
      setAudioBusy(false)
    }
  }, [
    audioActionLabel,
    audioDuration,
    audioKind,
    audioOptionState.temperature,
    audioOptionState.negative_prompt,
    audioOptionState.top_k,
    effectiveAudioModel,
    projectBusy,
    selectedAudioReady,
    selectedChoice,
    settings.cfgScale,
    settings.prompt,
    settings.seed,
    settings.steps,
  ])

  const setupStateFor = useCallback((id: (typeof AUDIO_MODELS)[number]['id']) => {
    if (!setupStatus) return 'missing' as const
    if (id === 'lab') {
      if (!setupStatus.labReady) return 'missing' as const
      return setupStatus.runtimeChecksPerformed ? 'ready' as const : 'detected' as const
    }
    if (id === 'mux') return setupStatus.muxReady ? 'ready' as const : 'missing' as const
    const choice = id === 'music' ? selectedMusicChoice : selectedSfxChoice
    if (!choice || choice.available === false || choice.installed !== true) return 'missing' as const
    if (id === 'sfx' && !setupStatus.runtimeChecksPerformed) {
      const selectedMmaudioPrepared = audioKind === 'sfx'
        && summarySfxModelId.startsWith('mmaudio:')
        && audioPreparedKey === currentAudioPreparationKey
      return selectedMmaudioPrepared || choice.routeStatus === 'prepared' || choice.routeStatus === 'completed'
        ? 'ready' as const
        : 'detected' as const
    }
    return 'ready' as const
  }, [audioKind, audioPreparedKey, currentAudioPreparationKey, selectedMusicChoice, selectedSfxChoice, summarySfxModelId, setupStatus])

  const refreshAudioProjects = useCallback(async () => {
    setProjectBusy(true)
    try { setAudioProjects(await fetchProAudioProjects()) }
    finally { setProjectBusy(false) }
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    void fetchProAudioProjects(controller.signal)
      .then(setAudioProjects)
      .catch((error: unknown) => {
        if (!controller.signal.aborted) setAudioError(`Audio projects could not be loaded: ${formatApiError(error)}`)
      })
    return () => controller.abort()
  }, [])

  const saveAudioProject = useCallback(async ({ name, projectId }: AudioProjectSaveRequest) => {
    setProjectBusy(true)
    try {
      const attribution = audioProjectAttribution(audioResult, effectiveAudioModel)
      const artifactKind = audioResult?.kind === 'music' || audioResult?.kind === 'sfx'
        ? audioResult.kind
        : audioKind
      const saved = await saveProAudioProject({
        name,
        project_id: projectId ?? currentProjectId,
        audio_path: audioResult?.outputPath || null,
        options: {
          prompt: audioResult?.prompt ?? settings.prompt,
          kind: artifactKind,
          model_id: attribution.modelId,
          negative_prompt: audioOptionState.negative_prompt,
          duration_seconds: audioResult?.durationSeconds ?? audioDuration,
          temperature: audioOptionState.temperature,
          cfg_coef: settings.cfgScale,
          top_k: audioOptionState.top_k,
          steps: settings.steps,
          seed: settings.seed,
        },
        sample_rate: audioResult?.sampleRate ?? 0,
        license_notice: attribution.licenseNotice,
        license: attribution.license,
      })
      setCurrentProjectId(saved.project_id)
      setSavedFingerprint(draftFingerprint)
      setAudioProjects(await fetchProAudioProjects())
    } finally { setProjectBusy(false) }
  }, [audioDuration, audioKind, audioOptionState, audioResult, currentProjectId, draftFingerprint, effectiveAudioModel, settings.cfgScale, settings.prompt, settings.seed, settings.steps])

  const loadAudioProject = useCallback(async (projectId: string) => {
    setProjectBusy(true)
    setAudioError('')
    try {
      const manifest: ProAudioProjectManifest = await loadProAudioProject(projectId)
      const options = manifest.options
      if (options.kind !== 'music' && options.kind !== 'sfx') throw new Error(`Audio project kind "${options.kind}" is not supported in Audio Studio yet.`)
      const kind: AudioKind = options.kind
      if (manifest.audio_url && !isSafeProjectAudioUrl(manifest.audio_url)) throw new Error('The project returned an unsafe audio URL.')
      onSettingsChange((current) => ({ ...current, prompt: options.prompt, cfgScale: options.cfg_coef, steps: options.steps, seed: options.seed }))
      setAudioKind(kind)
      setAudioDuration(options.duration_seconds)
      setAudioOptionState({ negative_prompt: options.negative_prompt, temperature: options.temperature, top_k: options.top_k })
      setAudioModelOverride(options.model_id)
      saveAudioModel(kind, options.model_id)
      setCurrentProjectId(manifest.project_id)
      setAudioResult(manifest.audio_url ? {
        status: 'loaded', message: `Loaded project “${manifest.name}”.`, outputPath: '', url: manifest.audio_url,
        prompt: options.prompt, kind, modelId: options.model_id, durationSeconds: manifest.track?.duration_seconds ?? options.duration_seconds,
        sampleRate: manifest.track?.sample_rate ?? 0, infotext: '', license: manifest.track?.license ?? undefined,
      } : null)
      setSavedFingerprint(JSON.stringify({ prompt: options.prompt, cfgScale: options.cfg_coef, steps: options.steps, seed: options.seed, audioKind: kind, audioDuration: options.duration_seconds, audioOptionState: { negative_prompt: options.negative_prompt, temperature: options.temperature, top_k: options.top_k }, modelId: options.model_id, audioUrl: manifest.audio_url || null, audioPath: null, sampleRate: manifest.audio_url ? manifest.track?.sample_rate ?? 0 : null, audioRevision }))
      setAudioProjects(await fetchProAudioProjects())
    } finally { setProjectBusy(false) }
  }, [audioRevision, onSettingsChange])

  const confirmAudioProjectLoad = useCallback(() => {
    if (!hasUnsavedAudioChanges) return true
    return window.confirm('This audio project has unsaved changes. Load another project and discard them?')
  }, [hasUnsavedAudioChanges])

  const selectAudioKind = (kind: AudioKind) => {
    setAudioKind(kind)
    setAudioModelOverride(storedAudioModel(kind))
  }

  return (
    <div className="studio-audio studio-full-surface" aria-label="Audio Studio layout">
      <aside className="studio-foundry-assets studio-audio-assets">
        <div className="studio-product-lockup compact">
          <span className="studio-logo-orb">A</span>
          <div>
            <strong>AIWF Studio</strong>
            <small>Audio Studio</small>
          </div>
        </div>
        <div className="studio-foundry-tabs">
          {['All', 'Audio', 'Video', 'Prompts', 'Models'].map((tab, index) => (
            <button key={tab} type="button" className={index === 1 ? 'active' : ''}>{tab}</button>
          ))}
        </div>
        <label className="studio-search-field">
          <Search size={14} aria-hidden="true" />
          <input value="" placeholder="Search audio assets..." readOnly />
        </label>
        <section className="studio-audio-card-list">
          <h3>Audio Workflows</h3>
          {AUDIO_PRESETS.map((preset) => (
            <button
              key={preset.label}
              type="button"
              className={preset.kind === audioKind ? 'active' : ''}
              disabled={preset.kind === null || audioPrepareBusy}
              title={preset.title}
              onClick={() => {
                if (preset.kind) selectAudioKind(preset.kind)
              }}
            >
              <Waves size={15} />
              <span>{preset.label}</span>
              <small>{preset.note}</small>
            </button>
          ))}
        </section>
        <section className="studio-audio-card-list">
          <h3>Audio engines</h3>
          {AUDIO_MODELS.map((model) => (
            <button
              key={model.id}
              type="button"
              className={model.id === audioKind ? 'active' : ''}
              disabled={model.id === 'lab' || model.id === 'mux' || audioPrepareBusy}
              title={model.id === 'music' || model.id === 'sfx'
                ? `Selected variant: ${model.id === 'music' ? selectedMusicEngineLabel : selectedSfxEngineLabel}. Status reflects that variant's files and dependencies. MMAudio shows detected until its isolated runtime passes preparation. Audio Studio prepares the selected model on entry or selection; MMAudio loads weights per render.`
                : model.id === 'lab'
                  ? 'Audio Lab DSP powers Voice Cleanup and Loudness Master in Gradio Audio Lab. Its setup status is shown below.'
                  : 'FFmpeg mux adds generated audio to video in Video Lab. Its setup status is shown below.'}
              onClick={() => {
                if (model.id === 'music' || model.id === 'sfx') selectAudioKind(model.id)
              }}
            >
              <Radio size={15} />
              <span>{model.id === 'music' ? 'Music generation' : model.id === 'sfx' ? 'Sound effects generation' : model.label}</span>
              <small>
                {model.id === 'music' ? selectedMusicEngineLabel : model.id === 'sfx' ? selectedSfxEngineLabel : model.id === 'lab' ? 'Voice Cleanup and Loudness Master · Gradio Audio Lab' : 'Video soundtracks · Video Lab'}
                {' · '}
                {setupStateFor(model.id) === 'ready'
                  ? 'configured'
                  : setupStateFor(model.id) === 'detected'
                    ? 'detected · full check needed'
                    : 'setup needed'}
              </small>
            </button>
          ))}
        </section>
        <section className="studio-audio-meter-card">
          <h3>System</h3>
          <span>{runtime.state}</span>
          <strong>{runtime.device || 'Local device'}</strong>
          <small>{statusMessage}</small>
        </section>
      </aside>

      <main className="studio-audio-main">
        <header className="studio-foundry-topbar">
          <div className="studio-document-title">
            <Music size={16} />
            <div>
              <strong>{settings.prompt || 'Untitled audio scene'}</strong>
              <small>{audioResult?.outputPath || 'No audio exported yet'} · local session</small>
            </div>
          </div>
          <div className="studio-foundry-top-actions">
            <AudioProjectControls
              projects={audioProjects}
              currentProjectId={currentProjectId}
              disabled={generationBusy || projectBusy || audioPrepareBusy}
              onRefresh={refreshAudioProjects}
              onSave={saveAudioProject}
              onLoad={loadAudioProject}
              onBeforeLoad={confirmAudioProjectLoad}
            />
            <button type="button" onClick={onOpenSettings}><Settings2 size={15} /></button>
            <button
              type="button"
              className="studio-export-button"
              disabled={!audioResult?.url}
              onClick={() => {
                if (audioResult?.url) window.open(audioResult.url, '_blank', 'noopener,noreferrer')
              }}
            >
              <Download size={15} /> Open audio
            </button>
          </div>
        </header>

        <section className="studio-audio-setup" aria-live="polite">
          <div className="studio-audio-setup-copy">
            <span className="studio-eyebrow">First-run setup</span>
            <strong>{setupStatus?.message || 'Checking the local audio engines...'}</strong>
            <small>
              {setupStatus?.estimatedDownload || 'Existing local models and environments are reused.'}
              {' '}{setupStatus?.licenseNotice || ''}
            </small>
            {/* research mode: off = commercial-safe models only; on = non-commercial models too, labeled */}
            <label className="studio-audio-research-mode">
              <input
                type="checkbox"
                checked={Boolean(setupStatus?.researchMode)}
                disabled={!setupStatus || researchBusy || generationBusy || setupBusy || audioPrepareBusy}
                onChange={(event) => void handleResearchMode(event.target.checked)}
              />
              <span>Allow non-commercial research models (MusicGen, MMAudio). Their output must not be used commercially.</span>
            </label>
          </div>
          <div className="studio-audio-setup-components" aria-label="Audio setup components">
            {AUDIO_MODELS.map((item) => {
              const state = setupStateFor(item.id)
              const ready = state === 'ready'
              const title = state === 'detected'
                ? item.id === 'sfx'
                  ? 'MMAudio files and environment are detected; its isolated runtime import check has not passed yet.'
                  : 'Audio Lab environment detected; its full isolated-engine self-test has not run.'
                : ready && item.id === 'lab'
                  ? 'Audio Lab isolated-engine self-test passed.'
                  : ready
                    ? 'Required files and dependencies are present; this does not mean the model is currently loaded.'
                    : 'Required files or dependencies are missing.'
              return (
                <span key={item.id} data-ready={ready} data-detected={state === 'detected'} title={title}>
                  {ready ? <CheckCircle2 size={13} /> : <AlertCircle size={13} />}
                  {item.id === 'music' ? selectedMusicEngineLabel : item.id === 'sfx' ? selectedSfxEngineLabel : item.label}{state === 'detected' ? ' · detected' : ''}
                </span>
              )
            })}
          </div>
          <button
            type="button"
            className="studio-audio-setup-button"
            disabled={setupBusy || generationBusy || audioPrepareBusy || (selectedAudioNeedsInstall && !selectedAudioSetupAction)}
            onClick={() => void handleAudioSetup()}
          >
            {setupBusy ? <Loader2 className="studio-spin" size={16} /> : <DownloadCloud size={16} />}
            {setupBusy
              ? 'Installing audio setup...'
              : selectedAudioNeedsInstall
                ? !selectedAudioNeedsRuntimeSetup
                  ? `Install ${selectedChoice?.label}`
                  : `Set up ${selectedChoice?.label} and audio runtime`
                : !selectedAudioNeedsRuntimeSetup
                ? 'Verify / repair minimum setup'
                : 'Download minimum models & dependencies'}
          </button>
          {onOpenModelSorter ? (
            <button
              type="button"
              className="studio-audio-setup-button"
              onClick={onOpenModelSorter}
            >
              <FolderOpen size={16} /> Find and organize local model files
            </button>
          ) : null}
        </section>

        <section className="studio-audio-monitor-row">
          <div className="studio-audio-preview-panel">
            <header>
              <strong>Preview Monitor</strong>
              <span>{audioResult ? `Last render: ${resultModelLabel}` : selectedAudioModelLabel || 'Audio model'}</span>
            </header>
            <label className="studio-audio-model-select">
              <span>Model</span>
              <select
                aria-label={`${audioKind === 'music' ? 'Music' : 'Sound effects'} model`}
                value={effectiveAudioModel}
                disabled={!audioModelChoices.length || generationBusy || projectBusy || setupBusy || audioPrepareBusy}
                onChange={(event) => void handleSelectAudioModel(event.target.value)}
              >
                {audioModelChoices.length
                  ? audioModelChoices.map((choice) => <option key={choice.id} value={choice.id} disabled={!choice.available}>{choice.available ? `${choice.label}${choice.installable && !choice.installed ? ' — not installed' : ''}` : `${choice.label} — unavailable`}</option>)
                  : <option value={effectiveAudioModel}>{selectedAudioModelLabel}</option>}
                {effectiveAudioModel && !audioModelChoices.some((choice) => choice.id === effectiveAudioModel)
                  ? <option value={effectiveAudioModel} disabled>{`Unavailable saved model: ${effectiveAudioModel}`}</option>
                  : null}
              </select>
              {selectedChoice ? <small>Selected: {selectedChoice.label}</small> : <small>{effectiveAudioModel || noModelLabel}</small>}
              {selectedChoice?.license ? (
                <small className={selectedChoice.license.commercial === 'yes' ? 'studio-audio-license' : 'studio-audio-license studio-audio-license-restricted'}>
                  Licence: {selectedChoice.license.license} — {selectedChoice.license.commercial === 'yes' ? 'commercial use OK' : selectedChoice.license.commercial === 'conditional' ? 'commercial use with conditions' : 'non-commercial only'}
                </small>
              ) : null}
              {selectedRouteStatus ? <small>Route status: {formatRouteLifecycleStatus(selectedRouteStatus, selectedChoice?.resident ?? null)}.</small> : null}
              {selectedChoice?.unavailableReason ? <small className="studio-audio-model-unavailable">{selectedChoice.unavailableReason}</small> : null}
              {selectedSharedAudioAssets ? <small>Matching MMAudio assets are available in a shared model root. Install copies them into Studio’s audio engine folder.</small> : null}
              {audioPreparationErrorKey === currentAudioPreparationKey && selectedChoice?.installed ? (
                <button
                  type="button"
                  className="studio-audio-setup-button"
                  disabled={generationBusy || projectBusy || setupBusy || audioPrepareBusy}
                  onClick={() => setAudioPreparationRetry((attempt) => attempt + 1)}
                >Retry model preparation</button>
              ) : null}
              {!selectedChoice && audioModelChoices.length ? <small className="studio-audio-model-unavailable">This saved model is no longer available for {audioKind === 'music' ? 'music' : 'sound effects'}.</small> : null}
            </label>
            {audioPrepareBusy ? <small role="status">Preparing selected audio model…</small> : null}
            <div className="studio-large-waveform" data-playing={generationBusy}>
              {Array.from({ length: 96 }, (_, index) => <span key={index} style={{ height: `${18 + ((index * 17) % 70)}%` }} />)}
            </div>
            {audioResult?.url ? (
              <div className="studio-audio-player">
                <audio controls src={audioResult.url} preload="metadata" />
                <small>{audioResult.message}</small>
              </div>
            ) : null}
            {audioError ? <div className="studio-audio-error"><AlertCircle size={14} /> {audioError}</div> : null}
            <div className="studio-audio-transport">
              <button type="button" disabled title="Clip editing will be connected to Audio Lab in a later pass."><Scissors size={14} /></button>
              <button
                type="button"
                className="primary"
                onClick={() => void handleGenerateAudio()}
                disabled={generationBusy || projectBusy || audioPrepareBusy || !selectedAudioCanGenerate}
              >
                {audioBusy ? <Loader2 className="studio-spin" size={16} /> : <Play size={16} />}
                {audioBusy ? 'Generating...' : `Generate ${audioActionLabel}`}
              </button>
              <button type="button" onClick={() => onSendToWorkflow?.('Audio Studio transport')}><Sparkles size={14} /> Send to workflow</button>
              <button type="button" disabled title="Use the player volume control for this build."><Volume2 size={14} /></button>
              <span>{audioResult ? `${audioResult.durationSeconds.toFixed(1)} sec · ${audioResult.sampleRate} Hz` : `${audioDuration} sec target`}</span>
            </div>
          </div>
          <div className="studio-scope-stack">
            <ScopeCard title="Spectrum" variant="spectrum" />
            <ScopeCard title="Loudness" variant="loudness" />
          </div>
        </section>

        <section className="studio-foundry-bottom-dock studio-audio-dock">
          <header>
            <div className="studio-dock-title">
              <strong>Timeline</strong>
              <small>Scenes, audio tracks, buses, and metadata lanes</small>
            </div>
            <div className="studio-dock-tabs" role="tablist" aria-label="Audio dock mode">
              <button type="button" className={dockMode === 'tracks' ? 'active' : ''} onClick={() => setDockMode('tracks')}>Tracks</button>
              <button type="button" className={dockMode === 'scenes' ? 'active' : ''} onClick={() => setDockMode('scenes')}>Scenes</button>
              <button type="button" className={dockMode === 'mixer' ? 'active' : ''} onClick={() => setDockMode('mixer')}>Mixer</button>
            </div>
          </header>
          {dockMode === 'tracks' ? (
            <div className="studio-track-board studio-audio-track-board">
              <div className="studio-track-ruler">
                {['00:00', '00:05', '00:10', '00:15', '00:20', '00:25', '00:30'].map((tick) => <span key={tick}>{tick}</span>)}
              </div>
              <AudioTrackRow label="V1" title="Video Reference" color="amber" blocks={['Scene image', 'Motion cue', 'Cut marker']} />
              <AudioTrackRow
                label="A1"
                title={audioKind === 'music' ? 'Generated Music' : 'Generated Sound Effects'}
                color="green"
                blocks={audioKind === 'music' ? ['Intro', 'Main phrase', 'Outro'] : ['Primary event', 'Room tone', 'Tail']}
              />
              <AudioTrackRow label="A2" title="Additional SFX" color="purple" blocks={['Wind', 'Helmet radio', 'Distant boom']} />
              <AudioTrackRow label="A3" title="Voice / Foley" color="blue" blocks={['Footsteps', 'Breath', 'Suit servo']} />
              <AudioTrackRow label="FX" title="Master Effects" color="cyan" blocks={['EQ', 'Compressor', 'Limiter']} />
              <AudioTrackRow label="MD" title="Metadata" color="slate" blocks={[`Prompt: ${settings.prompt.slice(0, 40) || 'Untitled'}`, `Model: ${selectedAudioModelLabel || 'not ready'} (${effectiveAudioModel})`, `Seed: ${settings.seed}`]} />
              <div className="studio-playhead" />
            </div>
          ) : dockMode === 'scenes' ? (
            <div className="studio-scene-strip studio-audio-scenes">
              {sceneRows.map((output, index) => (
                <button key={output.id} type="button">
                  <img src={output.thumbnailUrl} alt="" />
                  <strong>Scene {index + 1}</strong>
                  <small>{output.modelName || selectedModelName}</small>
                </button>
              ))}
              <button type="button" className="studio-new-variant"><Plus size={22} /> Add Scene</button>
            </div>
          ) : (
            <div className="studio-audio-mixer">
              {['A1', 'A2', 'A3', 'FX', 'MASTER'].map((channel, index) => (
                <div key={channel}>
                  <strong>{channel}</strong>
                  <div className="studio-channel-meter"><span style={{ height: `${40 + index * 10}%` }} /></div>
                  <input type="range" min="0" max="100" defaultValue={80 - index * 5} />
                  <small>S M</small>
                </div>
              ))}
            </div>
          )}
        </section>
      </main>

      <aside className="studio-foundry-inspector studio-audio-inspector">
        <header className="studio-inspector-tabs">
          <button type="button" className="active">Inspector</button>
          <button type="button">Effects</button>
        </header>
        <section>
          <span className="studio-eyebrow">Prompt</span>
          <textarea
            value={settings.prompt}
            rows={5}
            disabled={projectBusy}
            onChange={(event) => onSettingsChange((current) => ({ ...current, prompt: event.target.value }))}
          />
        </section>
        <section>
          <span className="studio-eyebrow">Effects Stack</span>
          {AUDIO_EFFECTS.map((effect) => (
            <button key={effect} type="button" className={activeEffect === effect ? 'studio-layer-row active' : 'studio-layer-row'} onClick={() => setActiveEffect(effect)}>
              <SlidersHorizontal size={14} />
              <span>{effect}</span>
              <small>{activeEffect === effect ? 'editing' : 'on'}</small>
            </button>
          ))}
        </section>
        <section>
          <span className="studio-eyebrow">Generation Settings</span>
          <label className="studio-field-mini">Type
            <select value={audioKind} disabled={projectBusy || audioPrepareBusy} onChange={(event) => selectAudioKind(event.target.value as AudioKind)}>
              <option value="music">Music</option>
              <option value="sfx">Sound effects</option>
            </select>
          </label>
          <label className="studio-field-mini">Duration
            <select value={audioDuration} disabled={projectBusy} onChange={(event) => setAudioDuration(Number(event.target.value))}>
              {[8, 15, 30].includes(audioDuration) ? null : <option value={audioDuration}>{audioDuration} sec</option>}
              <option value={8}>8 sec</option>
              <option value={15}>15 sec</option>
              <option value={30}>30 sec</option>
            </select>
          </label>
          <label className="studio-range-row">Guidance <input type="range" min="1" max="20" value={settings.cfgScale} disabled={projectBusy} onChange={(event) => onSettingsChange((current) => ({ ...current, cfgScale: Number(event.target.value) }))} /> <b>{settings.cfgScale}</b></label>
          <label className="studio-range-row">Steps <input type="range" min="1" max="100" value={settings.steps} disabled={projectBusy} onChange={(event) => onSettingsChange((current) => ({ ...current, steps: Number(event.target.value) }))} /> <b>{settings.steps}</b></label>
          <button
            type="button"
            className="studio-wide-button"
            onClick={() => void handleGenerateAudio()}
            disabled={generationBusy || projectBusy || !selectedAudioCanGenerate}
          >
            {audioBusy ? <Loader2 className="studio-spin" size={14} /> : <Sparkles size={14} />}
            {audioBusy ? 'Rendering audio...' : `Render ${audioActionLabel} Pass`}
          </button>
          <button type="button" className="studio-wide-button" onClick={() => onSendToWorkflow?.('Audio Studio render pass')}><Sparkles size={14} /> Send to workflow</button>
        </section>
      </aside>
    </div>
  )
}

function ScopeCard({ title, variant }: { title: string; variant: 'spectrum' | 'loudness' }) {
  return (
    <div className={`studio-audio-scope ${variant}`}>
      <strong>{title}</strong>
      <div>{Array.from({ length: 34 }, (_, index) => <span key={index} />)}</div>
    </div>
  )
}

function AudioTrackRow({ label, title, color, blocks }: { label: string; title: string; color: string; blocks: string[] }) {
  return (
    <div className="studio-track-row" data-color={color}>
      <div className="studio-track-label"><strong>{label}</strong><small>{title}</small></div>
      <div className="studio-track-lane">
        {blocks.map((block, index) => (
          <span key={block} style={{ width: `${22 + index * 9}%` }}>{block}</span>
        ))}
      </div>
    </div>
  )
}

function isSafeProjectAudioUrl(value: string): boolean {
  const prefix = `${API_BASE}/api/pro/outputs/`
  return (API_BASE ? value.startsWith(prefix) : value.startsWith('/api/pro/outputs/')) && !value.includes('\\') &&
    ![...value].some((character) => character.charCodeAt(0) <= 0x1f || character.charCodeAt(0) === 0x7f)
}
