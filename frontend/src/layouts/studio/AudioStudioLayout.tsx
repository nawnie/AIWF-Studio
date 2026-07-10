import { useCallback, useEffect, useMemo, useState } from 'react'
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
  Save,
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
  installMinimumProAudio,
  type ProAudioGenerateResult,
  type ProAudioStatus,
} from '../../api'
import type { LayoutProps } from './LayoutTypes'
import './studioLayouts.css'

const AUDIO_PRESETS = ['Video Soundtrack', 'SFX Burst', 'Ambient Loop', 'Voice Cleanup', 'Loudness Master']
const AUDIO_EFFECTS = ['Noise Gate', 'EQ', 'Compressor', 'Limiter', 'Stereo Width', 'Reverb Send']
const AUDIO_MODELS = [
  { id: 'music', label: 'MusicGen small' },
  { id: 'sfx', label: 'MMAudio small 16k' },
  { id: 'lab', label: 'Audio Lab DSP' },
  { id: 'mux', label: 'FFmpeg mux' },
] as const

export function AudioStudioLayout({
  settings,
  runtime,
  recentOutputs,
  selectedModelName,
  statusMessage,
  isGenerating,
  onSettingsChange,
  onSendToWorkflow,
  onOpenSettings,
}: LayoutProps) {
  const [dockMode, setDockMode] = useState<'tracks' | 'scenes' | 'mixer'>('tracks')
  const [activeEffect, setActiveEffect] = useState('EQ')
  const [setupStatus, setSetupStatus] = useState<ProAudioStatus | null>(null)
  const [setupBusy, setSetupBusy] = useState(false)
  const [audioBusy, setAudioBusy] = useState(false)
  const [audioDuration, setAudioDuration] = useState(8)
  const [audioResult, setAudioResult] = useState<ProAudioGenerateResult | null>(null)
  const [audioError, setAudioError] = useState('')
  const sceneRows = useMemo(() => recentOutputs.slice(0, 6), [recentOutputs])
  const generationBusy = isGenerating || audioBusy

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

  const handleMinimumSetup = useCallback(async () => {
    setSetupBusy(true)
    setAudioError('')
    try {
      const next = await installMinimumProAudio()
      setSetupStatus(next)
    } catch (error: unknown) {
      setAudioError(`Audio setup failed: ${formatApiError(error)}`)
    } finally {
      setSetupBusy(false)
    }
  }, [])

  const handleGenerateAudio = useCallback(async () => {
    if (!settings.prompt.trim()) {
      setAudioError('Enter an audio prompt before generating.')
      return
    }
    if (!setupStatus?.musicReady) {
      setAudioError('Install the minimum Audio setup before generating music.')
      return
    }
    setAudioBusy(true)
    setAudioError('')
    try {
      const result = await generateProAudio({
        prompt: settings.prompt,
        kind: 'music',
        modelId: setupStatus.defaults.music,
        durationSeconds: audioDuration,
        temperature: 1,
        cfgCoef: settings.cfgScale,
        topK: 250,
        steps: settings.steps,
        seed: settings.seed,
      })
      setAudioResult(result)
    } catch (error: unknown) {
      setAudioError(`Audio generation failed: ${formatApiError(error)}`)
    } finally {
      setAudioBusy(false)
    }
  }, [audioDuration, settings.cfgScale, settings.prompt, settings.seed, settings.steps, setupStatus])

  const setupReadyFor = useCallback((id: (typeof AUDIO_MODELS)[number]['id']) => {
    if (!setupStatus) return false
    if (id === 'music') return setupStatus.musicReady
    if (id === 'sfx') return setupStatus.sfxReady
    if (id === 'lab') return setupStatus.labReady
    return setupStatus.muxReady
  }, [setupStatus])

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
          {AUDIO_PRESETS.map((preset, index) => (
            <button key={preset} type="button" className={index === 0 ? 'active' : ''}>
              <Waves size={15} />
              <span>{preset}</span>
              <small>{index === 0 ? 'Video-aware' : 'Ready'}</small>
            </button>
          ))}
        </section>
        <section className="studio-audio-card-list">
          <h3>Audio stack</h3>
          {AUDIO_MODELS.map((model) => (
            <button key={model.id} type="button" disabled title="Model selection is coming in the next Audio Studio pass.">
              <Radio size={15} />
              <span>{model.label}</span>
              <small>{setupReadyFor(model.id) ? 'ready' : 'setup needed'}</small>
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
            <button type="button" disabled title="Audio project save is not implemented yet."><Save size={15} /> Save Project</button>
            <button type="button" disabled title="Audio project loading is not implemented yet."><FolderOpen size={15} /> Load</button>
            <button type="button" onClick={onOpenSettings}><Settings2 size={15} /></button>
            <button
              type="button"
              className="studio-export-button"
              disabled={!audioResult?.url}
              onClick={() => {
                if (audioResult?.url) window.open(audioResult.url, '_blank', 'noopener,noreferrer')
              }}
            >
              <Download size={15} /> Open WAV
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
          </div>
          <div className="studio-audio-setup-components" aria-label="Audio setup components">
            {AUDIO_MODELS.map((item) => {
              const ready = setupReadyFor(item.id)
              return (
                <span key={item.id} data-ready={ready}>
                  {ready ? <CheckCircle2 size={13} /> : <AlertCircle size={13} />}
                  {item.label}
                </span>
              )
            })}
          </div>
          <button
            type="button"
            className="studio-audio-setup-button"
            disabled={setupBusy || generationBusy}
            onClick={() => void handleMinimumSetup()}
          >
            {setupBusy ? <Loader2 className="studio-spin" size={16} /> : <DownloadCloud size={16} />}
            {setupBusy
              ? 'Installing audio setup...'
              : setupStatus?.minimumReady
                ? 'Verify / repair minimum setup'
                : 'Download minimum models & dependencies'}
          </button>
        </section>

        <section className="studio-audio-monitor-row">
          <div className="studio-audio-preview-panel">
            <header>
              <strong>Preview Monitor</strong>
              <span>{audioResult?.modelId || setupStatus?.defaults.music || 'MusicGen small'}</span>
            </header>
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
                disabled={generationBusy || !setupStatus?.musicReady}
              >
                {audioBusy ? <Loader2 className="studio-spin" size={16} /> : <Play size={16} />}
                {audioBusy ? 'Generating...' : 'Generate Music'}
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
              <AudioTrackRow label="A1" title="Music Bed" color="green" blocks={['Ambient score', 'Build section', 'Outro swell']} />
              <AudioTrackRow label="A2" title="SFX" color="purple" blocks={['Wind', 'Helmet radio', 'Distant boom']} />
              <AudioTrackRow label="A3" title="Voice / Foley" color="blue" blocks={['Footsteps', 'Breath', 'Suit servo']} />
              <AudioTrackRow label="FX" title="Master Effects" color="cyan" blocks={['EQ', 'Compressor', 'Limiter']} />
              <AudioTrackRow label="MD" title="Metadata" color="slate" blocks={[`Prompt: ${settings.prompt.slice(0, 40) || 'Untitled'}`, 'Model: MMAudio', `Seed: ${settings.seed}`]} />
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
          <label className="studio-field-mini">Duration
            <select value={audioDuration} onChange={(event) => setAudioDuration(Number(event.target.value))}>
              <option value={8}>8 sec</option>
              <option value={15}>15 sec</option>
              <option value={30}>30 sec</option>
            </select>
          </label>
          <label className="studio-range-row">Guidance <input type="range" min="1" max="20" value={settings.cfgScale} onChange={(event) => onSettingsChange((current) => ({ ...current, cfgScale: Number(event.target.value) }))} /> <b>{settings.cfgScale}</b></label>
          <label className="studio-range-row">Steps <input type="range" min="1" max="100" value={settings.steps} onChange={(event) => onSettingsChange((current) => ({ ...current, steps: Number(event.target.value) }))} /> <b>{settings.steps}</b></label>
          <button
            type="button"
            className="studio-wide-button"
            onClick={() => void handleGenerateAudio()}
            disabled={generationBusy || !setupStatus?.musicReady}
          >
            {audioBusy ? <Loader2 className="studio-spin" size={14} /> : <Sparkles size={14} />}
            {audioBusy ? 'Rendering audio...' : 'Render Music Pass'}
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
