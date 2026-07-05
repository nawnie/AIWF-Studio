import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Boxes,
  Brush,
  Database,
  FileImage,
  FolderOpen,
  GitBranch,
  Grid2x2,
  Hand,
  HardDrive,
  Image,
  Layers3,
  LayoutDashboard,
  Maximize2,
  Monitor,
  PanelLeft,
  Play,
  RefreshCcw,
  Settings,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Video,
  X,
} from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import {
  fetchProBootstrap,
  fetchProRuntime,
  formatApiError,
  generateProOutput,
  getFallbackBootstrap,
  getFallbackRuntime,
} from './api'
import type {
  AspectRatioOption,
  CreationMode,
  GenerationSettings,
  ProBootstrap,
  ProMode,
  ProRuntimeStatus,
  RecentOutput,
  ResourceMetric,
  ProModelOption,
} from './types'
import { PipelineAtlasPaidLayout } from './layouts/paid/PipelineAtlasPaidLayout'
import { MediaFoundryImagePaidLayout } from './layouts/paid/MediaFoundryImagePaidLayout'
import { AudioStudioPaidLayout } from './layouts/paid/AudioStudioPaidLayout'
import { AgenticChatPaidLayout } from './layouts/paid/AgenticChatPaidLayout'
import { ExtensionsHubPaidLayout } from './layouts/paid/ExtensionsHubPaidLayout'
import { PluginWorkspacePaidLayout } from './layouts/paid/PluginWorkspacePaidLayout'
import { SettingsArsenalPaidLayout } from './layouts/paid/SettingsArsenalPaidLayout'
import { ProjectCenterPaidLayout } from './layouts/paid/ProjectCenterPaidLayout'
import { CommandPalettePaid } from './layouts/paid/CommandPalettePaid'
import { ModelFamilyMatrixPaidLayout } from './layouts/paid/ModelFamilyMatrixPaidLayout'
import type { PaidUserTab, PaidWorkflowCodeBlock } from './layouts/paid/PaidLayoutTypes'
import { autosavePaidProject, buildAutosavePayload, fetchPaidUserTabs, loadPaidUserTabsFromStorage, savePaidUserTabsToStorage } from './layouts/paid/paidApiClient'
import { createWorkflowBlocksFromSettings, loadWorkflowBlocksFromStorage, renumberWorkflowBlocks, saveWorkflowBlocksToStorage } from './layouts/paid/workflowBlocks'
import './styles.css'

interface IconItem<T extends string> {
  id: T
  label: string
  icon: LucideIcon
}

const MODE_TABS: IconItem<ProMode>[] = [
  { id: 'image', label: 'Image', icon: Image },
  { id: 'video', label: 'Video', icon: Video },
  { id: 'inpaint', label: 'Inpaint', icon: Brush },
  { id: 'models', label: 'Models', icon: Boxes },
  { id: 'data', label: 'Data', icon: Database },
]

const BASE_RAIL_ITEMS: IconItem<string>[] = [
  { id: 'explore', label: 'Explore', icon: LayoutDashboard },
  { id: 'create', label: 'Create', icon: Sparkles },
  { id: 'project', label: 'Project', icon: FolderOpen },
  { id: 'pipeline', label: 'Pipeline', icon: GitBranch },
  { id: 'foundry', label: 'Foundry', icon: Layers3 },
  { id: 'audio', label: 'Audio', icon: SlidersHorizontal },
  { id: 'agent', label: 'Agent', icon: Sparkles },
  { id: 'extensions', label: 'Extensions', icon: Boxes },
  { id: 'canvas', label: 'Canvas', icon: Grid2x2 },
  { id: 'batch', label: 'Batch', icon: Layers3 },
  { id: 'workflows', label: 'Workflows', icon: Boxes },
  { id: 'models', label: 'Models', icon: Boxes },
  { id: 'families', label: 'Families', icon: ShieldCheck },
  { id: 'data', label: 'Data', icon: Database },
  { id: 'logs', label: 'Logs', icon: FileImage },
  { id: 'settings', label: 'Settings', icon: Settings },
]

const SYSTEM_NOTES = [
  ['Run locally, stay private', 'Everything runs on your machine. Your data never leaves.'],
  ['Optional service tabs', 'Enable integrations only when you need them.'],
  ['Hidden tool tabs', 'Advanced tabs stay tucked away by default.'],
  ['Models and datasets', 'Manage local models and datasets with full control.'],
  ['Safety by default', 'Local processing. You own your outputs and prompts.'],
] as const

function App() {
  const fallbackBootstrap = useMemo(() => getFallbackBootstrap(), [])
  const fallbackRuntime = useMemo(() => getFallbackRuntime(), [])
  const [bootstrap, setBootstrap] = useState<ProBootstrap>(fallbackBootstrap)
  const [runtime, setRuntime] = useState<ProRuntimeStatus>(fallbackRuntime)
  const [settings, setSettings] = useState<GenerationSettings>(fallbackBootstrap.defaults)
  const [activeMode, setActiveMode] = useState<ProMode>('image')
  const [activeRail, setActiveRail] = useState('explore')
  const [recentOutputs, setRecentOutputs] = useState<RecentOutput[]>(
    fallbackBootstrap.recentOutputs,
  )
  const [preview, setPreview] = useState<RecentOutput | null>(
    fallbackBootstrap.recentOutputs[0] ?? null,
  )
  const [showOnboarding, setShowOnboarding] = useState(!fallbackBootstrap.onboardingSeen)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [statusMessage, setStatusMessage] = useState('Ready.')
  const [isGenerating, setIsGenerating] = useState(false)
  const [paidTabs, setPaidTabs] = useState<PaidUserTab[]>(() => loadPaidUserTabsFromStorage())
  const [workflowBlocks, setWorkflowBlocks] = useState<PaidWorkflowCodeBlock[]>(() => loadWorkflowBlocksFromStorage())
  const [commandPaletteOpen, setCommandPaletteOpen] = useState(false)

  const railItems = useMemo<IconItem<string>[]>(() => [
    ...BASE_RAIL_ITEMS,
    ...paidTabs
      .filter((tab) => !tab.hidden)
      .map((tab) => ({ id: tab.id, label: tab.label, icon: Boxes })),
  ], [paidTabs])

  const updatePaidTabs = useCallback((tabs: PaidUserTab[]) => {
    setPaidTabs(tabs)
    savePaidUserTabsToStorage(tabs)
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    fetchProBootstrap(controller.signal)
      .then((nextBootstrap) => {
        setBootstrap(nextBootstrap)
        setRecentOutputs(nextBootstrap.recentOutputs)
        setPreview((currentPreview) => currentPreview ?? nextBootstrap.recentOutputs[0] ?? null)
        setShowOnboarding(!nextBootstrap.onboardingSeen)
        setSettings((current) => mergeBootstrapDefaults(current, nextBootstrap))
        setStatusMessage('Connected to /api/pro/bootstrap.')
      })
      .catch((error: unknown) => {
        if (isAbortError(error)) {
          return
        }
        setStatusMessage(`Using local shell defaults. ${formatApiError(error)}`)
      })

    return () => controller.abort()
  }, [])

  useEffect(() => {
    fetchPaidUserTabs().then(setPaidTabs).catch(() => undefined)
  }, [])

  useEffect(() => {
    const autosaveId = window.setInterval(() => {
      void autosavePaidProject(buildAutosavePayload(settings, activeRail, recentOutputs))
    }, 8000)
    return () => window.clearInterval(autosaveId)
  }, [activeRail, recentOutputs, settings])

  useEffect(() => {
    saveWorkflowBlocksToStorage(workflowBlocks)
  }, [workflowBlocks])

  useEffect(() => {
    const controller = new AbortController()
    const refreshRuntime = () => {
      fetchProRuntime(controller.signal)
        .then(setRuntime)
        .catch((error: unknown) => {
          if (!isAbortError(error)) {
            setStatusMessage(`Runtime status unavailable. ${formatApiError(error)}`)
          }
        })
    }

    refreshRuntime()
    const intervalId = window.setInterval(refreshRuntime, 10000)
    return () => {
      controller.abort()
      window.clearInterval(intervalId)
    }
  }, [])

  const selectedModel = useMemo(
    () =>
      bootstrap.models.find((model) => model.id === settings.modelId) ??
      bootstrap.models[0],
    [bootstrap.models, settings.modelId],
  )

  const activeRatio = useMemo(
    () =>
      bootstrap.aspectRatios.find((ratio) => ratio.id === settings.aspectRatioId) ??
      bootstrap.aspectRatios[0],
    [bootstrap.aspectRatios, settings.aspectRatioId],
  )

  const handleModeSelect = useCallback((mode: ProMode) => {
    setActiveMode(mode)
    if (isCreationMode(mode)) {
      setSettings((current) => ({ ...current, mode }))
      setActiveRail('create')
    } else {
      setActiveRail(mode)
    }
  }, [])

  const handleRailSelect = useCallback((id: string) => {
    setActiveRail(id)
    if (id === 'models') {
      setActiveMode('models')
    } else if (id === 'data') {
      setActiveMode('data')
    } else if (id === 'project' || id === 'pipeline' || id === 'foundry' || id === 'audio' || id === 'agent' || id === 'extensions' || id === 'families' || paidTabs.some((tab) => tab.id === id)) {
      setActiveMode('image')
    } else if (id === 'create') {
      setActiveMode('image')
      setSettings((current) => ({ ...current, mode: 'image' }))
    }
  }, [paidTabs])

  const handleRatioSelect = useCallback((ratio: AspectRatioOption) => {
    setSettings((current) => ({
      ...current,
      aspectRatioId: ratio.id,
      width: ratio.width,
      height: ratio.height,
    }))
  }, [])

  const handleGenerate = useCallback(async () => {
    const selectedStatus = (selectedModel?.status ?? '').toLowerCase()
    if (['broken-runtime', 'blocked-cleanly', 'unsupported-no-route'].includes(selectedStatus)) {
      setStatusMessage(`Generation blocked by model family gate: ${selectedModel?.reason || selectedModel?.status}`)
      return
    }
    if (!settings.prompt.trim()) {
      setStatusMessage('Enter a prompt before generating.')
      return
    }

    const controller = new AbortController()
    setIsGenerating(true)
    setStatusMessage('Submitting to /api/pro/generate...')
    try {
      const result = await generateProOutput(settings, controller.signal)
      const nextOutput = result.output
      if (nextOutput) {
        setPreview(nextOutput)
        setRecentOutputs((current) => dedupeOutputs([nextOutput, ...result.recentOutputs, ...current]))
      }
      setStatusMessage(result.message || `Generation ${result.status}.`)
    } catch (error: unknown) {
      setStatusMessage(`Generate failed. ${formatApiError(error)}`)
    } finally {
      setIsGenerating(false)
    }
  }, [selectedModel, settings])

  const handleSendToWorkflow = useCallback((source = 'Create panel') => {
    setWorkflowBlocks((current) => {
      const nextBlocks = createWorkflowBlocksFromSettings(
        {
          settings,
          bootstrap,
          runtime,
          selectedModel,
          selectedModelName: selectedModel?.name ?? settings.modelId,
          source,
        },
        current.length,
      )
      return renumberWorkflowBlocks([...current, ...nextBlocks])
    })
    setActiveRail('pipeline')
    setStatusMessage('Current settings were captured as a movable workflow code block.')
  }, [bootstrap, runtime, selectedModel, settings])

  return (
    <div className="aiwf-pro-shell" data-mode={activeMode} data-rail={activeRail}>
      <aside className="pro-rail" aria-label="Primary navigation">
        <button
          type="button"
          className="pro-logo-button"
          aria-label="AIWF Studio home"
          onClick={() => handleRailSelect('explore')}
        >
          <span className="pro-logo-mark">A</span>
        </button>
        <nav className="pro-rail-nav">
          {railItems.map((item) => (
            <RailButton
              key={item.id}
              item={item}
              active={activeRail === item.id}
              onSelect={handleRailSelect}
            />
          ))}
        </nav>
        <div className="pro-rail-footer">
          <span className="pro-local-dot" aria-hidden="true" />
          <span>Local First</span>
          <small>All systems go</small>
          <small>{bootstrap.version}</small>
        </div>
      </aside>

      <main className="pro-main">
        <TopBar
          bootstrap={bootstrap}
          runtime={runtime}
          onOpenSettings={() => handleRailSelect('settings')}
        />
        {['project', 'pipeline', 'foundry', 'audio', 'agent', 'extensions', 'families', 'settings'].includes(activeRail) || paidTabs.some((tab) => tab.id === activeRail) ? null : (
          <ModeTabs activeMode={activeMode} onSelect={handleModeSelect} />
        )}

        <section className="pro-workspace" aria-label="AIWF Pro workspace">
          {activeRail === 'project' ? (
            <ProjectCenterPaidLayout
              settings={settings}
              bootstrap={bootstrap}
              runtime={runtime}
              recentOutputs={recentOutputs}
              preview={preview}
              selectedModel={selectedModel}
              selectedModelName={selectedModel?.name ?? settings.modelId}
              statusMessage={statusMessage}
              isGenerating={isGenerating}
              onSettingsChange={setSettings}
              onGenerate={handleGenerate}
              onSendToWorkflow={handleSendToWorkflow}
              workflowBlocks={workflowBlocks}
              onWorkflowBlocksChange={setWorkflowBlocks}
              onPreviewSelect={setPreview}
              onOpenModels={() => handleRailSelect('models')}
              onOpenSettings={() => handleRailSelect('settings')}
            />
          ) : activeRail === 'pipeline' ? (
            <PipelineAtlasPaidLayout
              settings={settings}
              bootstrap={bootstrap}
              runtime={runtime}
              recentOutputs={recentOutputs}
              preview={preview}
              selectedModel={selectedModel}
              selectedModelName={selectedModel?.name ?? settings.modelId}
              statusMessage={statusMessage}
              isGenerating={isGenerating}
              onSettingsChange={setSettings}
              onGenerate={handleGenerate}
              onSendToWorkflow={handleSendToWorkflow}
              workflowBlocks={workflowBlocks}
              onWorkflowBlocksChange={setWorkflowBlocks}
              onPreviewSelect={setPreview}
              onOpenModels={() => handleRailSelect('models')}
              onOpenSettings={() => handleRailSelect('settings')}
            />
          ) : activeRail === 'foundry' ? (
            <MediaFoundryImagePaidLayout
              settings={settings}
              bootstrap={bootstrap}
              runtime={runtime}
              recentOutputs={recentOutputs}
              preview={preview}
              selectedModel={selectedModel}
              selectedModelName={selectedModel?.name ?? settings.modelId}
              statusMessage={statusMessage}
              isGenerating={isGenerating}
              onSettingsChange={setSettings}
              onGenerate={handleGenerate}
              onSendToWorkflow={handleSendToWorkflow}
              onPreviewSelect={setPreview}
              onOpenModels={() => handleRailSelect('models')}
              onOpenSettings={() => handleRailSelect('settings')}
            />
          ) : activeRail === 'audio' ? (
            <AudioStudioPaidLayout
              settings={settings}
              bootstrap={bootstrap}
              runtime={runtime}
              recentOutputs={recentOutputs}
              preview={preview}
              selectedModel={selectedModel}
              selectedModelName={selectedModel?.name ?? settings.modelId}
              statusMessage={statusMessage}
              isGenerating={isGenerating}
              onSettingsChange={setSettings}
              onGenerate={handleGenerate}
              onSendToWorkflow={handleSendToWorkflow}
              onPreviewSelect={setPreview}
              onOpenModels={() => handleRailSelect('models')}
              onOpenSettings={() => handleRailSelect('settings')}
            />
          ) : activeRail === 'agent' ? (
            <AgenticChatPaidLayout
              settings={settings}
              bootstrap={bootstrap}
              runtime={runtime}
              recentOutputs={recentOutputs}
              preview={preview}
              selectedModel={selectedModel}
              selectedModelName={selectedModel?.name ?? settings.modelId}
              statusMessage={statusMessage}
              isGenerating={isGenerating}
              onSettingsChange={setSettings}
              onGenerate={handleGenerate}
              onSendToWorkflow={handleSendToWorkflow}
              onPreviewSelect={setPreview}
              onOpenModels={() => handleRailSelect('models')}
              onOpenSettings={() => handleRailSelect('settings')}
            />
          ) : activeRail === 'extensions' ? (
            <ExtensionsHubPaidLayout
              paidTabs={paidTabs}
              onPaidTabsChange={updatePaidTabs}
              onOpenTab={handleRailSelect}
            />
          ) : activeRail === 'families' ? (
            <ModelFamilyMatrixPaidLayout
              settings={settings}
              bootstrap={bootstrap}
              runtime={runtime}
              recentOutputs={recentOutputs}
              preview={preview}
              selectedModel={selectedModel}
              selectedModelName={selectedModel?.name ?? settings.modelId}
              statusMessage={statusMessage}
              isGenerating={isGenerating}
              onSettingsChange={setSettings}
              onGenerate={handleGenerate}
              onSendToWorkflow={handleSendToWorkflow}
              onPreviewSelect={setPreview}
              onOpenModels={() => handleRailSelect('models')}
              onOpenSettings={() => handleRailSelect('settings')}
            />
          ) : activeRail === 'settings' ? (
            <SettingsArsenalPaidLayout
              settings={settings}
              bootstrap={bootstrap}
              runtime={runtime}
              recentOutputs={recentOutputs}
              preview={preview}
              selectedModel={selectedModel}
              selectedModelName={selectedModel?.name ?? settings.modelId}
              statusMessage={statusMessage}
              isGenerating={isGenerating}
              onSettingsChange={setSettings}
              onGenerate={handleGenerate}
              onSendToWorkflow={handleSendToWorkflow}
              onPreviewSelect={setPreview}
              onOpenModels={() => handleRailSelect('models')}
              onOpenSettings={() => handleRailSelect('settings')}
              paidTabs={paidTabs}
              onPaidTabsChange={updatePaidTabs}
            />
          ) : paidTabs.some((tab) => tab.id === activeRail) ? (
            <PluginWorkspacePaidLayout
              tab={paidTabs.find((tab) => tab.id === activeRail) ?? paidTabs[0]}
              onOpenExtensions={() => handleRailSelect('extensions')}
            />
          ) : (
            <>
              <PromptPanel
                settings={settings}
                bootstrap={bootstrap}
                selectedModel={selectedModel}
                selectedModelName={selectedModel?.name ?? settings.modelId}
                activeRatio={activeRatio}
                showAdvanced={showAdvanced}
                isGenerating={isGenerating}
                onSettingsChange={setSettings}
                onRatioSelect={handleRatioSelect}
                onGenerate={handleGenerate}
                onSendToWorkflow={handleSendToWorkflow}
                onToggleAdvanced={() => setShowAdvanced((value) => !value)}
              />
              <CanvasPreview
                activeMode={activeMode}
                preview={preview}
                statusMessage={statusMessage}
                width={settings.width}
                height={settings.height}
              />
              <RuntimePanel runtime={runtime} selectedModelName={selectedModel?.name ?? settings.modelId} />
            </>
          )}
        </section>

        {['project', 'pipeline', 'foundry', 'audio', 'agent', 'extensions', 'families', 'settings'].includes(activeRail) || paidTabs.some((tab) => tab.id === activeRail) ? null : (
          <RecentRail
            outputs={recentOutputs}
            activeOutputId={preview?.id}
            onSelect={setPreview}
            onOpenFolder={() => setStatusMessage('Recent outputs are served by the local backend.')}
          />
        )}
      </main>

      <CommandPalettePaid
        open={commandPaletteOpen}
        onOpenChange={setCommandPaletteOpen}
        onNavigate={handleRailSelect}
        onGenerate={handleGenerate}
      />

      {showOnboarding ? (
        <OnboardingModal
          onClose={() => setShowOnboarding(false)}
          onShowAdvanced={() => {
            setShowAdvanced(true)
            setShowOnboarding(false)
          }}
          onOpenData={() => {
            setActiveMode('data')
            setActiveRail('data')
            setShowOnboarding(false)
          }}
        />
      ) : null}
    </div>
  )
}

function TopBar({
  bootstrap,
  runtime,
  onOpenSettings,
}: {
  bootstrap: ProBootstrap
  runtime: ProRuntimeStatus
  onOpenSettings: () => void
}) {
  const vram = runtime.resources.find((metric) => metric.label.toLowerCase() === 'vram')

  return (
    <header className="pro-topbar">
      <div className="pro-titlebar">
        <h1>{bootstrap.workspaceName}</h1>
        <span>{bootstrap.subtitle}</span>
      </div>
      <div className="pro-topbar-status">
        <div className="pro-engine-status" data-state={runtime.state.toLowerCase()}>
          <span className="pro-status-dot" aria-hidden="true" />
          <span>Local Engine</span>
          <strong>{runtime.state}</strong>
        </div>
        {vram ? <MiniMetric metric={vram} /> : null}
        <button
          type="button"
          className="pro-icon-button"
          aria-label="Open settings"
          onClick={onOpenSettings}
        >
          <Settings size={18} aria-hidden="true" />
        </button>
        <div className="pro-window-controls" aria-hidden="true">
          <span />
          <span />
          <span />
        </div>
      </div>
    </header>
  )
}

function ModeTabs({
  activeMode,
  onSelect,
}: {
  activeMode: ProMode
  onSelect: (mode: ProMode) => void
}) {
  return (
    <nav className="pro-tabs" aria-label="Mode tabs">
      {MODE_TABS.map((item) => {
        const Icon = item.icon
        const active = activeMode === item.id
        return (
          <button
            key={item.id}
            type="button"
            className={active ? 'pro-tab pro-tab-active' : 'pro-tab'}
            aria-pressed={active}
            onClick={() => onSelect(item.id)}
          >
            <Icon size={16} aria-hidden="true" />
            <span>{item.label}</span>
          </button>
        )
      })}
    </nav>
  )
}

function PromptPanel({
  settings,
  bootstrap,
  selectedModel,
  selectedModelName,
  activeRatio,
  showAdvanced,
  isGenerating,
  onSettingsChange,
  onRatioSelect,
  onGenerate,
  onSendToWorkflow,
  onToggleAdvanced,
}: {
  settings: GenerationSettings
  bootstrap: ProBootstrap
  selectedModel: ProModelOption | undefined
  selectedModelName: string
  activeRatio: AspectRatioOption | undefined
  showAdvanced: boolean
  isGenerating: boolean
  onSettingsChange: (value: GenerationSettings | ((current: GenerationSettings) => GenerationSettings)) => void
  onRatioSelect: (ratio: AspectRatioOption) => void
  onGenerate: () => void
  onSendToWorkflow: (source?: string) => void
  onToggleAdvanced: () => void
}) {
  const selectedStatus = (selectedModel?.status ?? '').toLowerCase()
  const modelBlocked = ['broken-runtime', 'blocked-cleanly', 'unsupported-no-route'].includes(selectedStatus)
  const modelWarning = ['metadata-only', 'needs-smoke', 'experimental', 'candidate'].includes(selectedStatus)

  return (
    <aside className="pro-prompt-panel" aria-label="Prompt and generation settings">
      <PanelHeader title="Prompt" actionLabel="Prompt tools" icon={PanelLeft} />
      <label className="pro-field pro-prompt-field">
        <span>Prompt</span>
        <textarea
          value={settings.prompt}
          maxLength={1500}
          rows={5}
          onChange={(event) =>
            onSettingsChange((current) => ({ ...current, prompt: event.target.value }))
          }
        />
        <small>{settings.prompt.length} / 1500</small>
      </label>

      <label className="pro-field">
        <span>Negative prompt</span>
        <textarea
          value={settings.negativePrompt}
          maxLength={1500}
          rows={3}
          onChange={(event) =>
            onSettingsChange((current) => ({
              ...current,
              negativePrompt: event.target.value,
            }))
          }
        />
        <small>{settings.negativePrompt.length} / 1500</small>
      </label>

      <label className="pro-field">
        <span>Model</span>
        <div className="pro-select-row">
          <select
            value={settings.modelId}
            onChange={(event) =>
              onSettingsChange((current) => ({ ...current, modelId: event.target.value }))
            }
          >
            {bootstrap.models.map((model) => {
              const status = (model.status ?? '').toLowerCase()
              const blocked = ['broken-runtime', 'blocked-cleanly', 'unsupported-no-route'].includes(status)
              const label = model.status ? `${model.name} · ${model.status}` : model.name
              return (
                <option key={model.id} value={model.id} disabled={blocked}>
                  {label}
                </option>
              )
            })}
          </select>
          <button type="button" className="pro-icon-button" aria-label="Refresh models">
            <RefreshCcw size={16} aria-hidden="true" />
          </button>
        </div>
        {modelBlocked || modelWarning ? (
          <small className={modelBlocked ? 'pro-model-gate pro-model-gate-block' : 'pro-model-gate pro-model-gate-warn'}>
            {modelBlocked ? 'Blocked from normal selection' : 'Needs smoke receipt'}: {selectedModel?.reason || selectedModel?.suggestedAction || selectedModel?.status}
          </small>
        ) : null}
      </label>

      <fieldset className="pro-aspect-group">
        <legend>Aspect ratio</legend>
        <div className="pro-aspect-chips">
          {bootstrap.aspectRatios.map((ratio) => {
            const active = activeRatio?.id === ratio.id
            return (
              <button
                key={ratio.id}
                type="button"
                className={active ? 'pro-chip pro-chip-active' : 'pro-chip'}
                aria-pressed={active}
                onClick={() => onRatioSelect(ratio)}
              >
                {ratio.label}
              </button>
            )
          })}
        </div>
      </fieldset>

      <div className="pro-settings-block">
        <div className="pro-section-label">Image settings</div>
        <RangeField
          label="Steps"
          min={1}
          max={80}
          step={1}
          value={settings.steps}
          onChange={(value) => onSettingsChange((current) => ({ ...current, steps: value }))}
        />
        <RangeField
          label="CFG scale"
          min={0}
          max={20}
          step={0.5}
          value={settings.cfgScale}
          onChange={(value) => onSettingsChange((current) => ({ ...current, cfgScale: value }))}
        />
        <label className="pro-field pro-compact-field">
          <span>Sampler</span>
          <select
            value={settings.sampler}
            onChange={(event) =>
              onSettingsChange((current) => ({ ...current, sampler: event.target.value }))
            }
          >
            {bootstrap.samplers.map((sampler) => (
              <option key={sampler} value={sampler}>
                {sampler}
              </option>
            ))}
          </select>
        </label>
        <label className="pro-field pro-compact-field">
          <span>Seed</span>
          <input
            type="number"
            value={settings.seed}
            onChange={(event) =>
              onSettingsChange((current) => ({
                ...current,
                seed: Number(event.target.value),
              }))
            }
          />
        </label>
      </div>

      {showAdvanced ? (
        <div className="pro-advanced-panel">
          <label className="pro-field pro-compact-field">
            <span>Width</span>
            <input
              type="number"
              min={256}
              step={8}
              value={settings.width}
              onChange={(event) =>
                onSettingsChange((current) => ({
                  ...current,
                  width: Number(event.target.value),
                }))
              }
            />
          </label>
          <label className="pro-field pro-compact-field">
            <span>Height</span>
            <input
              type="number"
              min={256}
              step={8}
              value={settings.height}
              onChange={(event) =>
                onSettingsChange((current) => ({
                  ...current,
                  height: Number(event.target.value),
                }))
              }
            />
          </label>
          <label className="pro-field pro-compact-field">
            <span>Batch size</span>
            <input
              type="number"
              min={1}
              max={8}
              value={settings.batchSize}
              onChange={(event) =>
                onSettingsChange((current) => ({
                  ...current,
                  batchSize: Number(event.target.value),
                }))
              }
            />
          </label>
        </div>
      ) : null}

      {settings.mode === 'video' ? (
        <div className="pro-settings-block pro-wan-routing-block">
          <div className="pro-section-label">Video / Wan wiring</div>
          <label className="pro-field pro-compact-field">
            <span>Runtime route</span>
            <select value={settings.wanRuntimeMode} onChange={(event) => onSettingsChange((current) => ({ ...current, wanRuntimeMode: event.target.value }))}>
              <option value="fast_5b">Fast 5B TI2V</option>
              <option value="native_high_low">High/low GGUF</option>
              <option value="native_high_low_fp8_experimental">High/low FP8 experimental</option>
            </select>
          </label>
          <div className="pro-wan-grid">
            <label className="pro-field pro-compact-field"><span>High model</span><input value={settings.highNoiseModelId} placeholder="high-noise model id/path" onChange={(event) => onSettingsChange((current) => ({ ...current, highNoiseModelId: event.target.value }))} /></label>
            <label className="pro-field pro-compact-field"><span>Low model</span><input value={settings.lowNoiseModelId} placeholder="low-noise model id/path" onChange={(event) => onSettingsChange((current) => ({ ...current, lowNoiseModelId: event.target.value }))} /></label>
            <label className="pro-field pro-compact-field"><span>VAE</span><input value={settings.vaeId} placeholder="Wan VAE id/path" onChange={(event) => onSettingsChange((current) => ({ ...current, vaeId: event.target.value }))} /></label>
            <label className="pro-field pro-compact-field"><span>Text encoder</span><input value={settings.textEncoderPath} placeholder="UMT5/encoder path" onChange={(event) => onSettingsChange((current) => ({ ...current, textEncoderPath: event.target.value }))} /></label>
            <label className="pro-field pro-compact-field"><span>High LoRA</span><input value={settings.highNoiseLoraId} placeholder="optional" onChange={(event) => onSettingsChange((current) => ({ ...current, highNoiseLoraId: event.target.value }))} /></label>
            <label className="pro-field pro-compact-field"><span>Low LoRA</span><input value={settings.lowNoiseLoraId} placeholder="optional" onChange={(event) => onSettingsChange((current) => ({ ...current, lowNoiseLoraId: event.target.value }))} /></label>
          </div>
          <div className="pro-wan-grid pro-wan-grid-compact">
            <label className="pro-field pro-compact-field"><span>High steps</span><input type="number" min={1} max={60} value={settings.highNoiseSteps} onChange={(event) => onSettingsChange((current) => ({ ...current, highNoiseSteps: Number(event.target.value) }))} /></label>
            <label className="pro-field pro-compact-field"><span>Low steps</span><input type="number" min={1} max={60} value={settings.lowNoiseSteps} onChange={(event) => onSettingsChange((current) => ({ ...current, lowNoiseSteps: Number(event.target.value) }))} /></label>
            <label className="pro-field pro-compact-field"><span>Boundary</span><input type="number" min={0} max={1} step={0.01} value={settings.boundaryRatio} onChange={(event) => onSettingsChange((current) => ({ ...current, boundaryRatio: Number(event.target.value) }))} /></label>
            <label className="pro-field pro-compact-field"><span>Offload</span><select value={settings.wanOffload} onChange={(event) => onSettingsChange((current) => ({ ...current, wanOffload: event.target.value }))}><option value="balanced">balanced</option><option value="streamed">streamed</option><option value="group">group</option><option value="sequential">sequential</option><option value="resident">resident</option><option value="none">none</option></select></label>
          </div>
          <small>Captured into workflow code blocks as model pack, LoRA stack, and offload plan. This is wiring QA, not adapter certification.</small>
        </div>
      ) : null}

      <div className="pro-panel-actions">
        <button
          type="button"
          className="pro-generate-button"
          onClick={onGenerate}
          disabled={isGenerating || modelBlocked}
          title={modelBlocked ? 'This model is blocked by the family/readiness gate.' : undefined}
        >
          <Sparkles size={18} aria-hidden="true" />
          <span>{isGenerating ? 'Generating...' : modelBlocked ? 'Blocked' : 'Generate'}</span>
        </button>
        <button
          type="button"
          className="pro-workflow-send-button"
          onClick={() => onSendToWorkflow('Create panel')}
        >
          <GitBranch size={17} aria-hidden="true" />
          <span>Send to workflow</span>
        </button>
        <button
          type="button"
          className="pro-icon-button pro-sliders-button"
          aria-label="Toggle advanced settings"
          aria-pressed={showAdvanced}
          onClick={onToggleAdvanced}
        >
          <SlidersHorizontal size={18} aria-hidden="true" />
        </button>
      </div>

      <div className="pro-selected-model" title={selectedModelName}>
        <HardDrive size={14} aria-hidden="true" />
        <span>{selectedModelName}</span>
      </div>
    </aside>
  )
}

function CanvasPreview({
  activeMode,
  preview,
  statusMessage,
  width,
  height,
}: {
  activeMode: ProMode
  preview: RecentOutput | null
  statusMessage: string
  width: number
  height: number
}) {
  const aspectRatio = `${Math.max(1, width)} / ${Math.max(1, height)}`
  return (
    <section className="pro-canvas" aria-label="Canvas and output preview">
      <div className="pro-canvas-header">
        <span>Canvas / output preview</span>
        <div className="pro-canvas-tools" aria-label="Canvas tools">
          <button type="button" className="pro-icon-button" aria-label="Pan preview">
            <Hand size={16} aria-hidden="true" />
          </button>
          <button type="button" className="pro-icon-button" aria-label="Fit preview">
            <Maximize2 size={16} aria-hidden="true" />
          </button>
          <button type="button" className="pro-zoom-button">100%</button>
        </div>
      </div>

      <div className="pro-preview-stage">
        <div className="pro-output-frame" style={{ aspectRatio }}>
          {preview ? (
            <img src={preview.url} alt={preview.prompt} />
          ) : (
            <div className="pro-empty-preview">
              <Monitor size={42} aria-hidden="true" />
              <span>{activeMode === 'video' ? 'Video preview' : 'Image preview'}</span>
            </div>
          )}
        </div>
      </div>

      <div className="pro-canvas-footer">
        <span>{width}x{height}</span>
        <span>{preview?.modelName ?? 'Local model'}</span>
        <span>{statusMessage}</span>
      </div>
    </section>
  )
}

function RuntimePanel({
  runtime,
  selectedModelName,
}: {
  runtime: ProRuntimeStatus
  selectedModelName: string
}) {
  return (
    <aside className="pro-status-panel" aria-label="Runtime status">
      <div className="pro-status-heading">
        <span>Runtime status</span>
        <strong>
          <span className="pro-status-dot" aria-hidden="true" />
          {runtime.state}
        </strong>
      </div>

      <dl className="pro-runtime-list">
        <MetricRow label="Backend" value={runtime.backend} />
        <MetricRow label="Device" value={runtime.device} />
        <MetricRow label="Precision" value={runtime.precision} />
        <MetricRow label="Attention" value={runtime.attention} />
        <MetricRow label="Max resolution" value={runtime.maxResolution} />
      </dl>

      <div className="pro-resource-section">
        <span className="pro-section-label">Resource usage</span>
        {runtime.resources.map((metric) => (
          <ResourceBar key={metric.label} metric={metric} />
        ))}
      </div>

      <div className="pro-loaded-model">
        <span className="pro-section-label">Loaded model</span>
        <div className="pro-loaded-model-title">
          <strong>{runtime.loadedModel.name || selectedModelName}</strong>
          <span>{runtime.loadedModel.loaded ? 'Loaded' : 'Not loaded'}</span>
        </div>
        <dl className="pro-runtime-list">
          <MetricRow label="Type" value={runtime.loadedModel.type} />
          <MetricRow label="Base model" value={runtime.loadedModel.baseModel} />
          <MetricRow label="Size on disk" value={runtime.loadedModel.sizeOnDisk} />
          <MetricRow label="Precision" value={runtime.loadedModel.precision} />
          <MetricRow label="VAE" value={runtime.loadedModel.vae} />
          <MetricRow label="Text encoder" value={runtime.loadedModel.textEncoder} />
          <MetricRow label="UNet" value={runtime.loadedModel.unet} />
        </dl>
        <button type="button" className="pro-unload-button">Unload model</button>
      </div>

      <div className="pro-queue-row">
        <span>Queue</span>
        <strong>{runtime.queueCount} tasks</strong>
      </div>
    </aside>
  )
}

function RecentRail({
  outputs,
  activeOutputId,
  onSelect,
  onOpenFolder,
}: {
  outputs: RecentOutput[]
  activeOutputId: string | undefined
  onSelect: (output: RecentOutput) => void
  onOpenFolder: () => void
}) {
  return (
    <section className="pro-recent-rail" aria-label="Recent outputs">
      <div className="pro-recent-header">
        <span>Recent outputs</span>
        <div className="pro-recent-actions">
          <button type="button" className="pro-folder-button" onClick={onOpenFolder}>
            <FolderOpen size={14} aria-hidden="true" />
            <span>Open folder</span>
          </button>
          <button type="button" className="pro-icon-button" aria-label="Grid view">
            <Grid2x2 size={15} aria-hidden="true" />
          </button>
        </div>
      </div>
      <div className="pro-recent-strip">
        {outputs.map((output) => {
          const active = activeOutputId === output.id
          return (
            <button
              key={output.id}
              type="button"
              className={active ? 'pro-recent-thumb pro-recent-thumb-active' : 'pro-recent-thumb'}
              aria-pressed={active}
              onClick={() => onSelect(output)}
            >
              <img src={output.thumbnailUrl} alt={output.prompt} />
              <span>{output.width}x{output.height}</span>
              <small>{output.createdAt}</small>
            </button>
          )
        })}
      </div>
    </section>
  )
}

function OnboardingModal({
  onClose,
  onShowAdvanced,
  onOpenData,
}: {
  onClose: () => void
  onShowAdvanced: () => void
  onOpenData: () => void
}) {
  return (
    <div className="pro-modal" role="dialog" aria-modal="true" aria-labelledby="pro-modal-title">
      <div className="pro-modal-card">
        <button
          type="button"
          className="pro-modal-close"
          aria-label="Dismiss onboarding"
          onClick={onClose}
        >
          <X size={18} aria-hidden="true" />
        </button>
        <div className="pro-modal-brand" aria-hidden="true">A</div>
        <div className="pro-modal-copy">
          <h2 id="pro-modal-title">Welcome to AIWF Studio</h2>
          <p>AIWF Studio is a local-first workstation for open-source image and video generation.</p>
        </div>
        <div className="pro-modal-points">
          {SYSTEM_NOTES.map(([title, copy]) => (
            <div key={title} className="pro-modal-point">
              <ShieldCheck size={20} aria-hidden="true" />
              <div>
                <strong>{title}</strong>
                <span>{copy}</span>
              </div>
            </div>
          ))}
        </div>
        <div className="pro-modal-account">
          <div className="pro-avatar" aria-hidden="true" />
          <div>
            <strong>nawnie</strong>
            <span>GitHub account</span>
          </div>
        </div>
        <div className="pro-modal-actions">
          <button type="button" className="pro-primary-action" onClick={onClose}>
            <span>Get started</span>
            <Play size={16} aria-hidden="true" />
          </button>
          <button type="button" className="pro-secondary-action" onClick={onShowAdvanced}>
            Show advanced tools
          </button>
          <button type="button" className="pro-secondary-action" onClick={onOpenData}>
            Open dataset reference
          </button>
        </div>
        <small>You can change preferences anytime in Settings.</small>
      </div>
    </div>
  )
}

function PanelHeader({
  title,
  actionLabel,
  icon: Icon,
}: {
  title: string
  actionLabel: string
  icon: LucideIcon
}) {
  return (
    <div className="pro-panel-header">
      <span>{title}</span>
      <button type="button" className="pro-icon-button" aria-label={actionLabel}>
        <Icon size={15} aria-hidden="true" />
      </button>
    </div>
  )
}

function RailButton({
  item,
  active,
  onSelect,
}: {
  item: IconItem<string>
  active: boolean
  onSelect: (id: string) => void
}) {
  const Icon = item.icon
  return (
    <button
      type="button"
      className={active ? 'pro-rail-button pro-rail-button-active' : 'pro-rail-button'}
      aria-pressed={active}
      onClick={() => onSelect(item.id)}
    >
      <Icon size={18} aria-hidden="true" />
      <span>{item.label}</span>
    </button>
  )
}

function RangeField({
  label,
  min,
  max,
  step,
  value,
  onChange,
}: {
  label: string
  min: number
  max: number
  step: number
  value: number
  onChange: (value: number) => void
}) {
  return (
    <label className="pro-range-field">
      <span>{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
      <output>{value}</output>
    </label>
  )
}

function MiniMetric({ metric }: { metric: ResourceMetric }) {
  return (
    <div className="pro-mini-metric">
      <span>{metric.label}</span>
      <strong>{metric.value}</strong>
      <div
        className={`pro-meter pro-meter-${metric.tone}`}
        role="meter"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={metric.percent}
      >
        <span style={{ width: `${metric.percent}%` }} />
      </div>
    </div>
  )
}

function ResourceBar({ metric }: { metric: ResourceMetric }) {
  return (
    <div className="pro-resource-meter">
      <div className="pro-resource-labels">
        <span>{metric.label}</span>
        <strong>{metric.value}</strong>
        <small>{metric.percent}%</small>
      </div>
      <div
        className={`pro-meter pro-meter-${metric.tone}`}
        role="meter"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={metric.percent}
      >
        <span style={{ width: `${metric.percent}%` }} />
      </div>
    </div>
  )
}

function MetricRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="pro-metric-row">
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  )
}

function mergeBootstrapDefaults(
  current: GenerationSettings,
  nextBootstrap: ProBootstrap,
): GenerationSettings {
  const modelStillExists = nextBootstrap.models.some((model) => model.id === current.modelId)
  const samplerStillExists = nextBootstrap.samplers.includes(current.sampler)
  const ratioStillExists = nextBootstrap.aspectRatios.some((ratio) => ratio.id === current.aspectRatioId)
  const ratio = ratioStillExists
    ? nextBootstrap.aspectRatios.find((item) => item.id === current.aspectRatioId)
    : nextBootstrap.aspectRatios.find((item) => item.id === nextBootstrap.defaults.aspectRatioId)

  return {
    ...current,
    modelId: modelStillExists ? current.modelId : nextBootstrap.defaults.modelId,
    sampler: samplerStillExists ? current.sampler : nextBootstrap.defaults.sampler,
    aspectRatioId: ratio?.id ?? nextBootstrap.defaults.aspectRatioId,
    width: ratio?.width ?? current.width,
    height: ratio?.height ?? current.height,
  }
}

function dedupeOutputs(outputs: RecentOutput[]): RecentOutput[] {
  const seen = new Set<string>()
  const unique: RecentOutput[] = []
  for (const output of outputs) {
    if (seen.has(output.id)) {
      continue
    }
    seen.add(output.id)
    unique.push(output)
  }
  return unique.slice(0, 16)
}

function isCreationMode(mode: ProMode): mode is CreationMode {
  return mode === 'image' || mode === 'video' || mode === 'inpaint'
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError'
}

export default App
