import { useEffect, useMemo, useState } from 'react'
import {
  Archive,
  BadgeCheck,
  Boxes,
  BrainCircuit,
  ClipboardCheck,
  Download,
  FileJson,
  GitBranch,
  Library,
  ListChecks,
  PackagePlus,
  Play,
  Save,
  Search,
  ShieldCheck,
  Sparkles,
  Tags,
  Upload,
  Wand2,
} from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import type { PaidLayoutProps } from './PaidLayoutTypes'
import {
  addPaidQueueJob,
  controlPaidWorker,
  fetchPaidJobLog,
  fetchPaidWorkerStatus,
  addPaidVersion,
  analyzePaidQa,
  createPaidExportPlan,
  fetchPaidAgentPermissions,
  fetchPaidAssets,
  fetchPaidExportPresets,
  fetchPaidNodeRegistry,
  fetchPaidPlugins,
  fetchPaidProject,
  fetchPaidQueue,
  fetchPaidReceipts,
  fetchPaidWorkflowTemplates,
  installPaidPluginManifest,
  savePaidAgentPermissions,
  savePaidProject,
  updatePaidQueueJob,
  validatePaidWorkflow,
} from './paidApiClient'
import type {
  PaidAgentPermissions,
  PaidWorkerStatus,
  PaidAsset,
  PaidCharacter,
  PaidExportPreset,
  PaidNodeDefinition,
  PaidPlugin,
  PaidProject,
  PaidQaCheck,
  PaidQueueJob,
  PaidReceipt,
  PaidVersion,
  PaidWorkflowTemplate,
} from './paidApiClient'
import './paidLayouts.css'

const EMPTY_PERMISSIONS: PaidAgentPermissions = {
  observe: true,
  suggest: true,
  draft: true,
  executeWithApproval: false,
  trustedLocal: false,
  allowedTools: ['project-reader', 'workflow-json', 'prompt-refiner'],
}

const DEFAULT_MANIFEST = JSON.stringify({
  id: 'example.empty.workspace',
  name: 'Example Empty Workspace',
  version: '0.1.0',
  ui: { leftRail: true, workspace: 'plugin' },
  permissions: ['read_project'],
  nodes: [],
  apiRoutes: [],
}, null, 2)

export function ProjectCenterPaidLayout({
  settings,
  runtime,
  recentOutputs,
  preview,
  selectedModelName,
  statusMessage,
  isGenerating,
  onSettingsChange,
  onGenerate,
  onSendToWorkflow,
}: PaidLayoutProps) {
  const [project, setProject] = useState<PaidProject | null>(null)
  const [assets, setAssets] = useState<PaidAsset[]>([])
  const [receipts, setReceipts] = useState<PaidReceipt[]>([])
  const [queue, setQueue] = useState<PaidQueueJob[]>([])
  const [templates, setTemplates] = useState<PaidWorkflowTemplate[]>([])
  const [nodes, setNodes] = useState<PaidNodeDefinition[]>([])
  const [presets, setPresets] = useState<PaidExportPreset[]>([])
  const [plugins, setPlugins] = useState<PaidPlugin[]>([])
  const [permissions, setPermissions] = useState<PaidAgentPermissions>(EMPTY_PERMISSIONS)
  const [qaChecks, setQaChecks] = useState<PaidQaCheck[]>([])
  const [qaScore, setQaScore] = useState(0)
  const [activePanel, setActivePanel] = useState<'project' | 'assets' | 'queue' | 'prompt' | 'characters' | 'qa' | 'plugins' | 'export'>('project')
  const [message, setMessage] = useState('Media center state loaded locally while the backend answers.')
  const [manifestText, setManifestText] = useState(DEFAULT_MANIFEST)
  const [compareMode, setCompareMode] = useState(false)
  const [newCharacterName, setNewCharacterName] = useState('')

  const activeAsset = useMemo(() => {
    if (preview?.path) {
      return assets.find((asset) => asset.path === preview.path) ?? null
    }
    return assets[0] ?? null
  }, [assets, preview?.path])

  const activeProject = project ?? {
    id: 'default',
    name: 'AIWF Media Project',
    scenes: [],
    tracks: [],
    versions: [],
    promptStudio: {},
    characters: [],
  }

  const workflowPreview = useMemo(() => ({
    id: 'main',
    stages: templates[0]?.stages ?? ['prompt', 'model', 'upscale', 'receipt', 'output'],
  }), [templates])

  const refresh = () => {
    void Promise.all([
      fetchPaidProject('default').then(setProject),
      fetchPaidAssets().then(setAssets),
      fetchPaidReceipts().then(setReceipts),
      fetchPaidQueue().then(setQueue),
      fetchPaidWorkflowTemplates().then(setTemplates),
      fetchPaidNodeRegistry().then((payload) => setNodes(payload.nodes)),
      fetchPaidExportPresets().then(setPresets),
      fetchPaidPlugins().then(setPlugins),
      fetchPaidAgentPermissions().then(setPermissions),
    ]).then(() => setMessage('Project, assets, queue, nodes, plugins, and permissions refreshed.'))
  }

  useEffect(refresh, [])

  const saveProject = async (nextProject: PaidProject, nextMessage = 'Project saved.') => {
    setProject(nextProject)
    const saved = await savePaidProject(nextProject)
    setProject(saved)
    setMessage(nextMessage)
  }

  const updatePromptStudio = (field: string, value: string) => {
    const nextProject = {
      ...activeProject,
      promptStudio: {
        ...(activeProject.promptStudio ?? {}),
        [field]: value,
      },
    }
    setProject(nextProject)
  }

  const assemblePrompt = async () => {
    const promptStudio = activeProject.promptStudio ?? {}
    const assembled = [promptStudio.subject, promptStudio.style, promptStudio.camera, promptStudio.lighting, promptStudio.world]
      .filter(Boolean)
      .join(', ')
    const nextProject = { ...activeProject, promptStudio: { ...promptStudio, assembledPrompt: assembled } }
    await saveProject(nextProject, 'Prompt Studio assembled and saved.')
    onSettingsChange((current) => ({ ...current, prompt: assembled || current.prompt, negativePrompt: promptStudio.negative || current.negativePrompt }))
  }

  const addCharacter = async () => {
    const name = newCharacterName.trim()
    if (!name) {
      setMessage('Name the character or object first.')
      return
    }
    const character: PaidCharacter = {
      id: `character-${Date.now()}`,
      name,
      description: `Reference card for ${name}`,
      reference: preview?.path || activeAsset?.path || '',
      notes: settings.prompt.slice(0, 180),
    }
    await saveProject({ ...activeProject, characters: [character, ...(activeProject.characters ?? [])] }, `${name} added to the consistency board.`)
    setNewCharacterName('')
  }

  const addVersion = async () => {
    const versions = await addPaidVersion(activeProject.id || 'default', {
      label: preview?.prompt?.slice(0, 50) || 'Current preview',
      assetPath: preview?.path || activeAsset?.path || '',
      summary: `Model ${selectedModelName}, ${settings.width}x${settings.height}, ${settings.steps} steps`,
    })
    setProject({ ...activeProject, versions })
    setMessage('Version snapshot added to the project tree.')
  }

  const addQueue = async () => {
    const validation = await validatePaidWorkflow(workflowPreview)
    if (!validation.valid) {
      setMessage(`Queue blocked: ${validation.errors.join('; ')}`)
      return
    }
    const job = await addPaidQueueJob(`Run ${templates[0]?.label ?? 'workflow'}`, { workflow: workflowPreview, settings })
    if (job) {
      setQueue((current) => [job, ...current])
      setMessage('Workflow added to render queue.')
    }
  }

  const runQa = async () => {
    const result = await analyzePaidQa(settings.prompt, workflowPreview, preview?.path || activeAsset?.path || '')
    setQaScore(result.score)
    setQaChecks(result.checks)
    setMessage(`AI QA pass complete. Score ${result.score}.`)
  }

  const createExport = async (presetId: string) => {
    const plan = await createPaidExportPlan(presetId, activeAsset ? [activeAsset.id] : [], {
      width: settings.width,
      height: settings.height,
      modelId: settings.modelId,
      format: presetId,
    })
    setMessage(`Export plan created: ${String(plan.id || presetId)}`)
  }

  const installManifest = async () => {
    try {
      const manifest = JSON.parse(manifestText) as Record<string, unknown>
      const plugin = await installPaidPluginManifest(manifest)
      if (plugin) {
        setPlugins((current) => [plugin, ...current.filter((item) => item.id !== plugin.id)])
        setMessage(`Installed plugin manifest: ${plugin.name}`)
      }
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Manifest JSON could not be parsed.')
    }
  }

  const savePermissions = async (next: PaidAgentPermissions) => {
    setPermissions(next)
    const saved = await savePaidAgentPermissions(next)
    setPermissions(saved)
    setMessage('Agent permissions saved.')
  }

  return (
    <div className="paid-media-center paid-full-surface">
      <aside className="paid-media-nav">
        <div className="paid-product-lockup">
          <span className="paid-logo-orb">M</span>
          <div><strong>Media Center</strong><small>Project spine · PAID v4</small></div>
        </div>
        {[
          ['project', 'Project File', FileJson],
          ['assets', 'Assets + Versions', Library],
          ['queue', 'Render Queue', ListChecks],
          ['prompt', 'Prompt Studio', Wand2],
          ['characters', 'Characters', Tags],
          ['qa', 'AI QA Critic', ClipboardCheck],
          ['plugins', 'Plugins + Permissions', ShieldCheck],
          ['export', 'Export Center', Archive],
        ].map(([id, label, Icon]) => {
          const Glyph = Icon as typeof FileJson
          return <button key={String(id)} type="button" className={activePanel === id ? 'active' : ''} onClick={() => setActivePanel(id as typeof activePanel)}><Glyph size={15} />{String(label)}</button>
        })}
        <div className="paid-system-card">
          <span>Runtime</span>
          <strong>{runtime.state}</strong>
          <small>{statusMessage}</small>
        </div>
      </aside>

      <main className="paid-media-main">
        <header className="paid-media-hero">
          <div>
            <span className="paid-eyebrow">TRUE AI MEDIA CENTER</span>
            <h2>{activeProject.name}</h2>
            <p>{message}</p>
          </div>
          <div className="paid-toolbar-actions">
            <button type="button" onClick={refresh}><Search size={14} /> Refresh</button>
            <button type="button" onClick={() => saveProject(activeProject)}><Save size={14} /> Save Project</button>
            <button type="button" onClick={addQueue}><ListChecks size={14} /> Queue Workflow</button>
            <button type="button" className="paid-run-button" onClick={onGenerate} disabled={isGenerating}><Play size={14} /> {isGenerating ? 'Running' : 'Run Now'}</button>
            <button type="button" onClick={() => onSendToWorkflow?.('Project Center')}>Send to workflow</button>
          </div>
        </header>

        <section className="paid-media-grid">
          <section className="paid-media-stage">
            <header>
              <strong>{compareMode ? 'Compare Mode' : 'Active Canvas'}</strong>
              <div><button type="button" onClick={() => setCompareMode((value) => !value)}>A/B Compare</button><button type="button" onClick={addVersion}>Snapshot Version</button></div>
            </header>
            <div className={compareMode ? 'paid-stage-compare active' : 'paid-stage-compare'}>
              <div className="paid-stage-frame"><img src={preview?.thumbnailUrl || preview?.url || activeAsset?.url || '/paid-astronaut-canvas.png'} alt="Current media" /></div>
              {compareMode ? <div className="paid-stage-frame ghost"><img src={recentOutputs[1]?.thumbnailUrl || recentOutputs[1]?.url || '/paid-astronaut-canvas.png'} alt="Comparison media" /></div> : null}
            </div>
            <div className="paid-layer-stack">
              <span>Non-destructive stack</span>
              {['Base image', 'Mask layer', 'Inpaint patch', 'Color grade', 'Upscale/VSR', 'Export receipt'].map((layer, index) => <button type="button" key={layer}><b>{index + 1}</b>{layer}</button>)}
            </div>
          </section>

          <section className="paid-media-panel">
            {activePanel === 'project' ? <ProjectPanel project={activeProject} templates={templates} nodes={nodes} onSave={saveProject} /> : null}
            {activePanel === 'assets' ? <AssetPanel assets={assets} receipts={receipts} versions={activeProject.versions ?? []} /> : null}
            {activePanel === 'queue' ? <QueuePanel jobs={queue} onAction={async (jobId, action) => setQueue(await updatePaidQueueJob(jobId, action))} onRefresh={() => fetchPaidQueue().then(setQueue)} /> : null}
            {activePanel === 'prompt' ? <PromptStudioPanel project={activeProject} updatePromptStudio={updatePromptStudio} assemblePrompt={assemblePrompt} /> : null}
            {activePanel === 'characters' ? <CharactersPanel project={activeProject} newCharacterName={newCharacterName} setNewCharacterName={setNewCharacterName} addCharacter={addCharacter} /> : null}
            {activePanel === 'qa' ? <QaPanel score={qaScore} checks={qaChecks} runQa={runQa} workflowPreview={workflowPreview} /> : null}
            {activePanel === 'plugins' ? <PluginPanel plugins={plugins} manifestText={manifestText} setManifestText={setManifestText} installManifest={installManifest} permissions={permissions} savePermissions={savePermissions} /> : null}
            {activePanel === 'export' ? <ExportPanel presets={presets} createExport={createExport} activeAsset={activeAsset} /> : null}
          </section>
        </section>
      </main>
    </div>
  )
}

function ProjectPanel({ project, templates, nodes, onSave }: { project: PaidProject; templates: PaidWorkflowTemplate[]; nodes: PaidNodeDefinition[]; onSave: (project: PaidProject, message?: string) => void }) {
  return <div className="paid-panel-stack">
    <PanelTitle icon={FileJson} title="AIWF Project File" subtitle="Scenes, tracks, versions, workflows, receipts, and UI state." />
    <div className="paid-stat-grid"><span>Scenes <strong>{project.scenes?.length ?? 0}</strong></span><span>Tracks <strong>{project.tracks?.length ?? 0}</strong></span><span>Nodes <strong>{nodes.length}</strong></span><span>Templates <strong>{templates.length}</strong></span></div>
    <div className="paid-scene-list">{(project.scenes ?? []).map((scene) => <button type="button" key={scene.id}><strong>{scene.title}</strong><small>{scene.status || 'draft'} · {scene.notes}</small></button>)}</div>
    <button type="button" className="paid-wide-button" onClick={() => onSave({ ...project, scenes: [...(project.scenes ?? []), { id: `scene-${Date.now()}`, title: 'New scene', status: 'draft', notes: 'Timeline marker ready.' }] }, 'Scene marker added.')}>+ Add Scene Marker</button>
    <div className="paid-template-strip">{templates.map((template) => <span key={template.id}><GitBranch size={13} />{template.label}</span>)}</div>
  </div>
}

function AssetPanel({ assets, receipts, versions }: { assets: PaidAsset[]; receipts: PaidReceipt[]; versions: PaidVersion[] }) {
  return <div className="paid-panel-stack">
    <PanelTitle icon={Library} title="Asset Library + Version Tree" subtitle="Recent outputs become traceable project assets." />
    <div className="paid-asset-mini-grid">{assets.slice(0, 12).map((asset) => <button type="button" key={`${asset.path}-${asset.id}`}><span>{asset.kind}</span><strong>{asset.name}</strong><small>{asset.tags?.join(', ')}</small></button>)}</div>
    <h3>Version tree</h3>
    <div className="paid-version-tree">{versions.length ? versions.map((version, index) => <div key={version.id}><b>{index === 0 ? '●' : '├'}</b><span>{version.label || version.id}</span><small>{version.summary || version.createdAt}</small></div>) : <p>No versions yet. Snapshot the canvas to start the tree.</p>}</div>
    <h3>Receipts</h3>
    <div className="paid-receipt-list">{receipts.slice(0, 5).map((receipt) => <span key={receipt.path}><FileJson size={13} />{receipt.name}</span>)}</div>
  </div>
}

function QueuePanel({ jobs, onAction, onRefresh }: { jobs: PaidQueueJob[]; onAction: (jobId: string, action: string) => void; onRefresh: () => void }) {
  const [worker, setWorker] = useState<PaidWorkerStatus | null>(null)
  const [logJobId, setLogJobId] = useState<string | null>(null)
  const [logText, setLogText] = useState('')

  const refreshWorker = () => { fetchPaidWorkerStatus().then(setWorker) }
  useEffect(() => { refreshWorker() }, [])
  useEffect(() => {
    const hasActive = worker?.running || jobs.some((job) => job.status === 'running' || job.status === 'queued')
    if (!hasActive) return
    const timer = window.setInterval(() => { onRefresh(); refreshWorker() }, 1500)
    return () => window.clearInterval(timer)
  }, [worker?.running, jobs, onRefresh])

  const control = async (action: 'start' | 'stop' | 'run-next') => {
    setWorker(await controlPaidWorker(action))
    onRefresh()
  }
  const openLog = async (jobId: string) => {
    setLogJobId(jobId === logJobId ? null : jobId)
    if (jobId !== logJobId) setLogText(await fetchPaidJobLog(jobId))
  }

  return <div className="paid-panel-stack">
    <PanelTitle icon={ListChecks} title="Render Queue" subtitle="Background worker executes queued jobs through registered executors." />
    <div className="paid-stat-grid">
      <span>Worker <strong>{worker?.running ? 'running' : 'stopped'}</strong></span>
      <span>Done <strong>{worker?.completed ?? 0}</strong></span>
      <span>Failed <strong>{worker?.failed ?? 0}</strong></span>
      <span>Kinds <strong>{worker?.registeredKinds?.join(', ') || '—'}</strong></span>
    </div>
    <div>
      {worker?.running
        ? <button type="button" onClick={() => control('stop')}>Stop Worker</button>
        : <><button type="button" onClick={() => control('start')}>Start Worker</button><button type="button" onClick={() => control('run-next')}>Run Next Job</button></>}
    </div>
    {jobs.length ? jobs.map((job) => <div className="paid-queue-row" key={job.id}>
      <div>
        <strong>{job.label}</strong>
        <small>{job.status}{(job as PaidQueueJob & { stage?: string }).stage ? ` · ${(job as PaidQueueJob & { stage?: string }).stage}` : ''} · {job.createdAt}</small>
        {(job as PaidQueueJob & { error?: string }).error ? <small className="paid-queue-error">{(job as PaidQueueJob & { error?: string }).error}</small> : null}
      </div>
      <span>{job.progress ?? 0}%</span>
      <button type="button" onClick={() => onAction(job.id, job.status === 'paused' ? 'resume' : 'pause')}>{job.status === 'paused' ? 'Resume' : 'Pause'}</button>
      {job.status === 'failed' ? <button type="button" onClick={() => onAction(job.id, 'retry')}>Retry</button> : null}
      <button type="button" onClick={() => onAction(job.id, 'cancel')}>Cancel</button>
      <button type="button" onClick={() => openLog(job.id)}>Log</button>
      {logJobId === job.id ? <pre className="paid-job-log">{logText || 'No log yet.'}</pre> : null}
    </div>) : <p>No queued jobs yet. Add a workflow from the top bar.</p>}
  </div>
}

function PromptStudioPanel({ project, updatePromptStudio, assemblePrompt }: { project: PaidProject; updatePromptStudio: (field: string, value: string) => void; assemblePrompt: () => void }) {
  const studio = project.promptStudio ?? {}
  return <div className="paid-panel-stack"><PanelTitle icon={Wand2} title="Prompt Studio" subtitle="Structured prompts with reusable scene memory." />
    {['subject', 'style', 'camera', 'lighting', 'world', 'negative'].map((field) => <label className="paid-field-mini" key={field}>{field}<textarea value={String((studio as Record<string, unknown>)[field] ?? '')} onChange={(event) => updatePromptStudio(field, event.target.value)} rows={2} /></label>)}
    <button type="button" className="paid-wide-button" onClick={assemblePrompt}><Sparkles size={14} /> Assemble and send to prompt</button>
    <p>{studio.assembledPrompt || 'Assembled prompt appears here.'}</p>
  </div>
}

function CharactersPanel({ project, newCharacterName, setNewCharacterName, addCharacter }: { project: PaidProject; newCharacterName: string; setNewCharacterName: (value: string) => void; addCharacter: () => void }) {
  return <div className="paid-panel-stack"><PanelTitle icon={Tags} title="Character / Object Consistency" subtitle="Reference cards for people, products, places, vehicles, and creatures." />
    <label className="paid-field-mini">New card name<input value={newCharacterName} onChange={(event) => setNewCharacterName(event.target.value)} placeholder="Astronaut, product, room, vehicle..." /></label>
    <button type="button" className="paid-wide-button" onClick={addCharacter}><PackagePlus size={14} /> Add Reference Card</button>
    <div className="paid-character-grid">{(project.characters ?? []).map((card) => <button type="button" key={card.id}><strong>{card.name}</strong><small>{card.description}</small><p>{card.notes}</p></button>)}</div>
  </div>
}

function QaPanel({ score, checks, runQa, workflowPreview }: { score: number; checks: PaidQaCheck[]; runQa: () => void; workflowPreview: Record<string, unknown> }) {
  return <div className="paid-panel-stack"><PanelTitle icon={ClipboardCheck} title="AI QA Critic" subtitle="Checks prompt, workflow classes, receipt readiness, and selected media state." />
    <button type="button" className="paid-wide-button" onClick={runQa}><BrainCircuit size={14} /> Run QA Pass</button>
    <div className="paid-qa-score"><strong>{score || '—'}</strong><span>QA score</span></div>
    {checks.map((check) => <div className={`paid-qa-row ${check.status}`} key={check.id}><BadgeCheck size={15} /><div><strong>{check.label}</strong><small>{check.status}</small><p>{check.suggestion}</p></div></div>)}
    <pre>{JSON.stringify(workflowPreview, null, 2)}</pre>
  </div>
}

function PluginPanel({ plugins, manifestText, setManifestText, installManifest, permissions, savePermissions }: { plugins: PaidPlugin[]; manifestText: string; setManifestText: (value: string) => void; installManifest: () => void; permissions: PaidAgentPermissions; savePermissions: (value: PaidAgentPermissions) => void }) {
  const toggle = (key: keyof PaidAgentPermissions) => savePermissions({ ...permissions, [key]: !permissions[key] })
  return <div className="paid-panel-stack"><PanelTitle icon={ShieldCheck} title="Community Plugins + Agent Permissions" subtitle="Manifests, empty tabs, tools, skills, and permission gates." />
    <textarea className="paid-codebox" value={manifestText} onChange={(event) => setManifestText(event.target.value)} rows={8} />
    <button type="button" className="paid-wide-button" onClick={installManifest}><Upload size={14} /> Install Manifest</button>
    <div className="paid-permission-grid">{(['observe', 'suggest', 'draft', 'executeWithApproval', 'trustedLocal'] as Array<keyof PaidAgentPermissions>).map((key) => <label key={key}><input type="checkbox" checked={Boolean(permissions[key])} onChange={() => toggle(key)} />{key}</label>)}</div>
    <h3>Installed plugins</h3>
    {plugins.length ? plugins.map((plugin) => <div className="paid-plugin-row" key={plugin.id}><Boxes size={14} /><div><strong>{plugin.name}</strong><small>{plugin.version} · {(plugin.permissions ?? []).join(', ') || 'no declared permissions'}</small></div></div>) : <p>No plugins installed yet.</p>}
  </div>
}

function ExportPanel({ presets, createExport, activeAsset }: { presets: PaidExportPreset[]; createExport: (presetId: string) => void; activeAsset: PaidAsset | null }) {
  return <div className="paid-panel-stack"><PanelTitle icon={Archive} title="Export Center" subtitle="Web image, print image, YouTube, Reels, audio stems, and project archives." />
    <p>Active asset: <strong>{activeAsset?.name || 'none selected'}</strong></p>
    <div className="paid-export-grid">{presets.map((preset) => <button type="button" key={preset.id} onClick={() => createExport(preset.id)}><Download size={15} /><strong>{preset.label}</strong><small>{preset.summary}</small><span>{preset.outputs.join(', ')}</span></button>)}</div>
  </div>
}

function PanelTitle({ icon: Icon, title, subtitle }: { icon: LucideIcon; title: string; subtitle: string }) {
  return <header className="paid-panel-title"><Icon size={18} /><div><strong>{title}</strong><small>{subtitle}</small></div></header>
}
