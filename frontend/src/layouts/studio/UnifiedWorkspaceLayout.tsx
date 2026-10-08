// Studio Flow: one surface that links AIWF Studio, Dataset Studio, ReTrain and Qwen Chat.
//
// Every cross-app action goes through AIWF Pro's same-origin /api/pro/unified/* routes
// (see unifiedWorkspaceApi.ts). The browser never talks to the other apps directly.
// Nothing here starts training: ReTrain is reached only for import and dry-run preflight.
//
// Source text is ASCII only. Separators use HTML entities (&middot; &rarr; &hellip;)
// so no editor or shell encoding can turn them into replacement characters.
import { useCallback, useEffect, useMemo, useState } from 'react'
import { Activity, CheckCircle2, CircleDashed, Database, DownloadCloud, FolderGit2, Image as ImageIcon, LockKeyhole, MessageSquare, RefreshCw, Send, ShieldCheck } from 'lucide-react'
import type { RecentOutput } from '../../types'
import {
  askQwen, catalogOutputs, createProject, fetchModelFamilies, fetchModels, fetchPackages, fetchProject, fetchProjects, fetchQwenContext,
  fetchUnifiedStatus, importPackage, runPreflight, UnifiedApiError,
} from './unifiedWorkspaceApi'
import type {
  Capability, CatalogResult, DatasetPackage, ImportedDataset, ModelEntry, ModelFamily, PreflightResult, ProjectDetail, ProjectSummary, QwenAnswer, UnifiedStatus,
} from './unifiedWorkspaceApi'
import { ModelWeightsDialog } from './ModelWeightsDialog'
import './UnifiedWorkspaceLayout.css'

// --- small display helpers ------------------------------------------------------------
const shortHash = (value?: string) => (value ? value.slice(0, 12) : '')
const errorText = (error: unknown) => (error instanceof UnifiedApiError || error instanceof Error ? error.message : 'Unexpected error.')

// this maps a capability to the short headline shown in each app card
function capabilityHeadline(cap: Capability | undefined, readyText: string): string {
  if (!cap) return 'Not checked yet'
  if (cap.available && cap.state === 'ready') return readyText
  if (cap.available) return 'Running, partly available'
  switch (cap.state) {
    case 'not_running': return 'Not running'
    case 'auth_rejected': return 'Answered, but rejected credentials'
    case 'route_missing': return 'Answered, but missing the route'
    case 'not_configured': return 'Not configured'
    default: return 'Unavailable'
  }
}

// RecentOutput.path is the absolute output path from Pro; url is /api/pro/outputs/<path>.
// The backend confines either form to Studio's output folder.
function outputReference(output: RecentOutput): string | null {
  if (output.path) return output.path
  if (output.url?.startsWith('/api/pro/outputs/')) return output.url
  return null
}

type Busy = '' | 'status' | 'project' | 'catalog' | 'packages' | 'import' | 'preflight' | 'context' | 'qwen'

export function UnifiedWorkspaceLayout({ recentOutputs }: { recentOutputs: RecentOutput[] }) {
  // --- connection status for the three sibling apps ------------------------------------
  const [status, setStatus] = useState<UnifiedStatus | null>(null)
  const [statusMessage, setStatusMessage] = useState('Checking connections\u2026')
  const [busy, setBusy] = useState<Busy>('')

  // --- shared project identity ----------------------------------------------------------
  const [projects, setProjects] = useState<ProjectSummary[]>([])
  const [project, setProject] = useState<ProjectDetail | null>(null)
  const [newProjectName, setNewProjectName] = useState('')
  const [projectError, setProjectError] = useState('')

  // --- step state: catalog, packages, import, preflight, Qwen -----------------------------
  const [selectedIds, setSelectedIds] = useState<string[]>([])
  const [catalogResult, setCatalogResult] = useState<CatalogResult | null>(null)
  const [catalogError, setCatalogError] = useState('')
  const [packages, setPackages] = useState<DatasetPackage[] | null>(null)
  const [packagesError, setPackagesError] = useState('')
  const [selectedPackage, setSelectedPackage] = useState<string>('')
  const [imported, setImported] = useState<ImportedDataset | null>(null)
  const [importError, setImportError] = useState('')
  const [retrainModel, setRetrainModel] = useState('')
  const [method, setMethod] = useState('QLoRA')
  const [preflight, setPreflight] = useState<PreflightResult | null>(null)
  const [preflightError, setPreflightError] = useState('')
  const [contextPreview, setContextPreview] = useState('')
  const [qwenModel, setQwenModel] = useState('')
  const [question, setQuestion] = useState('')
  const [qwenAnswer, setQwenAnswer] = useState<QwenAnswer | null>(null)
  const [qwenError, setQwenError] = useState('')
  const [families, setFamilies] = useState<ModelFamily[]>([])
  const [weightsModel, setWeightsModel] = useState<ModelEntry | null>(null)
  const [weightsError, setWeightsError] = useState('')

  const caps = status?.capabilities
  const datasetCap = caps?.dataset_studio
  const retrainCap = caps?.retrain
  const qwenCap = caps?.qwen_chat
  const imageCap = caps?.comfyui
  const projectId = project?.project_id ?? ''

  // --- status and package loading (read-only GETs) ------------------------------------
  // State is applied in promise callbacks so the mount-time loads do not cascade renders.
  const applyStatus = useCallback((request: Promise<UnifiedStatus>) => request
    .then((next) => {
      setStatus(next)
      setStatusMessage(`Connections checked ${new Date(next.checked_at).toLocaleTimeString()}.`)
    })
    .catch((error: unknown) => {
      setStatus(null)
      setStatusMessage(errorText(error))
    }), [])

  const refreshStatus = useCallback(async () => {
    setBusy('status')
    await applyStatus(fetchUnifiedStatus())
    setBusy('')
  }, [applyStatus])

  const applyPackages = useCallback((request: Promise<DatasetPackage[]>) => request
    .then((next) => {
      setPackagesError('')
      setPackages(next)
      // a previous selection survives only if that exact revision is still listed
      setSelectedPackage((current) => (next.some((item) => `${item.package_name}|${item.manifest_sha256}` === current) ? current : ''))
    })
    .catch((error: unknown) => {
      setPackages(null)
      setPackagesError(errorText(error))
    }), [])

  const loadPackages = useCallback(async () => {
    setBusy('packages')
    await applyPackages(fetchPackages())
    setBusy('')
  }, [applyPackages])

  // this effect loads status, projects and the model-family table once on open
  useEffect(() => {
    void applyStatus(fetchUnifiedStatus())
    fetchProjects().then(setProjects).catch(() => setProjects([]))
    fetchModelFamilies().then(setFamilies).catch(() => setFamilies([]))
  }, [applyStatus])

  // this effect lists packages once Dataset Studio reports its list route is present
  useEffect(() => {
    if (datasetCap?.package_listing) void applyPackages(fetchPackages())
  }, [datasetCap?.package_listing, applyPackages])

  // model pickers: an explicit user choice wins; otherwise the first text-capable
  // ReTrain model and the first loaded Qwen model are shown as the default
  const retrainModels = useMemo(() => (retrainCap?.models ?? []).filter((item) => item.text_capable !== false), [retrainCap])
  const qwenModels = useMemo(() => qwenCap?.models ?? [], [qwenCap])
  const activeRetrainModel = retrainModels.some((item) => item.model_id === retrainModel) ? retrainModel : (retrainModels[0]?.model_id ?? '')
  const activeQwenModel = qwenModels.some((item) => item.model_id === qwenModel) ? qwenModel : ((qwenModels.find((item) => item.loaded) ?? qwenModels[0])?.model_id ?? '')

  // --- project actions ----------------------------------------------------------------
  const openProject = async (id: string) => {
    setProjectError('')
    setCatalogResult(null); setImported(null); setPreflight(null); setQwenAnswer(null); setContextPreview('')
    if (!id) { setProject(null); return }
    setBusy('project')
    try {
      setProject(await fetchProject(id))
    } catch (error) {
      setProjectError(errorText(error))
    } finally {
      setBusy('')
    }
  }

  const refreshProject = async () => {
    if (!projectId) return
    try {
      setProject(await fetchProject(projectId))
      setProjects(await fetchProjects())
    } catch {
      // the action already succeeded; a stale activity list is not worth an error banner
    }
  }

  const submitNewProject = async () => {
    const name = newProjectName.trim()
    if (!name) return
    setBusy('project')
    setProjectError('')
    try {
      const created = await createProject(name)
      setProject(created)
      setProjects(await fetchProjects())
      setNewProjectName('')
      setCatalogResult(null); setImported(null); setPreflight(null); setQwenAnswer(null); setContextPreview('')
    } catch (error) {
      setProjectError(errorText(error))
    } finally {
      setBusy('')
    }
  }

  // --- step 1: catalog selected Studio outputs in Dataset Studio -------------------------
  const toggleOutput = (id: string) => setSelectedIds((current) => (current.includes(id) ? current.filter((item) => item !== id) : [...current, id]))
  const selectedRefs = recentOutputs.filter((item) => selectedIds.includes(item.id)).map(outputReference).filter((item): item is string => Boolean(item))
  const canCatalog = Boolean(projectId && datasetCap?.catalog_studio_outputs && selectedRefs.length && !busy)

  const submitCatalog = async () => {
    setBusy('catalog')
    setCatalogError('')
    setCatalogResult(null)
    try {
      setCatalogResult(await catalogOutputs(projectId, selectedRefs))
      await refreshProject()
    } catch (error) {
      setCatalogError(errorText(error))
    } finally {
      setBusy('')
    }
  }

  // --- step 2: import the explicitly selected package revision into ReTrain --------------
  const chosenPackage = packages?.find((item) => `${item.package_name}|${item.manifest_sha256}` === selectedPackage && item.status === 'ready')
  const canImport = Boolean(projectId && retrainCap?.available && chosenPackage && !busy)

  const submitImport = async () => {
    if (!chosenPackage?.manifest_sha256) return
    setBusy('import')
    setImportError('')
    setImported(null)
    setPreflight(null)
    try {
      setImported(await importPackage(projectId, { package_name: chosenPackage.package_name, manifest_sha256: chosenPackage.manifest_sha256 }))
      await refreshProject()
    } catch (error) {
      const message = errorText(error)
      setImportError(error instanceof UnifiedApiError && error.code === 'stale_revision' ? `${message} The list was refreshed.` : message)
      if (error instanceof UnifiedApiError && error.code === 'stale_revision') void loadPackages()
    } finally {
      setBusy('')
    }
  }

  // --- step 3: dry-run preflight bound to the imported revision -------------------------
  const canPreflight = Boolean(projectId && imported && activeRetrainModel && retrainCap?.available && !busy)
  const submitPreflight = async () => {
    if (!imported) return
    setBusy('preflight')
    setPreflightError('')
    setPreflight(null)
    try {
      setPreflight(await runPreflight(projectId, imported, activeRetrainModel, method))
      await refreshProject()
    } catch (error) {
      setPreflightError(errorText(error))
    } finally {
      setBusy('')
    }
  }

  // --- step 4: explicit project context to Qwen Chat -------------------------------------
  const loadContext = async () => {
    setBusy('context')
    setQwenError('')
    try {
      setContextPreview(await fetchQwenContext(projectId))
    } catch (error) {
      setQwenError(errorText(error))
    } finally {
      setBusy('')
    }
  }
  const canAsk = Boolean(projectId && qwenCap?.available && activeQwenModel && question.trim() && contextPreview && !busy)
  const submitQuestion = async () => {
    setBusy('qwen')
    setQwenError('')
    setQwenAnswer(null)
    try {
      const answer = await askQwen(projectId, activeQwenModel, question.trim())
      setQwenAnswer(answer)
      setContextPreview(answer.context_sent)
      await refreshProject()
    } catch (error) {
      setQwenError(errorText(error))
    } finally {
      setBusy('')
    }
  }

  // gates that are not "ready" are the reasons behind a warning or blocked plan
  const openGates = preflight ? preflight.result.gates.filter((item) => item.state !== 'ready') : []
  const missingDependencies = preflight ? preflight.result.dependencies.filter((item) => !item.available) : []
  const estimate = preflight?.result.estimate
  const openWeights = async () => {
    setWeightsError('')
    try {
      const catalog = await fetchModels(true)
      const selected = catalog.models.find((item) => item.model_id === activeRetrainModel)
      if (!selected) {
        setWeightsError('Weights are not available in the local download catalog for this ReTrain model.')
        return
      }
      setWeightsModel(selected)
    } catch (error) {
      setWeightsError(errorText(error))
    }
  }

  return (
    <section className="pro-workspace-surface unified-workspace" aria-label="Unified Studio workflow">
      <header className="unified-workspace__header">
        <div>
          <span className="unified-workspace__eyebrow">Shared project workflow</span>
          <h1>Studio Flow</h1>
          <p>AIWF Studio &rarr; Dataset Studio &rarr; ReTrain &rarr; Qwen Chat, linked by one project ID.</p>
        </div>
        <button type="button" className="unified-workspace__refresh" onClick={() => void refreshStatus()} disabled={busy === 'status'}>
          <RefreshCw size={16} aria-hidden="true" /> {busy === 'status' ? <>Checking&hellip;</> : 'Check connection status'}
        </button>
      </header>

      {/* --- connection cards: one honest line per sibling app --- */}
      <div className="unified-workspace__apps" aria-label="Connected apps">
        {([
          ['Dataset Studio', datasetCap, 'Connected', Database],
          ['ReTrain', retrainCap, 'Connected (dry run only)', Activity],
          ['Qwen Chat', qwenCap, 'Connected', MessageSquare],
          ['Qwen Image 2.1', imageCap, 'Workflow prerequisites detected', ImageIcon],
        ] as const).map(([label, cap, readyText, Icon]) => (
          <article key={label} className={`unified-workspace__app unified-workspace__app--${cap?.available ? (cap.state === 'ready' ? 'ready' : 'partial') : 'down'}`}>
            <span className="unified-workspace__stage-label"><Icon size={14} aria-hidden="true" /> {label}</span>
            <strong>{capabilityHeadline(cap, readyText)}</strong>
            <p>{cap?.reason ?? 'Status has not been received from AIWF Pro.'}</p>
          </article>
        ))}
      </div>
      <p className="unified-workspace__status" aria-live="polite">{statusMessage}</p>

      {/* --- shared project identity --- */}
      <section className="unified-workspace__project" aria-labelledby="unified-project-title">
        <div className="unified-workspace__section-head">
          <div>
            <h2 id="unified-project-title"><FolderGit2 size={16} aria-hidden="true" /> Shared project</h2>
            <p>The project ID is written into Dataset Studio as a tag and recorded with every import, preflight and Qwen message.</p>
          </div>
        </div>
        <div className="unified-workspace__project-row">
          <label>
            <span>Project</span>
            <select aria-label="Choose shared project" value={projectId} onChange={(event) => void openProject(event.target.value)} disabled={busy === 'project'}>
              <option value="">Choose a project&hellip;</option>
              {projects.map((item) => <option key={item.project_id} value={item.project_id}>{item.name}</option>)}
            </select>
          </label>
          <label>
            <span>New project</span>
            <input aria-label="New project name" value={newProjectName} maxLength={120} placeholder="Project name" onChange={(event) => setNewProjectName(event.target.value)} />
          </label>
          <button type="button" onClick={() => void submitNewProject()} disabled={!newProjectName.trim() || busy === 'project'}>Create project</button>
        </div>
        <div className="unified-workspace__identity" role="status">
          <LockKeyhole size={17} aria-hidden="true" />
          {project
            ? <span><strong>Shared project identity:</strong> {project.name} <code>{project.project_id}</code></span>
            : <span><strong>Shared project identity:</strong> none selected. Choose or create a project to enable cross-app actions.</span>}
        </div>
        {projectError && <p className="unified-workspace__error" role="alert">{projectError}</p>}
      </section>

      {/* --- step 1: Studio outputs -> Dataset Studio --- */}
      <section className="unified-workspace__step" aria-labelledby="unified-step-catalog">
        <div className="unified-workspace__section-head">
          <div>
            <h2 id="unified-step-catalog"><span className="unified-workspace__step-no">1</span> Catalog Studio outputs in Dataset Studio</h2>
            <p>Adds the files to Dataset Studio&apos;s catalog with the project tag, model tag, and the generation prompt as an import caption.</p>
          </div>
          <span>{selectedIds.length} selected &middot; {recentOutputs.length} available</span>
        </div>
        {recentOutputs.length === 0 ? <p className="unified-workspace__empty">No Studio output receipts are loaded.</p> : (
          <ul className="unified-workspace__output-list">
            {recentOutputs.map((output) => (
              <li key={output.id}>
                <label>
                  <input type="checkbox" checked={selectedIds.includes(output.id)} onChange={() => toggleOutput(output.id)} />
                  <span className="unified-workspace__output-name">{output.prompt || output.id}</span>
                  <small>{output.modelName || output.mode} &middot; {output.width} &times; {output.height}</small>
                </label>
              </li>
            ))}
          </ul>
        )}
        <div className="unified-workspace__actions">
          <button type="button" onClick={() => void submitCatalog()} disabled={!canCatalog}>
            {busy === 'catalog' ? <>Cataloging&hellip;</> : `Catalog ${selectedRefs.length || ''} in Dataset Studio`}
          </button>
          {!datasetCap?.catalog_studio_outputs && <small>Needs Dataset Studio running with Studio&apos;s output folder as a catalog root.</small>}
        </div>
        {catalogError && <p className="unified-workspace__error" role="alert">{catalogError}</p>}
        {catalogResult && (
          <div className="unified-workspace__result" role="status">
            <CheckCircle2 size={16} aria-hidden="true" />
            <div>
              <strong>Cataloged {catalogResult.assets.length} asset{catalogResult.assets.length === 1 ? '' : 's'} in &ldquo;{catalogResult.collection.name}&rdquo;</strong>
              <p>Asset IDs {catalogResult.assets.map((item) => item.asset_id).join(', ')} &middot; tag <code>{catalogResult.project_tag}</code></p>
              <p className="unified-workspace__note">{catalogResult.note}</p>
            </div>
          </div>
        )}
      </section>

      {/* --- step 2: explicit package -> ReTrain import --- */}
      <section className="unified-workspace__step" aria-labelledby="unified-step-import">
        <div className="unified-workspace__section-head">
          <div>
            <h2 id="unified-step-import"><span className="unified-workspace__step-no">2</span> Send a Dataset Studio text package to ReTrain</h2>
            <p>Pick one package revision. Studio downloads Dataset Studio&apos;s verified ZIP, checks the revision hash, and ReTrain re-verifies every row.</p>
          </div>
          <button type="button" onClick={() => void loadPackages()} disabled={!datasetCap?.package_listing || busy === 'packages'}>Refresh packages</button>
        </div>
        {packagesError && <p className="unified-workspace__error" role="alert">{packagesError}</p>}
        {packages === null && !packagesError && <p className="unified-workspace__empty">Package list unavailable until Dataset Studio is connected.</p>}
        {packages?.length === 0 && <p className="unified-workspace__empty">Dataset Studio has no published ReTrain packages yet. Create one from text assets in Dataset Studio.</p>}
        {packages && packages.length > 0 && (
          <ul className="unified-workspace__package-list" role="radiogroup" aria-label="Dataset Studio packages">
            {packages.map((item) => {
              const key = `${item.package_name}|${item.manifest_sha256}`
              return (
                <li key={key} className={item.status === 'ready' ? '' : 'is-invalid'}>
                  <label>
                    <input type="radio" name="unified-package" value={key} checked={selectedPackage === key} disabled={item.status !== 'ready'} onChange={() => setSelectedPackage(key)} />
                    <span className="unified-workspace__output-name">{item.package_name}</span>
                    {item.status === 'ready'
                      ? <small>rev <code>{shortHash(item.manifest_sha256)}</code> &middot; {item.counts?.train ?? 0} train / {item.counts?.validation ?? 0} validation rows &middot; {item.modality}</small>
                      : <small>Not selectable: {item.reason}</small>}
                  </label>
                </li>
              )
            })}
          </ul>
        )}
        <div className="unified-workspace__actions">
          <button type="button" onClick={() => void submitImport()} disabled={!canImport}>
            {busy === 'import' ? <>Importing&hellip;</> : 'Import selected revision into ReTrain'}
          </button>
          {!retrainCap?.available && <small>Needs the ReTrain API with the package import route.</small>}
        </div>
        {importError && <p className="unified-workspace__error" role="alert">{importError}</p>}
        {imported && (
          <div className="unified-workspace__result" role="status">
            <CheckCircle2 size={16} aria-hidden="true" />
            <div>
              <strong>{imported.reused ? 'Already in ReTrain (same revision reused)' : 'Imported into ReTrain'}: {imported.package_name}</strong>
              <p>Dataset <code>{imported.dataset_id.slice(0, 19)}&hellip;</code> &middot; {imported.counts.train ?? 0} train / {imported.counts.validation ?? 0} validation rows &middot; {imported.modality}</p>
            </div>
          </div>
        )}
      </section>

      {/* --- step 3: preflight (dry run only) --- */}
      <section className="unified-workspace__step" aria-labelledby="unified-step-preflight">
        <div className="unified-workspace__section-head">
          <div>
            <h2 id="unified-step-preflight"><span className="unified-workspace__step-no">3</span> ReTrain preflight (dry run)</h2>
            <p>Plans a run against exactly the imported revision. Training is never started from Studio Flow; start it in ReTrain.</p>
          </div>
        </div>
        <div className="unified-workspace__project-row">
          <label>
            <span>Base model</span>
            <select aria-label="ReTrain model" value={activeRetrainModel} onChange={(event) => setRetrainModel(event.target.value)} disabled={!retrainModels.length}>
              {!retrainModels.length && <option value="">No ReTrain models listed</option>}
              {retrainModels.map((item) => (
                <option key={item.model_id} value={item.model_id}>{item.label ?? item.model_id}{item.local_weights_present === false ? ' (weights not downloaded)' : ''}</option>
              ))}
            </select>
          </label>
          <button type="button" onClick={() => void openWeights()} disabled={!activeRetrainModel || !retrainCap?.available}>
            <DownloadCloud size={15} aria-hidden="true" /> Model weights
          </button>
          <label>
            <span>Method</span>
            <select aria-label="Training method" value={method} onChange={(event) => setMethod(event.target.value)}>
              {['QLoRA', 'LoRA', 'Full fine-tune'].map((item) => <option key={item}>{item}</option>)}
            </select>
          </label>
          <button type="button" onClick={() => void submitPreflight()} disabled={!canPreflight}>
            {busy === 'preflight' ? <>Planning&hellip;</> : 'Run preflight'}
          </button>
        </div>
        {weightsError && <p className="unified-workspace__error" role="alert">{weightsError}</p>}
        {!imported && <p className="unified-workspace__empty">Import a package first; the preflight is bound to that revision.</p>}
        {preflightError && <p className="unified-workspace__error" role="alert">{preflightError}</p>}
        {preflight && (
          <div className="unified-workspace__result" role="status">
            <ShieldCheck size={16} aria-hidden="true" />
            <div>
              <strong>Plan status: {preflight.result.plan_status ?? 'unknown'} &middot; training not started</strong>
              <p>{preflight.message} Model {preflight.result.model_id} on revision <code>{shortHash(preflight.result.manifest_sha256)}</code>.</p>
              {estimate?.fit_state && <p>VRAM estimate: {estimate.estimated_gb} GB of {estimate.limit_gb} GB ({estimate.fit_state}).</p>}
              {openGates.length > 0 && (
                <ul className="unified-workspace__issues" aria-label="Preflight gates needing attention">
                  {openGates.map((item) => <li key={item.gate}><strong>{item.gate}</strong> ({item.state}): {item.detail}</li>)}
                </ul>
              )}
              {missingDependencies.length > 0 && <p>Missing in ReTrain&apos;s environment: {missingDependencies.map((item) => item.label).join(', ')}</p>}
              {openGates.length === 0 && missingDependencies.length === 0 && <p>All {preflight.result.gates.length} gates are ready.</p>}
            </div>
          </div>
        )}
      </section>

      {/* --- step 4: explicit project context -> Qwen Chat --- */}
      <section className="unified-workspace__step" aria-labelledby="unified-step-qwen">
        <div className="unified-workspace__section-head">
          <div>
            <h2 id="unified-step-qwen"><span className="unified-workspace__step-no">4</span> Ask Qwen Chat about this project</h2>
            <p>Preview exactly what will be sent: project name, IDs, counts and revision hashes. No files, paths, or chat history.</p>
          </div>
          <button type="button" onClick={() => void loadContext()} disabled={!projectId || busy === 'context'}>Preview context</button>
        </div>
        {contextPreview && <pre className="unified-workspace__context" aria-label="Context sent to Qwen Chat">{contextPreview}</pre>}
        <div className="unified-workspace__project-row">
          <label>
            <span>Qwen model</span>
            <select aria-label="Qwen Chat model" value={activeQwenModel} onChange={(event) => setQwenModel(event.target.value)} disabled={!qwenModels.length}>
              {!qwenModels.length && <option value="">Qwen Chat not connected</option>}
              {qwenModels.map((item) => <option key={item.model_id} value={item.model_id}>{item.label ?? (item.family ? `${item.model_id} (${item.family})` : item.model_id)}{item.loaded ? ' (loaded)' : ' (not reported loaded)'}</option>)}
            </select>
          </label>
          <label className="unified-workspace__grow">
            <span>Question</span>
            <input aria-label="Question for Qwen Chat" value={question} maxLength={4000} placeholder="e.g. Is this package a good size for a first QLoRA run?" onChange={(event) => setQuestion(event.target.value)} />
          </label>
          <button type="button" onClick={() => void submitQuestion()} disabled={!canAsk}>
            <Send size={15} aria-hidden="true" /> {busy === 'qwen' ? <>Sending&hellip;</> : 'Send with context'}
          </button>
        </div>
        {!contextPreview && projectId && <p className="unified-workspace__empty">Preview the context before sending; nothing is sent until you press Send.</p>}
        {qwenError && <p className="unified-workspace__error" role="alert">{qwenError}</p>}
        {qwenAnswer && (
          <div className="unified-workspace__result" role="status">
            <MessageSquare size={16} aria-hidden="true" />
            <div><strong>{qwenAnswer.model_id}</strong><p className="unified-workspace__answer">{qwenAnswer.answer}</p></div>
          </div>
        )}
      </section>

      {/* --- trained-model compatibility and project activity --- */}
      <section className="unified-workspace__step" aria-labelledby="unified-families">
        <div className="unified-workspace__section-head">
          <div>
            <h2 id="unified-families"><CircleDashed size={16} aria-hidden="true" /> Trained model compatibility</h2>
            <p>Artifacts from these apps are not interchangeable across model families.</p>
          </div>
        </div>
      {families.length > 0 && (
          <table className="unified-workspace__families">
            <thead><tr><th>Artifact</th><th>Works in</th><th>Does not work in</th></tr></thead>
            <tbody>
              {families.map((item) => (
                <tr key={item.artifact}><td><strong>{item.artifact}</strong><small>{item.family}</small></td><td>{item.usable_in.join('; ')}</td><td>{item.not_usable_in.join('; ')}</td></tr>
              ))}
            </tbody>
          </table>
      )}
      {weightsModel && (
        <ModelWeightsDialog
          model={weightsModel}
          onChanged={(models) => {
            setStatus((current) => current ? {
              ...current,
              capabilities: {
                ...current.capabilities,
                retrain: {
                  ...current.capabilities.retrain,
                  models: models.map((item) => ({
                    model_id: item.model_id,
                    label: item.label,
                    text_capable: item.text_capable,
                    local_weights_present: item.present,
                  })),
                },
              },
            } : current)
            setWeightsModel(models.find((item) => item.model_id === weightsModel.model_id) ?? null)
          }}
          onClose={() => setWeightsModel(null)}
          onContinueWithout={() => setWeightsModel(null)}
        />
      )}
        {project && project.events.length > 0 && (
          <ol className="unified-workspace__events" aria-label="Project activity">
            {[...project.events].reverse().slice(0, 12).map((event) => (
              <li key={event.event_id}>
                <code>{event.kind}</code>
                {typeof event.question === 'string' && <span>: {event.question}</span>}
                <small>{new Date(event.at).toLocaleString()}</small>
              </li>
            ))}
          </ol>
        )}
      </section>
    </section>
  )
}
