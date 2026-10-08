import { useEffect, useRef, useState, type KeyboardEvent } from 'react'
import './AudioProjectControls.css'

export interface AudioProjectChoice {
  project_id: string
  name: string
  updated_at: string
  has_audio: boolean
  audio_missing?: boolean
}

export interface AudioProjectSaveRequest {
  name: string
  projectId?: string
}

export interface AudioProjectControlsProps {
  projects: AudioProjectChoice[]
  currentProjectId?: string | null
  disabled?: boolean
  onRefresh: () => Promise<void> | void
  onSave: (request: AudioProjectSaveRequest) => Promise<void> | void
  onLoad: (projectId: string) => Promise<void> | void
  onBeforeLoad?: (projectId: string) => boolean | Promise<boolean>
}

/** Small accessible dialog controls; all persistence stays behind the caller's API seam. */
export function AudioProjectControls({
  projects,
  currentProjectId,
  disabled = false,
  onRefresh,
  onSave,
  onLoad,
  onBeforeLoad,
}: AudioProjectControlsProps) {
  const [dialog, setDialog] = useState<'save' | 'load' | null>(null)
  const [projectName, setProjectName] = useState('Untitled audio project')
  const [selectedId, setSelectedId] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const dialogRef = useRef<HTMLElement>(null)
  const openerRef = useRef<HTMLElement | null>(null)
  const wasOpenRef = useRef(false)
  const busyNow = disabled || busy
  const selectedProjectId = projects.some((project) => project.project_id === selectedId)
    ? selectedId
    : currentProjectId && projects.some((project) => project.project_id === currentProjectId)
      ? currentProjectId
      : projects[0]?.project_id || ''

  useEffect(() => {
    if (dialog) {
      wasOpenRef.current = true
      return
    }
    if (wasOpenRef.current) {
      wasOpenRef.current = false
      openerRef.current?.focus()
    }
  }, [dialog])

  function openSave(opener: HTMLElement) {
    openerRef.current = opener
    const current = projects.find((project) => project.project_id === currentProjectId)
    setProjectName(current?.name || 'Untitled audio project')
    setError('')
    setDialog('save')
  }

  async function openLoad(opener: HTMLElement) {
    openerRef.current = opener
    setError('')
    setDialog('load')
    try {
      await onRefresh()
    } catch (cause) {
      setError(errorMessage(cause, 'Could not refresh audio projects.'))
    }
  }

  async function save() {
    const name = projectName.trim()
    if (!name) {
      setError('Enter a project name.')
      return
    }
    setBusy(true)
    setError('')
    try {
      await onSave({ name, ...(currentProjectId ? { projectId: currentProjectId } : {}) })
      setDialog(null)
    } catch (cause) {
      setError(errorMessage(cause, 'Could not save this audio project.'))
    } finally {
      setBusy(false)
    }
  }

  async function load() {
    if (!selectedProjectId) return
    setBusy(true)
    setError('')
    try {
      if (onBeforeLoad && !(await onBeforeLoad(selectedProjectId))) return
      await onLoad(selectedProjectId)
      setDialog(null)
    } catch (cause) {
      setError(errorMessage(cause, 'Could not load this audio project.'))
    } finally {
      setBusy(false)
    }
  }

  function cancel() {
    if (!busy) {
      setDialog(null)
      setError('')
    }
  }

  function handleDialogKeyDown(event: KeyboardEvent<HTMLElement>) {
    if (event.key === 'Escape') {
      event.preventDefault()
      cancel()
      return
    }
    if (event.key !== 'Tab') return
    const focusable = dialogRef.current?.querySelectorAll<HTMLElement>(
      'button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), a[href], [tabindex]:not([tabindex="-1"])',
    )
    if (!focusable?.length) {
      event.preventDefault()
      dialogRef.current?.focus()
      return
    }
    const first = focusable[0]
    const last = focusable[focusable.length - 1]
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault()
      last.focus()
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault()
      first.focus()
    }
  }

  return (
    <div className="audio-project-controls">
      <button type="button" disabled={disabled} onClick={(event) => openSave(event.currentTarget)}>Save Project</button>
      <button type="button" disabled={disabled} onClick={(event) => void openLoad(event.currentTarget)}>Load</button>
      {dialog ? (
        <div className="audio-project-backdrop">
          <section
            aria-labelledby="audio-project-dialog-title"
            aria-modal="true"
            className="audio-project-dialog"
            onKeyDown={handleDialogKeyDown}
            ref={dialogRef}
            role="dialog"
            tabIndex={-1}
          >
            <h2 id="audio-project-dialog-title">{dialog === 'save' ? 'Save audio project' : 'Load audio project'}</h2>
            {dialog === 'save' ? (
              <label>
                Project name
                <input
                  autoFocus
                  maxLength={100}
                  onChange={(event) => setProjectName(event.target.value)}
                  value={projectName}
                />
              </label>
            ) : (
              <label>
                Project
                <select
                  autoFocus
                  onChange={(event) => setSelectedId(event.target.value)}
                  value={selectedProjectId}
                >
                  {projects.length === 0 ? <option value="">No saved projects</option> : null}
                  {projects.map((project) => (
                    <option key={project.project_id} value={project.project_id}>
                      {project.name}{project.audio_missing ? ' (audio missing)' : ''}
                    </option>
                  ))}
                </select>
              </label>
            )}
            {error ? <p role="alert">{error}</p> : null}
            {dialog === 'load' && projects.length === 0 ? <p>No saved audio projects yet.</p> : null}
            <footer>
              <button type="button" disabled={busy} onClick={cancel}>Cancel</button>
              {dialog === 'save' ? (
                <button type="button" disabled={busyNow} onClick={() => void save()}>
                  {busy ? 'Saving…' : 'Save'}
                </button>
              ) : (
                <button
                  type="button"
                  disabled={busyNow || !selectedProjectId}
                  onClick={() => void load()}
                >
                  {busy ? 'Loading…' : 'Load'}
                </button>
              )}
            </footer>
          </section>
        </div>
      ) : null}
    </div>
  )
}

function errorMessage(error: unknown, fallback: string): string {
  return error instanceof Error && error.message ? error.message : fallback
}
