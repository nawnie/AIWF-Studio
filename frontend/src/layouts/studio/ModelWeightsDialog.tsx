// Popup shown when a model's weights are not on this PC.
//
// Open on Hugging Face  -> offers the download (size shown) and runs it with progress and Cancel.
// Gated on Hugging Face -> explains the license step and takes the user's key once, then re-checks.
// The key goes only to the local server (never kept in state after saving, never shown again).
// Downloads happen only after the user presses the download button.
import { useCallback, useEffect, useRef, useState } from 'react'
import { DownloadCloud, KeyRound, ShieldAlert, X } from 'lucide-react'
import { cancelModelDownload, fetchModelDownload, fetchModels, saveHfToken, startModelDownload } from './unifiedWorkspaceApi'
import type { ModelDownload, ModelEntry } from './unifiedWorkspaceApi'
import './ModelWeightsDialog.css'

export function formatBytes(value: number | null | undefined): string {
  if (!value || value < 0) return 'unknown size'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let size = value
  let unit = 0
  while (size >= 1024 && unit < units.length - 1) { size /= 1024; unit += 1 }
  return `${size >= 100 || unit === 0 ? Math.round(size) : size.toFixed(1)} ${units[unit]}`
}

const errorText = (error: unknown) => (error instanceof Error ? error.message : 'Unexpected error.')

type Props = {
  model: ModelEntry
  // called after anything changed (key saved, download finished) so the page can refresh its model list
  onChanged: (models: ModelEntry[]) => void
  onClose: () => void
  // optional: let the user continue anyway (for example to see what a preflight says without weights)
  onContinueWithout?: () => void
}

export function ModelWeightsDialog({ model: initial, onChanged, onClose, onContinueWithout }: Props) {
  const [model, setModel] = useState<ModelEntry>(initial)
  const [token, setToken] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [job, setJob] = useState<ModelDownload | null>(null)
  const dialogRef = useRef<HTMLDivElement | null>(null)

  // focus the dialog on open and close it with Escape
  useEffect(() => {
    dialogRef.current?.focus()
    const onKey = (event: KeyboardEvent) => { if (event.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  // re-read the catalog and keep this dialog's model current (e.g. after saving a key)
  const refresh = useCallback(async () => {
    const body = await fetchModels(true)
    onChanged(body.models)
    const next = body.models.find((item) => item.model_id === initial.model_id)
    if (next) setModel(next)
    return next
  }, [initial.model_id, onChanged])

  // poll a running download once a second until it stops
  useEffect(() => {
    if (!job || !['starting', 'running'].includes(job.status)) return undefined
    const timer = window.setInterval(() => {
      fetchModelDownload(job.job_id).then((next) => {
        setJob(next)
        if (next.status === 'completed') void refresh()
      }).catch((err: unknown) => setError(errorText(err)))
    }, 1000)
    return () => window.clearInterval(timer)
  }, [job, refresh])

  const run = async (action: () => Promise<void>) => {
    setBusy(true)
    setError('')
    try { await action() } catch (err) { setError(errorText(err)) } finally { setBusy(false) }
  }

  const access = model.access
  const size = formatBytes(access?.download_bytes)
  const state = access?.state ?? 'unknown'
  const downloading = job && ['starting', 'running'].includes(job.status)
  const percent = job && job.total_bytes > 0 ? Math.min(100, Math.round((job.done_bytes / job.total_bytes) * 100)) : 0
  const repoUrl = model.hf_repo ? `https://huggingface.co/${model.hf_repo}` : ''

  return (
    <div className="model-dialog__backdrop" role="presentation">
      <div className="model-dialog" role="dialog" aria-modal="true" aria-labelledby="model-dialog-title" tabIndex={-1} ref={dialogRef}>
        <header className="model-dialog__head">
          <div>
            <h2 id="model-dialog-title">{model.present ? `${model.label} assets are present` : `${model.label} is not on this PC`}</h2>
            <p>{model.hf_repo ? <>Hugging Face repository <code>{model.hf_repo}</code></> : 'No Hugging Face repository is known for this model.'}</p>
          </div>
          <button type="button" className="model-dialog__close" aria-label="Close" onClick={onClose}><X size={18} aria-hidden="true" /></button>
        </header>

        {model.present && !job && <p className="model-dialog__note">These catalog assets are installed. Run the selected route&apos;s preflight to check its supporting files and runtime before generating.</p>}

        {/* --- download in progress / finished --- */}
        {job && (
          <div className="model-dialog__progress" role="status" aria-live="polite">
            <div className="model-dialog__bar" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent}><span style={{ width: `${percent}%` }} /></div>
            <p>
              {job.status === 'completed' && 'Download finished. Catalog assets are present; route readiness has not been checked.'}
              {job.status === 'cancelled' && 'Download cancelled. Partial files are kept, so downloading again resumes.'}
              {job.status === 'failed' && `Download stopped. ${job.message}`}
              {downloading && `Downloading ${formatBytes(job.done_bytes)} of ${formatBytes(job.total_bytes)} (${percent}%), file ${Math.min(job.files_done + 1, job.files_total)} of ${job.files_total}`}
            </p>
            {downloading && <button type="button" onClick={() => void run(async () => setJob(await cancelModelDownload(job.job_id)))} disabled={busy}>Cancel download</button>}
          </div>
        )}

        {/* --- not on this PC, and not downloading: what to do depends on Hugging Face access --- */}
        {!model.present && !downloading && job?.status !== 'completed' && (
          <>
            {(state === 'open' || state === 'gated_ok') && (
              <div className="model-dialog__body">
                <DownloadCloud size={20} aria-hidden="true" />
                <div>
                  <p>{state === 'open' ? 'This model is open: no account or key is needed.' : 'This model is gated and your saved key has access.'}</p>
                  <p>Download size: <strong>{size}</strong>{access?.file_count ? ` in ${access.file_count} files` : ''}. It is saved in your models folder and can be resumed if interrupted.</p>
                  <div className="model-dialog__actions">
                    <button type="button" className="is-primary" onClick={() => void run(async () => setJob(await startModelDownload(model.model_id)))} disabled={busy}>Download {size}</button>
                    <button type="button" onClick={onClose}>Not now</button>
                  </div>
                </div>
              </div>
            )}

            {(state === 'gated_no_key' || state === 'gated_no_access') && (
              <div className="model-dialog__body">
                <ShieldAlert size={20} aria-hidden="true" />
                <div>
                  <p>
                    {state === 'gated_no_key'
                      ? 'This model is gated by its publisher. To download it you need a free Hugging Face account, to accept the model license on its page, and to add your access key here.'
                      : 'Your key is saved, but this account has not been granted access yet. Accept the license on the model page, then check again.'}
                  </p>
                  {repoUrl && <p><a href={repoUrl} target="_blank" rel="noreferrer">Open {model.hf_repo} on Hugging Face</a> to accept the license. Create a read key at <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer">huggingface.co/settings/tokens</a>.</p>}
                  <form
                    className="model-dialog__key"
                    onSubmit={(event) => {
                      event.preventDefault()
                      void run(async () => {
                        await saveHfToken(token.trim())
                        setToken('')
                        await refresh()
                      })
                    }}
                  >
                    <label>
                      <span><KeyRound size={14} aria-hidden="true" /> Hugging Face access key</span>
                      <input type="password" autoComplete="off" spellCheck={false} value={token} onChange={(event) => setToken(event.target.value)} placeholder="hf_..." aria-label="Hugging Face access key" />
                    </label>
                    <button type="submit" className="is-primary" disabled={busy || !token.trim()}>Save key</button>
                    {state === 'gated_no_access' && <button type="button" onClick={() => void run(async () => { await refresh() })} disabled={busy}>Check again</button>}
                  </form>
                  <p className="model-dialog__fine">The key is checked with Hugging Face, then stored on this PC in Hugging Face&apos;s standard token file. Studio never shows it again.</p>
                </div>
              </div>
            )}

            {(state === 'unavailable' || state === 'unknown') && (
              <div className="model-dialog__body">
                <ShieldAlert size={20} aria-hidden="true" />
                <div>
                  <p>{access?.message || 'Studio could not check this model on Hugging Face.'}</p>
                  <p>You can add the weights by hand to your models folder, or try again once you are online.</p>
                  <div className="model-dialog__actions">
                    <button type="button" onClick={() => void run(async () => { await refresh() })} disabled={busy}>Check again</button>
                  </div>
                </div>
              </div>
            )}
          </>
        )}

        {error && <p className="model-dialog__error" role="alert">{error}</p>}
        <footer className="model-dialog__foot">
          {onContinueWithout && !model.present && <button type="button" onClick={onContinueWithout}>Continue without the weights</button>}
          <button type="button" onClick={onClose}>{model.present || job?.status === 'completed' ? 'Done' : 'Close'}</button>
        </footer>
      </div>
    </div>
  )
}
