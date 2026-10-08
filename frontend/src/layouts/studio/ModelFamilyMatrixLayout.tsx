import { useEffect, useMemo, useState } from 'react'
import { Boxes, Cpu, Database, Filter, Search, ShieldCheck, Video, Zap } from 'lucide-react'
import type { LayoutProps } from './LayoutTypes'
import { CivitaiSupportPanel } from './CivitaiSupportPanel'
import { fetchModelFamilies, fallbackModelFamilyMatrix } from './studioApiClient'
import type { ModelFamily, ModelFamilyMatrix } from './studioApiClient'
import './studioLayouts.css'

type FamilyFilter = 'all' | 'image' | 'video' | 'assistant' | 'gaps'

function familyStatusClass(status: string): string {
  const normalized = status.trim().toLowerCase()
  if (normalized.includes('blocked') || normalized.includes('missing') || normalized.includes('unsupported')) return 'blocked'
  if (normalized.includes('partial') || normalized.includes('experimental') || normalized.includes('gated') || normalized.includes('metadata')) return 'warning'
  if (normalized.includes('supported') || normalized.includes('smoked')) return 'ready'
  return 'neutral'
}

function statusTotal(family: ModelFamily, status: string): number {
  return Number(family.localReadiness?.[status] ?? 0)
}

function localBlockerTotal(family: ModelFamily): number {
  return ['blocked-cleanly', 'broken-runtime', 'unsupported-no-route']
    .reduce((sum, status) => sum + statusTotal(family, status), 0)
}

function totalLocalRows(family: ModelFamily): number {
  return Object.values(family.localReadiness ?? {}).reduce((sum, value) => sum + Number(value || 0), 0)
}

function hasOpenGap(family: ModelFamily): boolean {
  const text = `${family.status} ${family.blockers.join(' ')} ${family.precisions.map((item) => item.status).join(' ')}`.toLowerCase()
  return text.includes('missing') || text.includes('blocked') || text.includes('unsupported') || text.includes('metadata') || text.includes('not-a-current-route')
}

function familyIcon(family: ModelFamily) {
  if (family.category === 'video') return <Video size={16} aria-hidden="true" />
  if (family.category === 'assistant') return <Cpu size={16} aria-hidden="true" />
  return <Boxes size={16} aria-hidden="true" />
}

function precisionStatusClass(status: string): string {
  const normalized = status.trim().toLowerCase()
  if (normalized.includes('blocked') || normalized.includes('missing') || normalized.includes('unsupported') || normalized.includes('not-a-current')) return 'blocked'
  if (normalized === 'supported' || normalized.includes('supported-smoked')) return 'ready'
  if (normalized.includes('supported') || normalized.includes('experimental') || normalized.includes('partial') || normalized.includes('candidate') || normalized.includes('metadata') || normalized.includes('runtime') || normalized.includes('preferred') || normalized.includes('fallback') || normalized.includes('optional') || normalized.includes('cataloged') || normalized.includes('sidecar') || normalized.includes('folder-only') || normalized.includes('provider-dependent') || normalized.includes('configured') || normalized.includes('platform-limited')) return 'warning'
  return 'neutral'
}

function readinessStatusLabel(status: string): string {
  const labels: Record<string, string> = {
    working: 'Smoke-backed',
    'metadata-only': 'Metadata found · runtime unverified',
    'unsupported-no-route': 'No generation route',
    'blocked-cleanly': 'Blocked',
    'broken-runtime': 'Runtime unavailable',
  }
  return labels[status] ?? humanizeStatus(status)
}

function supportStatusLabel(status: string): string {
  const labels: Record<string, string> = {
    supported: 'Supported',
    'supported-gated': 'Supported with access gate',
    'supported-experimental-quants': 'Supported · experimental quant formats',
    'experimental-blocked-on-windows-gguf': 'Experimental · Windows GGUF blocked',
    'blocked-until-runtime': 'Generation unavailable until runtime support is added',
    'supported-plus-sidecars': 'Supported with required sidecars',
    'partial-supported': 'Partially supported',
    partial: 'Partially supported',
    'supported-smoked': 'Smoke-backed',
    'supported-smoked-silent': 'Smoke-backed · silent output observed',
    'supported-when-folder-installed': 'Supported when a complete model folder is installed',
    'runtime-dependent': 'Runtime dependency not verified',
    'experimental-unverified': 'Experimental · runtime unverified',
    'supported-sidecar': 'Supported with an additional runtime',
    'blocked-runtime': 'Runtime unavailable',
    sidecar: 'Requires an additional runtime',
    'blocked-cleanly': 'Blocked · generation not attempted',
    'folder-only': 'Requires a complete Diffusers folder',
    'supported-experimental': 'Experimental support',
    'experimental-supported': 'Experimental support',
    'blocked-probe-only': 'Not available · probe only',
    'metadata-only': 'Metadata only · runtime unverified',
    cataloged: 'Cataloged · generation unsupported',
    optional: 'Optional',
    preferred: 'Preferred',
    fallback: 'Fallback',
    runtime: 'Runtime dependent',
    'provider-dependent': 'Requires a compatible provider',
  }
  return labels[status.trim().toLowerCase()] ?? `Other status: ${humanizeStatus(status)}`
}

function humanizeStatus(status: string): string {
  return status.trim().replace(/[-_]+/g, ' ').replace(/\b\w/g, (letter) => letter.toUpperCase())
}

export function ModelFamilyMatrixLayout(props: LayoutProps) {
  const [matrix, setMatrix] = useState<ModelFamilyMatrix>(() => fallbackModelFamilyMatrix())
  const [query, setQuery] = useState('')
  const [filter, setFilter] = useState<FamilyFilter>('all')
  const [selectedId, setSelectedId] = useState('wan')
  const [loadError, setLoadError] = useState('')
  const [showCivitaiSupport, setShowCivitaiSupport] = useState(false)

  useEffect(() => {
    let active = true
    fetchModelFamilies()
      .then((nextMatrix) => {
        if (!active) return
        setMatrix(nextMatrix)
        setLoadError('')
      })
      .catch(() => {
        if (!active) return
        setMatrix(fallbackModelFamilyMatrix())
        setLoadError('Live support data is unavailable. Showing the offline fallback matrix.')
      })
    return () => { active = false }
  }, [])

  const filteredFamilies = useMemo(() => {
    const needle = query.trim().toLowerCase()
    return matrix.families.filter((family) => {
      const matchesFilter =
        filter === 'all' ||
        family.category === filter ||
        (filter === 'gaps' && hasOpenGap(family))
      if (!matchesFilter) return false
      if (!needle) return true
      return [family.label, family.id, family.category, family.status, family.summary, ...family.storage, ...family.sidecars, ...family.blockers]
        .join(' ')
        .toLowerCase()
        .includes(needle)
    })
  }, [filter, matrix.families, query])

  const selectedFamily = matrix.families.find((family) => family.id === selectedId) ?? filteredFamilies[0] ?? matrix.families[0]
  const readyCount = matrix.families.filter((family) => familyStatusClass(family.status) === 'ready').length
  const gapCount = matrix.families.filter(hasOpenGap).length
  const localRecordCount = Number(matrix.readiness?.recordCount ?? 0)
  const selectedBlockers = (matrix.blockedExamples ?? []).filter((item) => item.family === selectedFamily?.id).slice(0, 6)
  const selectedBlockerTotal = selectedFamily ? localBlockerTotal(selectedFamily) : 0
  const detectedPrecisionRows = Object.values(selectedFamily?.localDetectedPrecisions ?? {})
    .reduce((sum, value) => sum + Number(value || 0), 0)

  return (
    <div className="studio-full-surface studio-family-matrix">
      <header className="studio-family-header">
        <div className="studio-product-lockup">
          <span className="studio-logo-orb"><ShieldCheck size={19} aria-hidden="true" /></span>
          <div>
            <strong>Model Family Support</strong>
            <small role={loadError ? 'alert' : undefined} aria-live="polite">
              {loadError || `${props.bootstrap.workspaceName} · code-indexed loader map`}
            </small>
          </div>
        </div>
        {!showCivitaiSupport ? (
          <div className="studio-family-search studio-search-field">
            <Search size={15} aria-hidden="true" />
            <input value={query} placeholder="Search families, quants, loaders, blockers..." onChange={(event) => setQuery(event.target.value)} />
          </div>
        ) : null}
        <div className="studio-family-chip-row" aria-label="Support views">
          {!showCivitaiSupport ? (['all', 'image', 'video', 'assistant', 'gaps'] as const).map((item) => (
            <button key={item} className={filter === item ? 'active' : ''} type="button" onClick={() => setFilter(item)}>
              <Filter size={13} aria-hidden="true" /> {item}
            </button>
          )) : null}
          <button className={showCivitaiSupport ? 'active' : ''} type="button" aria-pressed={showCivitaiSupport} onClick={() => setShowCivitaiSupport((value) => !value)}>
            {showCivitaiSupport ? 'Model families' : 'Civitai resource types'}
          </button>
        </div>
      </header>

      {!showCivitaiSupport ? <section className="studio-family-scoreboard" aria-label="Support overview">
        <article><span>Families</span><strong>{matrix.families.length}</strong><small>{filteredFamilies.length} visible</small></article>
        <article><span>Code-supported families</span><strong>{readyCount}</strong><small>family support status; local asset readiness is listed below</small></article>
        <article><span>Open gaps</span><strong>{gapCount}</strong><small>blocked, metadata, or missing</small></article>
        <article><span>Local ledger rows</span><strong>{localRecordCount}</strong><small>{matrix.readiness?.error || 'readiness overlay'}</small></article>
      </section> : null}

      {showCivitaiSupport ? <CivitaiSupportPanel /> : (
      <section className="studio-family-body">
        <aside className="studio-family-list" aria-label="Model families">
          {filteredFamilies.map((family) => (
            <button
              key={family.id}
              type="button"
              className={selectedFamily?.id === family.id ? 'studio-family-card active' : 'studio-family-card'}
              onClick={() => setSelectedId(family.id)}
            >
              <span className={`studio-family-status-dot ${familyStatusClass(family.status)}`} />
              <span className="studio-family-card-icon">{familyIcon(family)}</span>
              <strong>{family.label}</strong>
              <small>{family.category} · {supportStatusLabel(family.status)}</small>
              <em>{totalLocalRows(family)} local rows</em>
            </button>
          ))}
        </aside>

        {selectedFamily ? (
          <main className="studio-family-detail" aria-label={`${selectedFamily.label} support details`}>
            <div className="studio-family-detail-hero">
              <div>
                <span className="studio-eyebrow">{selectedFamily.category} family</span>
                <h2>{selectedFamily.label}</h2>
                <p>{selectedFamily.summary}</p>
              </div>
              <div className={`studio-family-big-status ${familyStatusClass(selectedFamily.status)}`}>
                <Zap size={16} aria-hidden="true" />
                {supportStatusLabel(selectedFamily.status)}
              </div>
            </div>

            <div className="studio-family-grid">
              <section className="studio-family-panel wide">
                <header><strong>Precision and quant support</strong><small>{detectedPrecisionRows} local precision hits</small></header>
                <div className="studio-precision-table" role="table">
                  <div role="row" className="head"><span>Precision</span><span>Status</span><span>Loader</span><span>Notes</span></div>
                  {selectedFamily.precisions.map((precision) => (
                    <div key={`${selectedFamily.id}-${precision.name}-${precision.status}`} role="row">
                      <strong>{precision.name}</strong>
                      <span className={`studio-status-pill ${precisionStatusClass(precision.status)}`}>{supportStatusLabel(precision.status)}</span>
                      <span>{precision.loader}</span>
                      <small>{precision.notes || 'current code path'}</small>
                    </div>
                  ))}
                </div>
              </section>

              <section className="studio-family-panel">
                <header><strong>Local readiness</strong><small>from pipeline_readiness</small></header>
                <div className="studio-family-meters">
                  {['working', 'metadata-only', 'unsupported-no-route', 'blocked-cleanly', 'broken-runtime'].map((status) => (
                    <div key={status}><span>{readinessStatusLabel(status)}</span><strong>{statusTotal(selectedFamily, status)}</strong></div>
                  ))}
                </div>
              </section>

              <section className="studio-family-panel">
                <header><strong>Storage contracts</strong><small>accepted shapes</small></header>
                <div className="studio-family-token-wrap">
                  {selectedFamily.storage.map((item) => <span key={item}>{item}</span>)}
                </div>
              </section>

              <section className="studio-family-panel wide">
                <header><strong>Routes and loaders</strong><small>how the model is actually loaded</small></header>
                <div className="studio-route-list">
                  {selectedFamily.routes.map((route) => (
                    <article key={route.id}>
                      <strong>{route.id}</strong>
                      <span className={`studio-status-pill ${precisionStatusClass(route.status)}`}>{supportStatusLabel(route.status)}</span>
                      <small>{route.kind}</small>
                      <code>{route.entrypoint}</code>
                      {route.notes ? <p>{route.notes}</p> : null}
                    </article>
                  ))}
                </div>
              </section>

              <section className="studio-family-panel">
                <header><strong>Sidecars</strong><small>required companions</small></header>
                <div className="studio-family-token-wrap">
                  {selectedFamily.sidecars.map((item) => <span key={item}>{item}</span>)}
                </div>
              </section>

              <section className="studio-family-panel">
                <header><strong>LoRA policy</strong><small>adapter truth serum</small></header>
                <p>{selectedFamily.lora}</p>
              </section>

              <section className="studio-family-panel wide">
                <header><strong>Blockers and missing pieces</strong><small>do not expose as working</small></header>
                <ul className="studio-family-blockers">
                  {selectedFamily.blockers.map((item) => <li key={item}>{item}</li>)}
                </ul>
              </section>
            </div>
          </main>
        ) : null}

        <aside className="studio-family-evidence" aria-label="Source evidence">
          <section>
            <header><Database size={15} aria-hidden="true" /><strong>Precision vocabulary</strong></header>
            <div className="studio-family-token-wrap compact">
              {matrix.precisionVocabulary.map((item) => <span key={item}>{item}</span>)}
            </div>
          </section>
          <section>
            <header><Cpu size={15} aria-hidden="true" /><strong>Code modules</strong></header>
            <ul>
              {(selectedFamily?.modules ?? []).map((item) => <li key={item}><code>{item}</code></li>)}
            </ul>
          </section>
          <section>
            <header><ShieldCheck size={15} aria-hidden="true" /><strong>Blocked examples</strong></header>
            <ul>
              {selectedBlockers.map((item) => (
                <li key={`${item.status}-${item.path}-${item.route}`}>
                  <strong>{item.status}</strong>
                  <span>{item.reason || item.route || item.path}</span>
                </li>
              ))}
              {selectedBlockers.length === 0 && selectedBlockerTotal === 0 ? <li><span>No local blockers found for this family.</span></li> : null}
              {selectedBlockers.length === 0 && selectedBlockerTotal > 0 ? <li><span>{selectedBlockerTotal} local blocker records exist; details are outside the current example limit.</span></li> : null}
            </ul>
          </section>
        </aside>
      </section>
      )}
    </div>
  )
}
