import { useEffect, useMemo, useState } from 'react'
import { Boxes, Cpu, Database, Filter, Search, ShieldCheck, Video, Zap } from 'lucide-react'
import type { PaidLayoutProps } from './PaidLayoutTypes'
import { fetchPaidModelFamilies, fallbackPaidModelFamilyMatrix } from './paidApiClient'
import type { PaidModelFamily, PaidModelFamilyMatrix } from './paidApiClient'
import './paidLayouts.css'

type FamilyFilter = 'all' | 'image' | 'video' | 'assistant' | 'gaps'

function familyStatusClass(status: string): string {
  const normalized = status.toLowerCase()
  if (normalized.includes('blocked') || normalized.includes('missing')) return 'blocked'
  if (normalized.includes('partial') || normalized.includes('experimental') || normalized.includes('gated')) return 'warning'
  if (normalized.includes('supported') || normalized.includes('smoked')) return 'ready'
  return 'neutral'
}

function statusTotal(family: PaidModelFamily, status: string): number {
  return Number(family.localReadiness?.[status] ?? 0)
}

function totalLocalRows(family: PaidModelFamily): number {
  return Object.values(family.localReadiness ?? {}).reduce((sum, value) => sum + Number(value || 0), 0)
}

function hasOpenGap(family: PaidModelFamily): boolean {
  const text = `${family.status} ${family.blockers.join(' ')} ${family.precisions.map((item) => item.status).join(' ')}`.toLowerCase()
  return text.includes('missing') || text.includes('blocked') || text.includes('metadata') || text.includes('not-a-current-route')
}

function familyIcon(family: PaidModelFamily) {
  if (family.category === 'video') return <Video size={16} aria-hidden="true" />
  if (family.category === 'assistant') return <Cpu size={16} aria-hidden="true" />
  return <Boxes size={16} aria-hidden="true" />
}

function precisionStatusClass(status: string): string {
  const normalized = status.toLowerCase()
  if (normalized.includes('blocked') || normalized.includes('missing') || normalized.includes('not-a-current')) return 'blocked'
  if (normalized.includes('experimental') || normalized.includes('partial') || normalized.includes('candidate') || normalized.includes('metadata')) return 'warning'
  return 'ready'
}

export function ModelFamilyMatrixPaidLayout(props: PaidLayoutProps) {
  const [matrix, setMatrix] = useState<PaidModelFamilyMatrix>(() => fallbackPaidModelFamilyMatrix())
  const [query, setQuery] = useState('')
  const [filter, setFilter] = useState<FamilyFilter>('all')
  const [selectedId, setSelectedId] = useState('wan')

  useEffect(() => {
    let active = true
    fetchPaidModelFamilies().then((nextMatrix) => {
      if (!active) return
      setMatrix(nextMatrix)
      if (!nextMatrix.families.some((family) => family.id === selectedId)) {
        setSelectedId(nextMatrix.families[0]?.id ?? '')
      }
    })
    return () => { active = false }
  }, [selectedId])

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
  const detectedPrecisionRows = Object.values(selectedFamily?.localDetectedPrecisions ?? {})
    .reduce((sum, value) => sum + Number(value || 0), 0)

  return (
    <div className="paid-full-surface paid-family-matrix">
      <header className="paid-family-header">
        <div className="paid-product-lockup">
          <span className="paid-logo-orb"><ShieldCheck size={19} aria-hidden="true" /></span>
          <div>
            <strong>Model Family Support</strong>
            <small>{props.bootstrap.workspaceName} · code-indexed loader map</small>
          </div>
        </div>
        <div className="paid-family-search paid-search-field">
          <Search size={15} aria-hidden="true" />
          <input value={query} placeholder="Search families, quants, loaders, blockers..." onChange={(event) => setQuery(event.target.value)} />
        </div>
        <div className="paid-family-chip-row" aria-label="Family filters">
          {(['all', 'image', 'video', 'assistant', 'gaps'] as const).map((item) => (
            <button key={item} className={filter === item ? 'active' : ''} type="button" onClick={() => setFilter(item)}>
              <Filter size={13} aria-hidden="true" /> {item}
            </button>
          ))}
        </div>
      </header>

      <section className="paid-family-scoreboard" aria-label="Support overview">
        <article><span>Families</span><strong>{matrix.families.length}</strong><small>{filteredFamilies.length} visible</small></article>
        <article><span>Supported lanes</span><strong>{readyCount}</strong><small>ready or smoke-backed</small></article>
        <article><span>Open gaps</span><strong>{gapCount}</strong><small>blocked, metadata, or missing</small></article>
        <article><span>Local ledger rows</span><strong>{localRecordCount}</strong><small>{matrix.readiness?.error || 'readiness overlay'}</small></article>
      </section>

      <section className="paid-family-body">
        <aside className="paid-family-list" aria-label="Model families">
          {filteredFamilies.map((family) => (
            <button
              key={family.id}
              type="button"
              className={selectedFamily?.id === family.id ? 'paid-family-card active' : 'paid-family-card'}
              onClick={() => setSelectedId(family.id)}
            >
              <span className={`paid-family-status-dot ${familyStatusClass(family.status)}`} />
              <span className="paid-family-card-icon">{familyIcon(family)}</span>
              <strong>{family.label}</strong>
              <small>{family.category} · {family.status}</small>
              <em>{totalLocalRows(family)} local rows</em>
            </button>
          ))}
        </aside>

        {selectedFamily ? (
          <main className="paid-family-detail" aria-label={`${selectedFamily.label} support details`}>
            <div className="paid-family-detail-hero">
              <div>
                <span className="paid-eyebrow">{selectedFamily.category} family</span>
                <h2>{selectedFamily.label}</h2>
                <p>{selectedFamily.summary}</p>
              </div>
              <div className={`paid-family-big-status ${familyStatusClass(selectedFamily.status)}`}>
                <Zap size={16} aria-hidden="true" />
                {selectedFamily.status}
              </div>
            </div>

            <div className="paid-family-grid">
              <section className="paid-family-panel wide">
                <header><strong>Precision and quant support</strong><small>{detectedPrecisionRows} local precision hits</small></header>
                <div className="paid-precision-table" role="table">
                  <div role="row" className="head"><span>Precision</span><span>Status</span><span>Loader</span><span>Notes</span></div>
                  {selectedFamily.precisions.map((precision) => (
                    <div key={`${selectedFamily.id}-${precision.name}-${precision.status}`} role="row">
                      <strong>{precision.name}</strong>
                      <span className={`paid-status-pill ${precisionStatusClass(precision.status)}`}>{precision.status}</span>
                      <span>{precision.loader}</span>
                      <small>{precision.notes || 'current code path'}</small>
                    </div>
                  ))}
                </div>
              </section>

              <section className="paid-family-panel">
                <header><strong>Local readiness</strong><small>from pipeline_readiness</small></header>
                <div className="paid-family-meters">
                  {['working', 'metadata-only', 'unsupported-no-route', 'blocked-cleanly', 'broken-runtime'].map((status) => (
                    <div key={status}><span>{status}</span><strong>{statusTotal(selectedFamily, status)}</strong></div>
                  ))}
                </div>
              </section>

              <section className="paid-family-panel">
                <header><strong>Storage contracts</strong><small>accepted shapes</small></header>
                <div className="paid-family-token-wrap">
                  {selectedFamily.storage.map((item) => <span key={item}>{item}</span>)}
                </div>
              </section>

              <section className="paid-family-panel wide">
                <header><strong>Routes and loaders</strong><small>how the model is actually loaded</small></header>
                <div className="paid-route-list">
                  {selectedFamily.routes.map((route) => (
                    <article key={route.id}>
                      <strong>{route.id}</strong>
                      <span className={`paid-status-pill ${precisionStatusClass(route.status)}`}>{route.status}</span>
                      <small>{route.kind}</small>
                      <code>{route.entrypoint}</code>
                      {route.notes ? <p>{route.notes}</p> : null}
                    </article>
                  ))}
                </div>
              </section>

              <section className="paid-family-panel">
                <header><strong>Sidecars</strong><small>required companions</small></header>
                <div className="paid-family-token-wrap">
                  {selectedFamily.sidecars.map((item) => <span key={item}>{item}</span>)}
                </div>
              </section>

              <section className="paid-family-panel">
                <header><strong>LoRA policy</strong><small>adapter truth serum</small></header>
                <p>{selectedFamily.lora}</p>
              </section>

              <section className="paid-family-panel wide">
                <header><strong>Blockers and missing pieces</strong><small>do not expose as working</small></header>
                <ul className="paid-family-blockers">
                  {selectedFamily.blockers.map((item) => <li key={item}>{item}</li>)}
                </ul>
              </section>
            </div>
          </main>
        ) : null}

        <aside className="paid-family-evidence" aria-label="Source evidence">
          <section>
            <header><Database size={15} aria-hidden="true" /><strong>Precision vocabulary</strong></header>
            <div className="paid-family-token-wrap compact">
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
              {(matrix.blockedExamples ?? []).filter((item) => item.family === selectedFamily?.id).slice(0, 6).map((item) => (
                <li key={`${item.status}-${item.path}-${item.route}`}>
                  <strong>{item.status}</strong>
                  <span>{item.reason || item.route || item.path}</span>
                </li>
              ))}
              {(matrix.blockedExamples ?? []).filter((item) => item.family === selectedFamily?.id).length === 0 ? <li><span>No local blockers found for this family.</span></li> : null}
            </ul>
          </section>
        </aside>
      </section>
    </div>
  )
}
