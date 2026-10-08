import { useEffect, useState } from 'react'
import { AlertCircle, Boxes, LoaderCircle } from 'lucide-react'
import { fetchCivitaiSupport } from './studioApiClient'
import type { CivitaiSupportCatalog } from './studioApiClient'
import './CivitaiSupportPanel.css'

export function CivitaiSupportPanel() {
  const [catalog, setCatalog] = useState<CivitaiSupportCatalog | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    let active = true
    fetchCivitaiSupport()
      .then((next) => {
        if (!active) return
        setCatalog(next)
        setError('')
      })
      .catch(() => {
        if (!active) return
        setCatalog(null)
        setError('Civitai support data could not be loaded.')
      })
    return () => { active = false }
  }, [])

  if (!catalog) {
    return (
      <section className="studio-civitai-support" aria-live="polite">
        <div className="studio-civitai-support-state">
          {error ? <AlertCircle size={18} aria-hidden="true" /> : <LoaderCircle size={18} className="studio-civitai-spin" aria-hidden="true" />}
          <span>{error || 'Loading resource route map…'}</span>
        </div>
      </section>
    )
  }

  return (
    <section className="studio-civitai-support" aria-label="Civitai resource capabilities">
      <header className="studio-civitai-support-heading">
        <div>
          <span className="studio-eyebrow"><Boxes size={14} aria-hidden="true" /> Civitai resource types</span>
          <h2>What Studio can generate and train today</h2>
          <p>A Civitai type describes a resource. It does not guarantee that the file is a standalone generator or trainable target.</p>
        </div>
        <small>Route map · {catalog.resources.length} categories</small>
      </header>

      {catalog.limitations.length ? (
        <ul className="studio-civitai-limitations">
          {catalog.limitations.map((item) => <li key={item}>{item}</li>)}
        </ul>
      ) : null}

      <div className="studio-civitai-grid" role="table" aria-label="Generation and training support by resource type">
        <div className="studio-civitai-row studio-civitai-row-head" role="row">
          <span role="columnheader">Civitai type</span>
          <span role="columnheader">Generate / use</span>
          <span role="columnheader">Train</span>
          <span role="columnheader">PC validation</span>
        </div>
        {catalog.resources.map((resource) => (
          <article className="studio-civitai-row" role="row" key={resource.id}>
            <strong role="cell">{resource.label}</strong>
            <p role="cell">{resource.generation}</p>
            <p role="cell">{resource.training}</p>
            <p role="cell">{resource.verification}</p>
          </article>
        ))}
      </div>
    </section>
  )
}
