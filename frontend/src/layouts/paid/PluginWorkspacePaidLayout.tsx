import { FileJson, Plug, Settings2, Sparkles } from 'lucide-react'
import type { PaidUserTab } from './PaidLayoutTypes'
import './paidLayouts.css'

export function PluginWorkspacePaidLayout({ tab, onOpenExtensions }: { tab: PaidUserTab; onOpenExtensions: () => void }) {
  return (
    <div className="paid-plugin-workspace paid-full-surface" aria-label={`${tab.label} plugin workspace`}>
      <main>
        <section className="paid-plugin-hero">
          <span className="paid-logo-orb"><Plug size={28} /></span>
          <div>
            <span className="paid-eyebrow">COMMUNITY WORKSPACE</span>
            <h1>{tab.label}</h1>
            <p>{tab.description || 'This is an empty tab created by the extension system. Add plugin UI here when the community package is ready.'}</p>
          </div>
        </section>
        <section className="paid-plugin-grid">
          <article>
            <FileJson size={20} />
            <strong>Manifest</strong>
            <p>Declare id, label, icon, workspace type, permissions, and entrypoint.</p>
          </article>
          <article>
            <Settings2 size={20} />
            <strong>Settings</strong>
            <p>Users can hide, show, rename, or remove community tabs.</p>
          </article>
          <article>
            <Sparkles size={20} />
            <strong>Future UI</strong>
            <p>Drop a real React workspace here later without changing the backend contract.</p>
          </article>
        </section>
        <button type="button" className="paid-run-button" onClick={onOpenExtensions}>Open Extension Manager</button>
      </main>
      <aside>
        <span className="paid-eyebrow">Workspace JSON</span>
        <pre>{JSON.stringify(tab, null, 2)}</pre>
      </aside>
    </div>
  )
}
