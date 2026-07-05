import { useMemo, useState } from 'react'
import {
  Eye,
  EyeOff,
  LayoutGrid,
  Plus,
  Plug,
  Save,
  Settings2,
  Sparkles,
  Trash2,
} from 'lucide-react'
import type { PaidTabsProps, PaidUserTab } from './PaidLayoutTypes'
import { safeTabId } from './PaidLayoutTypes'
import { savePaidUserTabs } from './paidApiClient'
import './paidLayouts.css'

const EMPTY_TAB_TEMPLATE: PaidUserTab = {
  id: 'plugin-my-workspace',
  label: 'My Workspace',
  icon: 'grid',
  color: '#8b5cf6',
  hidden: false,
  workspaceType: 'empty',
  description: 'Empty community workspace tab. Replace later with plugin UI.',
}

export function ExtensionsHubPaidLayout({ paidTabs, onPaidTabsChange, onOpenTab }: PaidTabsProps) {
  const [draftLabel, setDraftLabel] = useState('My Workspace')
  const visibleCount = useMemo(() => paidTabs.filter((tab) => !tab.hidden).length, [paidTabs])
  const persist = (tabs: PaidUserTab[]) => {
    onPaidTabsChange(tabs)
    void savePaidUserTabs(tabs)
  }

  const addEmptyTab = () => {
    const label = draftLabel.trim() || 'My Workspace'
    const id = `plugin-${safeTabId(label)}`
    const tab: PaidUserTab = {
      ...EMPTY_TAB_TEMPLATE,
      id,
      label,
      description: `Empty workspace created for ${label}.`,
    }
    const nextTabs = [...paidTabs.filter((item) => item.id !== id), tab]
    persist(nextTabs)
    onOpenTab?.(id)
  }

  return (
    <div className="paid-extensions paid-full-surface" aria-label="Extensions hub paid layout">
      <aside className="paid-extension-sidebar">
        <div className="paid-product-lockup compact">
          <span className="paid-logo-orb">A</span>
          <div>
            <strong>AIWF Extensions</strong>
            <small>Community workspaces · PAID v3</small>
          </div>
        </div>
        <section className="paid-agent-card">
          <header><Plus size={16} /><strong>Create Empty Tab</strong></header>
          <label className="paid-field-mini">Tab label
            <input value={draftLabel} onChange={(event) => setDraftLabel(event.target.value)} />
          </label>
          <button type="button" className="paid-wide-button" onClick={addEmptyTab}><Plus size={14} /> Add to Left Bar</button>
          <small>Tabs are scrollable in the rail and can be hidden below.</small>
        </section>
        <section className="paid-agent-card">
          <header><Sparkles size={16} /><strong>API Surface</strong></header>
          <code>/api/pro/extensions/tabs</code>
          <code>/api/pro/extensions/register-tab</code>
          <code>/api/pro/extensions/workspaces/:id</code>
          <code>/api/pro/agent/chat</code>
        </section>
      </aside>

      <main className="paid-extension-main">
        <header className="paid-agent-header">
          <div>
            <span className="paid-eyebrow">COMMUNITY EXTENSIONS</span>
            <strong>Tabs, empty workspaces, manifests, skills, and plugin settings</strong>
            <small>{paidTabs.length} registered · {visibleCount} visible in the left bar</small>
          </div>
          <button type="button" className="paid-run-button" onClick={() => void savePaidUserTabs(paidTabs)}><Save size={15} /> Save Registry</button>
        </header>

        <section className="paid-extension-grid">
          {paidTabs.length ? paidTabs.map((tab) => (
            <article key={tab.id} className={tab.hidden ? 'hidden' : ''}>
              <header>
                <span style={{ background: tab.color || '#8b5cf6' }}><Plug size={16} /></span>
                <div>
                  <strong>{tab.label}</strong>
                  <small>{tab.id}</small>
                </div>
              </header>
              <p>{tab.description || 'Empty extension workspace.'}</p>
              <div className="paid-extension-actions">
                <button type="button" onClick={() => onOpenTab?.(tab.id)}><LayoutGrid size={14} /> Open</button>
                <button type="button" onClick={() => persist(paidTabs.map((item) => item.id === tab.id ? { ...item, hidden: !item.hidden } : item))}>
                  {tab.hidden ? <Eye size={14} /> : <EyeOff size={14} />}
                  {tab.hidden ? 'Show' : 'Hide'}
                </button>
                <button type="button" onClick={() => persist(paidTabs.filter((item) => item.id !== tab.id))}><Trash2 size={14} /> Remove</button>
              </div>
            </article>
          )) : (
            <article className="paid-empty-extension-card">
              <header><span><Plug size={16} /></span><div><strong>No community tabs yet</strong><small>Create one from the sidebar.</small></div></header>
              <p>This creates a real empty workspace entry in the left rail, ready for plugin UI later.</p>
            </article>
          )}
        </section>
      </main>

      <aside className="paid-extension-inspector">
        <header><Settings2 size={16} /><strong>Extension Policy</strong></header>
        <section>
          <span className="paid-eyebrow">User Friendly Rules</span>
          <ul>
            <li>Added tabs appear in the left rail automatically.</li>
            <li>Rail remains scrollable as the community grows.</li>
            <li>Users can hide tabs in Settings or here.</li>
            <li>Empty tabs are safe defaults: no hidden execution.</li>
          </ul>
        </section>
        <section>
          <span className="paid-eyebrow">Manifest Example</span>
          <pre>{`{\n  "id": "plugin-my-workspace",\n  "label": "My Workspace",\n  "workspaceType": "empty",\n  "rail": { "visible": true }\n}`}</pre>
        </section>
      </aside>
    </div>
  )
}
