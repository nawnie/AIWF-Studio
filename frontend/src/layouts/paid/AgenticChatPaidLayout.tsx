import { useEffect, useMemo, useState } from 'react'
import {
  Bot,
  Braces,
  CheckCircle2,
  Code2,
  FileJson,
  MessageSquare,
  Play,
  Plug,
  RefreshCcw,
  Send,
  ShieldCheck,
  Sparkles,
  Wrench,
} from 'lucide-react'
import type { PaidLayoutProps } from './PaidLayoutTypes'
import {
  fetchPaidAgentModels,
  fetchPaidAgentTools,
  streamPaidAgentChat,
} from './paidApiClient'
import type { PaidAgentMessage, PaidAgentModel, PaidAgentTool } from './paidApiClient'
import './paidLayouts.css'

const STARTER_MESSAGES: PaidAgentMessage[] = [
  {
    role: 'system',
    content:
      'AIWF Agent is local-first. It can plan, inspect, and draft patches, but destructive actions require human confirmation.',
  },
  {
    role: 'assistant',
    content:
      'Ready. Pick an Ollama model, enable tools, and ask for a plan, workflow JSON, code patch, or UI review.',
  },
]

const FALLBACK_TOOLS: PaidAgentTool[] = [
  { id: 'workflow-json', label: 'Workflow JSON', group: 'Studio', status: 'available', description: 'Create, inspect, and explain Pipeline Atlas workflow files.' },
  { id: 'prompt-refiner', label: 'Prompt Refiner', group: 'Create', status: 'available', description: 'Improve prompt structure and negative prompt coverage.' },
  { id: 'plugin-manager', label: 'Plugin Manager', group: 'Extensions', status: 'safe', description: 'Draft manifests and empty workspaces for community tabs.' },
  { id: 'log-viewer', label: 'Log Viewer', group: 'Runtime', status: 'read-only', description: 'Summarize logs and suggest next troubleshooting steps.' },
  { id: 'patch-draft', label: 'Patch Draft', group: 'Code', status: 'draft-only', description: 'Draft code diffs for user review.' },
]

export function AgenticChatPaidLayout({
  settings,
  runtime,
  selectedModelName,
  statusMessage,
  onSendToWorkflow,
}: PaidLayoutProps) {
  const [models, setModels] = useState<PaidAgentModel[]>([])
  const [tools, setTools] = useState<PaidAgentTool[]>(FALLBACK_TOOLS)
  const [selectedModel, setSelectedModel] = useState('')
  const [messages, setMessages] = useState<PaidAgentMessage[]>(STARTER_MESSAGES)
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [enabledTools, setEnabledTools] = useState<string[]>(['workflow-json', 'prompt-refiner', 'patch-draft'])
  const [activePanel, setActivePanel] = useState<'plan' | 'patch' | 'skills' | 'plugins'>('plan')
  const [connectionMessage, setConnectionMessage] = useState('Ollama backend not checked yet.')

  const visibleMessages = useMemo(() => messages.filter((message) => message.role !== 'system'), [messages])

  const refreshBackend = () => {
    setConnectionMessage('Checking Ollama at 127.0.0.1:11434...')
    fetchPaidAgentModels()
      .then((nextModels) => {
        setModels(nextModels)
        setSelectedModel((current) => current || nextModels[0]?.id || '')
        setConnectionMessage(nextModels.length ? `Loaded ${nextModels.length} Ollama model(s).` : 'Ollama answered but returned no models.')
      })
      .catch((error: unknown) => {
        setConnectionMessage(error instanceof Error ? error.message : 'Ollama is unavailable.')
      })
    fetchPaidAgentTools().then((nextTools) => {
      if (nextTools.length) {
        setTools(nextTools)
      }
    }).catch(() => undefined)
  }

  useEffect(() => {
    refreshBackend()
  }, [])

  const send = async () => {
    const content = input.trim()
    if (!content || busy) {
      return
    }
    const nextMessages: PaidAgentMessage[] = [...messages, { role: 'user', content }]
    setMessages(nextMessages)
    setInput('')
    setBusy(true)
    try {
      setMessages([...nextMessages, { role: 'assistant', content: '' }])
      const reply = await streamPaidAgentChat(selectedModel, nextMessages, enabledTools, (partial) => {
        setMessages([...nextMessages, { role: 'assistant', content: partial }])
      })
      setMessages([...nextMessages, { role: 'assistant', content: reply }])
    } catch (error) {
      const detail = error instanceof Error ? error.message : 'Agent request failed.'
      setMessages([...nextMessages, { role: 'assistant', content: `Ollama request failed: ${detail}` }])
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="paid-agent paid-full-surface" aria-label="Advanced agentic chat paid layout">
      <aside className="paid-agent-left">
        <div className="paid-product-lockup compact">
          <span className="paid-logo-orb">A</span>
          <div>
            <strong>AIWF Agent</strong>
            <small>Ollama · tools · skills · plugins</small>
          </div>
        </div>
        <section className="paid-agent-card">
          <header><Bot size={16} /><strong>Backend Loader</strong></header>
          <label className="paid-field-mini">Model
            <select value={selectedModel} onChange={(event) => setSelectedModel(event.target.value)}>
              {models.length ? models.map((model) => <option key={model.id} value={model.id}>{model.name}</option>) : <option value="">No Ollama model loaded</option>}
            </select>
          </label>
          <button type="button" className="paid-wide-button" onClick={refreshBackend}><RefreshCcw size={14} /> Refresh Ollama</button>
          <small>{connectionMessage}</small>
        </section>
        <section className="paid-agent-card">
          <header><Wrench size={16} /><strong>Tool Permissions</strong></header>
          {tools.map((tool) => (
            <label key={tool.id} className="paid-tool-toggle">
              <input
                type="checkbox"
                checked={enabledTools.includes(tool.id)}
                onChange={(event) => {
                  setEnabledTools((current) => event.target.checked ? [...current, tool.id] : current.filter((id) => id !== tool.id))
                }}
              />
              <span>
                <strong>{tool.label}</strong>
                <small>{tool.status} · {tool.description}</small>
              </span>
            </label>
          ))}
        </section>
      </aside>

      <main className="paid-agent-main">
        <header className="paid-agent-header">
          <div>
            <span className="paid-eyebrow">ADVANCED AGENTIC CHAT</span>
            <strong>Plan, inspect, draft, and use AIWF tools safely</strong>
            <small>{runtime.state} · {selectedModelName} · {statusMessage}</small>
          </div>
          <div className="paid-agent-mode-tabs">
            {[
              ['plan', Sparkles],
              ['patch', Code2],
              ['skills', Braces],
              ['plugins', Plug],
            ].map(([id, Icon]) => {
              const TabIcon = Icon as typeof Sparkles
              return <button key={id as string} type="button" className={activePanel === id ? 'active' : ''} onClick={() => setActivePanel(id as typeof activePanel)}><TabIcon size={15} />{id as string}</button>
            })}
          </div>
        </header>

        <section className="paid-agent-chat">
          {visibleMessages.map((message, index) => (
            <article key={`${message.role}-${index}`} className={`paid-chat-bubble ${message.role}`}>
              <span>{message.role === 'assistant' ? <Bot size={16} /> : <MessageSquare size={16} />}</span>
              <p>{message.content}</p>
            </article>
          ))}
        </section>

        <footer className="paid-agent-composer">
          <textarea
            value={input}
            rows={3}
            placeholder="Ask the local agent to plan a workflow, draft a patch, inspect logs, or build a plugin tab..."
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
                void send()
              }
            }}
          />
          <button type="button" className="paid-run-button" onClick={send} disabled={busy || !input.trim()}><Send size={16} /> {busy ? 'Thinking...' : 'Send'}</button>
          <button type="button" className="paid-wide-button" onClick={() => onSendToWorkflow?.('Agentic Chat prompt')}>Send to workflow</button>
        </footer>
      </main>

      <aside className="paid-agent-right">
        <section className="paid-agent-card big">
          <header><ShieldCheck size={16} /><strong>Guardrails</strong></header>
          <ul>
            <li>Read, inspect, and draft by default.</li>
            <li>No destructive repo or file actions without confirmation.</li>
            <li>Plugin tabs run as declared workspaces, not hidden code.</li>
            <li>Ollama remains local unless the user changes its URL.</li>
          </ul>
        </section>
        <section className="paid-agent-card big">
          <header>{activePanel === 'patch' ? <Code2 size={16} /> : activePanel === 'plugins' ? <Plug size={16} /> : <FileJson size={16} />}<strong>{activePanel.toUpperCase()} Workspace</strong></header>
          {activePanel === 'plan' ? (
            <div className="paid-agent-plan">
              <span><CheckCircle2 size={14} /> Understand request</span>
              <span><CheckCircle2 size={14} /> Check tool permissions</span>
              <span><CheckCircle2 size={14} /> Draft steps</span>
              <span><Play size={14} /> Wait for user approval</span>
            </div>
          ) : activePanel === 'patch' ? (
            <pre>{`// Draft-only patch lane\n// Ask: "create a patch for Pipeline Atlas JSON save"\n// The agent should return a reviewed diff, not silently write files.`}</pre>
          ) : activePanel === 'skills' ? (
            <div className="paid-skill-grid">
              {['repo review', 'workflow authoring', 'prompt tuning', 'log triage', 'plugin manifest'].map((skill) => <span key={skill}>{skill}</span>)}
            </div>
          ) : (
            <pre>{`{\n  "id": "my-empty-tab",\n  "label": "My Workspace",\n  "workspaceType": "empty",\n  "entry": "/api/pro/extensions/workspaces/my-empty-tab"\n}`}</pre>
          )}
        </section>
        <section className="paid-agent-card big">
          <header><Sparkles size={16} /><strong>Current Context</strong></header>
          <small>Prompt</small>
          <p>{settings.prompt || 'No active prompt yet.'}</p>
          <small>Enabled tools</small>
          <p>{enabledTools.join(', ') || 'none'}</p>
        </section>
      </aside>
    </div>
  )
}
