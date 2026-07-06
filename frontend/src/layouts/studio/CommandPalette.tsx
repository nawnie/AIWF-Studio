import { useEffect, useMemo, useState } from 'react'
import { Command, GitBranch, Image, Library, MessageSquare, Play, Settings, Sparkles, Volume2 } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import './studioLayouts.css'

interface CommandItem {
  id: string
  label: string
  hint: string
  group: string
  icon: LucideIcon
  action: () => void
}

export function CommandPalette({
  open,
  onOpenChange,
  onNavigate,
  onGenerate,
}: {
  open: boolean
  onOpenChange: (value: boolean) => void
  onNavigate: (rail: string) => void
  onGenerate: () => void
}) {
  const [query, setQuery] = useState('')
  const commands = useMemo<CommandItem[]>(() => [
    { id: 'run', label: 'Run current generation', hint: 'Submit the current prompt or workflow', group: 'Create', icon: Play, action: onGenerate },
    { id: 'pipeline', label: 'Open Pipeline Atlas', hint: 'Stage graph with type-safe nodes', group: 'Workspaces', icon: GitBranch, action: () => onNavigate('pipeline') },
    { id: 'foundry', label: 'Open Media Foundry', hint: 'Image canvas, layers, scenes, and tracks', group: 'Workspaces', icon: Image, action: () => onNavigate('foundry') },
    { id: 'audio', label: 'Open Audio Studio', hint: 'Waveform, mixer, stems, and audio prompts', group: 'Workspaces', icon: Volume2, action: () => onNavigate('audio') },
    { id: 'project', label: 'Open Media Center', hint: 'Project, assets, queue, export, QA, plugins', group: 'Workspaces', icon: Library, action: () => onNavigate('project') },
    { id: 'agent', label: 'Open Agentic Chat', hint: 'Ollama-backed project assistant', group: 'Agent', icon: MessageSquare, action: () => onNavigate('agent') },
    { id: 'settings', label: 'Open Settings Arsenal', hint: 'A1111-style settings density', group: 'System', icon: Settings, action: () => onNavigate('settings') },
    { id: 'extensions', label: 'Open Extensions Hub', hint: 'Create empty tabs and manage community workspaces', group: 'Community', icon: Sparkles, action: () => onNavigate('extensions') },
  ], [onGenerate, onNavigate])
  const filtered = commands.filter((item) => `${item.label} ${item.hint} ${item.group}`.toLowerCase().includes(query.toLowerCase().trim()))

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault()
        onOpenChange(!open)
      }
      if (event.key === 'Escape') {
        onOpenChange(false)
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [onOpenChange, open])

  if (!open) {
    return null
  }

  return (
    <div className="studio-command-backdrop" role="dialog" aria-modal="true" aria-label="Command palette" onMouseDown={() => onOpenChange(false)}>
      <div className="studio-command-panel" onMouseDown={(event) => event.stopPropagation()}>
        <label className="studio-command-input"><Command size={18} /><input autoFocus value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search actions, workspaces, tools..." /></label>
        <div className="studio-command-list">
          {filtered.map((item) => {
            const Icon = item.icon
            return <button type="button" key={item.id} onClick={() => { item.action(); onOpenChange(false) }}><Icon size={16} /><div><strong>{item.label}</strong><small>{item.group} · {item.hint}</small></div></button>
          })}
        </div>
        <footer>Ctrl+K opens this palette. Escape closes it. Tiny robot says: fewer hunts, more clicks that mean something.</footer>
      </div>
    </div>
  )
}
