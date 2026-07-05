import type { Dispatch, SetStateAction } from 'react'
import type {
  GenerationSettings,
  ProBootstrap,
  ProRuntimeStatus,
  RecentOutput,
  ProModelOption,
} from '../../types'

export interface PaidUserTab {
  id: string
  label: string
  icon?: string
  color?: string
  hidden?: boolean
  workspaceType?: 'empty' | 'iframe' | 'markdown' | 'tool' | 'custom'
  description?: string
}

export interface PaidWorkflowCodeBlock {
  id: string
  label: string
  kind: 'generation' | 'workflow' | 'qa' | 'export'
  nodeId: string
  source: string
  createdAt: string
  summary: string
  order: number
  classes: {
    requires: string[]
    produces: string[]
  }
  payload: Record<string, unknown>
  code: string
}

export interface PaidLayoutProps {
  settings: GenerationSettings
  bootstrap: ProBootstrap
  runtime: ProRuntimeStatus
  recentOutputs: RecentOutput[]
  preview: RecentOutput | null
  selectedModel: ProModelOption | undefined
  selectedModelName: string
  statusMessage: string
  isGenerating: boolean
  onSettingsChange: Dispatch<SetStateAction<GenerationSettings>>
  onGenerate: () => void
  onSendToWorkflow?: (source?: string) => void
  workflowBlocks?: PaidWorkflowCodeBlock[]
  onWorkflowBlocksChange?: Dispatch<SetStateAction<PaidWorkflowCodeBlock[]>>
  onPreviewSelect: (output: RecentOutput) => void
  onOpenModels: () => void
  onOpenSettings: () => void
}

export interface PaidTabsProps {
  paidTabs: PaidUserTab[]
  onPaidTabsChange: (tabs: PaidUserTab[]) => void
  onOpenTab?: (id: string) => void
}

export function stageTime(index: number): string {
  return `${Math.round((index + 1) * 3.7 * 10) / 10}s`
}

export function displayDate(value: string): string {
  if (!value) {
    return 'Just now'
  }
  try {
    const date = new Date(value)
    if (!Number.isNaN(date.getTime())) {
      return date.toLocaleString()
    }
  } catch {
    // Local display helper only.
  }
  return value
}

export function selectedImage(preview: RecentOutput | null, recentOutputs: RecentOutput[]): RecentOutput | null {
  return preview ?? recentOutputs[0] ?? null
}

export function safeTabId(label: string): string {
  const cleaned = label
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
  return cleaned || `workspace-${Date.now()}`
}

export function formatPercent(value: number | undefined): string {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return '0%'
  }
  return `${Math.max(0, Math.min(100, Math.round(value)))}%`
}
