import type { AgentModel } from './studioApiClient'

export function formatAgentModelLabel(selectedModelId: string, models: AgentModel[]): string {
  const selected = models.find((model) => model.id === selectedModelId)
  return selected?.name || selectedModelId || 'No Ollama model selected'
}
