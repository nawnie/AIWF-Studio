import type { ProModelOption } from '../../types'

/** Return only models accepted by Video Lab's Wan fast_5b continuation route. */
export function videoLabExtendModels(models: readonly ProModelOption[]): ProModelOption[] {
  return models.filter((model) => {
    if (model.engineId !== 'wan') return false
    if (model.checkpointPathStatus === 'missing') return false
    const routeStatus = model.routeStatus?.trim().toLowerCase().replace(/[\s_]+/g, '-')
    if (['blocked', 'missing-assets', 'blocked-runtime'].includes(routeStatus ?? '')) return false
    const status = model.status?.trim().toLowerCase().replace(/[\s_]+/g, '-')
    if (['blocked-cleanly', 'blocked-runtime', 'broken-runtime', 'unsupported-no-route', 'missing-assets', 'needs-snapshot', 'disabled'].includes(status ?? '')) return false
    const identity = `${model.id} ${model.name}`.toLowerCase().replace(/[^a-z0-9]+/g, ' ')
    return /(^|\s)ti2v(\s|$)/.test(identity) && /(^|\s)5b(\s|$)/.test(identity)
  })
}
