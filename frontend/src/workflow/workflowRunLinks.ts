/** Only local app paths and explicit loopback URLs are rendered as run links. */
export function safeWorkflowHref(href: string, apiBase = ''): string | null {
  const value = href.trim()
  if ([...href].some((character) => character.charCodeAt(0) <= 0x1f || character.charCodeAt(0) === 0x7f) || value.includes('\\') || value.startsWith('//')) return null
  if (value.startsWith('/')) return `${apiBase}${value}`
  try {
    const parsed = new URL(value)
    if (!['http:', 'https:'].includes(parsed.protocol) || parsed.username || parsed.password) return null
    if (!['localhost', '127.0.0.1', '[::1]'].includes(parsed.hostname)) return null
    return parsed.href
  } catch {
    return null
  }
}
