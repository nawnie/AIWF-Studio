const LOOPBACK_HOSTS = new Set(['localhost', '127.0.0.1', '[::1]', '::1'])

export function resolveLoopbackApiBase(rawValue: string | undefined, variableName = 'VITE_AIWF_API_BASE'): string {
  const value = (rawValue ?? '').trim()
  if (!value) {
    return ''
  }

  let parsed: URL
  try {
    parsed = new URL(value)
  } catch {
    throw new Error(`${variableName} must be an absolute HTTP URL for a loopback host.`)
  }
  if (!['http:', 'https:'].includes(parsed.protocol) || !LOOPBACK_HOSTS.has(parsed.hostname)) {
    throw new Error(`${variableName} is restricted to localhost, 127.0.0.1, or ::1.`)
  }
  if (parsed.username || parsed.password || parsed.search || parsed.hash) {
    throw new Error(`${variableName} cannot contain credentials, a query string, or a fragment.`)
  }
  return parsed.toString().replace(/\/$/, '')
}
